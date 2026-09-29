"""Pulse schedule of the benchmark circuit, transcribed from JAQSI.

JAQSI implements a pulse-level gate as a sequence of time evolutions
$\\mathrm{d}U/\\mathrm{d}t = -i H(t) U$, one per basis gate of the gate's
decomposition.  This module restates that sequence for the benchmark circuit
(a Hadamard layer followed by $CRX$ rings) as plain matrices and
coefficient callables, so that every simulator adapter integrates the identical
ODE sequence rather than its own pulse model.

The transcription mirrors ``jaqsi.pulses`` with the rotating-wave approximation
enabled and either the ``gaussian`` or the ``drag`` envelope at its calibrated
defaults, which are read from jaqsi.  Under the RWA the carrier drops out of the
coefficients, leaving

$$ H(t) = \\tfrac{1}{2}\\,w\\,\\bigl(E(t)\\,P + Q(t)\\,P_\\perp\\bigr), \\qquad
   E(t) = A e^{-(t - T/2)^2/(2\\sigma^2)}, \\qquad
   Q(t) = -\\beta \\dot{E}(t) = \\frac{\\beta (t - T/2)}{\\sigma^2} E(t) $$

for the driven $RY$ ($P = Y$) of duration $T$, whose envelope is centred at the
pulse midpoint and whose quadrature $Q$ drives $P_\\perp = -X$ for ``drag`` and
vanishes for ``gaussian``, and a constant $H$ for the virtual $RZ$, the $CZ$
coupling and the Hadamard correction phase.  Only numpy is imported at module
level so the module stays usable without the optional pulse backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from benchmark.circuits import CircuitSpec
from benchmark.config import PULSE_ENVELOPES

# Calibrated scale factors of the constant-coefficient gates.  Both act over a
# unit time span, so $RZ(w)$ evolves under $\frac{w}{2} Z$ and $CZ$ under
# $0.3183\\pi\\,H_{CZ} \approx \pi\\,\\lvert 11\rangle\\langle 11\rvert$.
RZ_SCALE: float = 0.5
CZ_SCALE: float = 0.3183098783513154

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_ID = np.eye(2, dtype=complex)

# $H_{CZ} = \frac{\pi}{4}(II - ZI - IZ + ZZ)$
H_CZ = (np.pi / 4) * (
    np.kron(_ID, _ID)
    - np.kron(PAULI_Z, _ID)
    - np.kron(_ID, PAULI_Z)
    + np.kron(PAULI_Z, PAULI_Z)
)

# Correction phase of the Hadamard decomposition, applied with coefficient -1
# so that the segment contributes a global phase $i$.
H_CORRECTION = (np.pi / 2) * _ID


@dataclass(frozen=True)
class Segment:
    """One time evolution of the pulse schedule.

    The Hamiltonian is ``coeff(t) * op`` acting on ``wires``, plus
    ``quad_coeff(t) * quad_op`` for a DRAG rotation, integrated from $0$ to
    ``duration``.  ``angle_fn`` maps the circuit's input parameter vector to
    the segment's scale factor; ``envelope`` holds the envelope parameters of a
    driven rotation and is ``None`` for the constant-coefficient gates.
    """

    op: np.ndarray
    wires: Tuple[int, ...]
    duration: float
    angle_fn: Callable[[np.ndarray], float]
    envelope: Optional[Tuple[float, ...]] = None
    quad_op: Optional[np.ndarray] = None

    @property
    def ops(self) -> Tuple[np.ndarray, ...]:
        """Operators of the Hamiltonian terms, the DRAG quadrature last."""
        return (self.op,) if self.quad_op is None else (self.op, self.quad_op)


@dataclass(frozen=True)
class Channel:
    """Depolarizing channel of probability ``p`` on ``wire``.

    Sits between the segments of two gates, where the ``noise`` mode's circuit
    spec places it, which is also where jaqsi's pulse gates apply their noise.
    """

    wire: int
    p: float


def ry_envelope(envelope: str) -> Tuple[Tuple[float, ...], float]:
    """Return the calibrated envelope parameters of the driven $RY$ and its duration.

    Read from ``PulseEnvelope.REGISTRY`` in jaqsi: $(A, \\sigma)$ for the
    Gaussian and $(A, \\beta, \\sigma)$ for DRAG.
    """
    from jaqsi.pulses import PulseEnvelope

    *params, duration = (
        float(x) for x in PulseEnvelope.get(envelope)["defaults"]["RY"]
    )
    return tuple(params), duration


def in_phase_env(t, envelope: Tuple[float, ...], duration: float, xp=np):
    """Evaluate the in-phase envelope $E(t)$ of a driven rotation.

    Matches ``PulseEnvelope.gaussian`` and ``PulseEnvelope.drag``, which are
    the same Gaussian, centred at the midpoint of the pulse of length
    *duration*.  Pass ``xp=jnp`` for the JAX-based backends.
    """
    amplitude, sigma = envelope[0], envelope[-1]
    return amplitude * xp.exp(-((t - duration / 2) ** 2) / (2.0 * sigma**2))


def quadrature_env(t, envelope: Tuple[float, float, float], duration: float, xp=np):
    """Evaluate the DRAG quadrature envelope $Q(t) = -\\beta \\dot{E}(t)$.

    Matches ``PulseEnvelope.drag_quadrature``.
    """
    _, beta, sigma = envelope
    offset = t - duration / 2
    return beta * offset / sigma**2 * in_phase_env(t, envelope, duration, xp)


def make_coeff_fns(segment: Segment, params, xp=np) -> List[Callable]:
    """Return the coefficients $c(t)$ of the terms of *segment* at *params*.

    The list follows ``segment.ops``.  Constant-coefficient segments return
    their scale factor unchanged, so the callables are valid for every segment
    type.
    """
    angle = segment.angle_fn(params)
    if segment.envelope is None:
        return [lambda t: angle]

    envelope, duration = segment.envelope, segment.duration

    def coeff(t):
        return 0.5 * in_phase_env(t, envelope, duration, xp) * angle

    def quad_coeff(t):
        return 0.5 * quadrature_env(t, envelope, duration, xp) * angle

    return [coeff] if segment.quad_op is None else [coeff, quad_coeff]


def embed(op: np.ndarray, wires: Tuple[int, ...], n_qubits: int, xp=np):
    """Embed *op* into the full register, qubit $0$ being the most significant.

    Kronecker-multiplies *op* with the identity on the remaining qubits and
    permutes the result into ascending wire order, matching the big-endian
    basis convention used by JAQSI and PennyLane.
    """
    rest = [q for q in range(n_qubits) if q not in wires]
    full = xp.kron(op, xp.eye(2 ** len(rest), dtype=op.dtype)) if rest else op

    order = list(wires) + rest
    perm = [order.index(q) for q in range(n_qubits)]
    tensor = full.reshape([2] * (2 * n_qubits))
    tensor = tensor.transpose(perm + [p + n_qubits for p in perm])
    return tensor.reshape(2**n_qubits, 2**n_qubits)


def apply_local(u, psi, wires: Tuple[int, ...], n_qubits: int, xp=np):
    """Apply the local unitary *u* on *wires* to the statevector *psi*.

    Contracts *u* with the corresponding axes of *psi* instead of embedding it
    into the full register, so the caller only ever integrates a $2 \\times 2$
    or $4 \\times 4$ ODE.  Equivalent to ``embed(u, wires, n_qubits) @ psi``
    under the same big-endian convention.
    """
    k = len(wires)
    tensor = psi.reshape([2] * n_qubits)
    contracted = xp.tensordot(
        u.reshape([2] * (2 * k)),
        tensor,
        axes=(list(range(k, 2 * k)), list(wires)),
    )

    # tensordot leaves the wire axes in front, followed by the untouched ones.
    rest = [q for q in range(n_qubits) if q not in wires]
    order = list(wires) + rest
    perm = [order.index(q) for q in range(n_qubits)]
    return contracted.transpose(perm).reshape(2**n_qubits)


def _contract_axes(op, tensor, axes: List[int], xp=np):
    """Contract the $k$-qubit operator tensor *op* into *axes* of *tensor*."""
    k = len(axes)
    contracted = xp.tensordot(op, tensor, axes=(list(range(k, 2 * k)), axes))

    # tensordot leaves the contracted axes in front, followed by the others.
    rest = [a for a in range(tensor.ndim) if a not in axes]
    order = axes + rest
    return contracted.transpose([order.index(a) for a in range(tensor.ndim)])


def apply_local_density(u, rho, wires: Tuple[int, ...], n_qubits: int, xp=np):
    """Return $U \\rho U^\\dagger$ for the local unitary *u* on *wires*.

    The density-matrix counterpart of :func:`apply_local`: *u* acts on the ket
    axes of *rho* and its conjugate on the bra axes, under the same big-endian
    convention.
    """
    k = len(wires)
    u = u.reshape([2] * (2 * k))
    tensor = rho.reshape([2] * (2 * n_qubits))
    tensor = _contract_axes(u, tensor, list(wires), xp)
    tensor = _contract_axes(xp.conj(u), tensor, [w + n_qubits for w in wires], xp)
    return tensor.reshape(2**n_qubits, 2**n_qubits)


def depolarize(rho, p: float, wire: int, n_qubits: int, xp=np):
    """Apply the depolarizing channel of probability *p* on *wire* to *rho*.

    $\\rho \\mapsto (1 - p)\\rho + \\frac{p}{3}(X\\rho X + Y\\rho Y + Z\\rho Z)$,
    the Kraus form jaqsi, PennyLane and Qulacs use.
    """
    flipped = sum(
        apply_local_density(xp.asarray(pauli), rho, (wire,), n_qubits, xp)
        for pauli in (PAULI_X, PAULI_Y, PAULI_Z)
    )
    return (1 - p) * rho + (p / 3) * flipped


def _pauli_z_expvals(probs, n_qubits: int, xp=np):
    """Return per-qubit $\\langle Z \\rangle$ from a big-endian probability vector."""
    reshaped = probs.reshape([2] * n_qubits)
    return xp.stack(
        [
            xp.sum(reshaped, axis=tuple(j for j in range(n_qubits) if j != i))
            @ xp.array([1.0, -1.0])
            for i in range(n_qubits)
        ]
    )


def project_state(psi, mode: str, n_qubits: int, xp=np):
    """Map the final statevector *psi* onto the requested measurement mode.

    Shared by the adapters whose backend solves for a statevector only; the
    pulse schedule is unitary, so the density matrix is the pure-state
    projector.
    """
    if mode == "state":
        return psi
    if mode == "probs":
        return xp.abs(psi) ** 2
    if mode == "density":
        return xp.outer(psi, xp.conj(psi))
    if mode == "expval":
        return _pauli_z_expvals(xp.abs(psi) ** 2, n_qubits, xp)
    raise ValueError(f"Unsupported mode: {mode!r}")


def _rz(angle_fn: Callable[[np.ndarray], float], wire: int) -> Segment:
    """Virtual $RZ$: constant $H = \\frac{w}{2} Z$ over a unit time span."""
    return Segment(
        op=PAULI_Z,
        wires=(wire,),
        duration=1.0,
        angle_fn=lambda params: RZ_SCALE * angle_fn(params),
    )


def _ry(
    angle_fn: Callable[[np.ndarray], float], wire: int, envelope: str
) -> Segment:
    """Driven $RY$ rotation with the calibrated *envelope*.

    The DRAG quadrature drives $-X$, the axis the carrier phase $\\pi/2$ of
    $RY$ rotates the quadrature onto.
    """
    params, duration = ry_envelope(envelope)
    return Segment(
        op=PAULI_Y,
        wires=(wire,),
        duration=duration,
        angle_fn=angle_fn,
        envelope=params,
        quad_op=-PAULI_X if envelope == "drag" else None,
    )


def _cz(control: int, target: int) -> Segment:
    """$CZ$ coupling: constant $H = 0.3183\\pi\\,H_{CZ}$ over a unit time span."""
    return Segment(
        op=H_CZ,
        wires=(control, target),
        duration=1.0,
        angle_fn=lambda params: CZ_SCALE * np.pi,
    )


def _correction(wire: int) -> Segment:
    """Hadamard correction phase: constant $H = -\\frac{\\pi}{2} I$."""
    return Segment(
        op=H_CORRECTION,
        wires=(wire,),
        duration=1.0,
        angle_fn=lambda params: -1.0,
    )


def _hadamard(wire: int, envelope: str) -> List[Segment]:
    """Hadamard as $RZ(\\pi)$, $RY(\\frac{\\pi}{2})$ and the correction phase."""
    return [
        _rz(lambda params: np.pi, wire),
        _ry(lambda params: np.pi / 2, wire, envelope),
        _correction(wire),
    ]


def _cnot(control: int, target: int, envelope: str) -> List[Segment]:
    """$CX$ as $H$, $CZ$, $H$ on the target."""
    return [
        *_hadamard(target, envelope),
        _cz(control, target),
        *_hadamard(target, envelope),
    ]


def _crx(control: int, target: int, index: int, envelope: str) -> List[Segment]:
    """$CRX(\\phi)$ as two $CX$ interleaved with $RY(\\pm\\frac{\\phi}{2})$.

    *index* is the position of this gate's angle within the circuit's input
    parameter vector.
    """
    return [
        _rz(lambda params: np.pi / 2, target),
        _ry(lambda params: params[index] / 2, target, envelope),
        *_cnot(control, target, envelope),
        _ry(lambda params: -params[index] / 2, target, envelope),
        *_cnot(control, target, envelope),
        _rz(lambda params: -np.pi / 2, target),
    ]


def build_schedule(
    spec: CircuitSpec, envelope: str = "gaussian"
) -> List[Union[Segment, Channel]]:
    """Return the pulse schedule of *spec* in execution order.

    *envelope* is the jaqsi pulse envelope of the driven rotations,
    ``gaussian`` or ``drag``.  Only the Hadamard and $CRX$ decompositions are
    transcribed, so the ``crx_ring`` family is the only one with a pulse-level
    counterpart: its Hadamard layer contributes $3 n$ segments and every $CRX$
    ring another $18 n$, i.e. $21 n$ at a depth of one.  A noisy spec adds a
    :class:`Channel` wherever it holds a ``DEPOL`` op, after the segments of
    the gate before it.
    """
    if envelope not in PULSE_ENVELOPES:
        raise ValueError(
            f"Envelope {envelope!r} has no pulse transcription. "
            f"Available: {PULSE_ENVELOPES}"
        )
    segments: List[Union[Segment, Channel]] = []
    for op in spec.ops:
        if op.gate == "H":
            segments.extend(_hadamard(op.wires[0], envelope))
        elif op.gate == "CRX":
            segments.extend(_crx(op.wires[0], op.wires[1], op.index, envelope))
        elif op.gate == "DEPOL":
            segments.append(Channel(op.wires[0], spec.depolarizing))
        else:
            raise ValueError(f"Gate {op.gate!r} has no pulse transcription")
    return segments
