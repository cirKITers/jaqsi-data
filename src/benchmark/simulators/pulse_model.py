"""Pulse schedule of the benchmark circuit, transcribed from JAQSI.

JAQSI implements a pulse-level gate as a sequence of time evolutions
$\\mathrm{d}U/\\mathrm{d}t = -i H(t) U$, one per basis gate of the gate's
decomposition.  This module restates that sequence for the benchmark circuit
(a Hadamard layer followed by $CRX$ rings) as plain matrices and
coefficient callables, so that every simulator adapter integrates the identical
ODE sequence rather than its own pulse model.

The transcription mirrors ``jaqsi.pulses`` with the shipped defaults,
i.e. the ``drag`` envelope with the rotating-wave approximation enabled.  Under
the RWA the carrier drops out of the coefficients, leaving

$$ H(t) = \\tfrac{1}{2}\\,\\Omega(t)\\,w\\,P, \\qquad
   \\Omega(t) = A e^{-t^2/(8\\sigma^2)}\\left(1 - \\frac{\\beta t}{2\\sigma^2}\\right) $$

for the driven rotations ($P \\in \\{X, Y\\}$) and a constant $H$ for the
virtual $RZ$, the $CZ$ coupling and the Hadamard correction phase.  Only numpy
is imported here so the module stays usable without the optional pulse
backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from benchmark.circuits import CircuitSpec

# Calibrated drag parameters $(A, \beta, \sigma)$ and gate durations, taken
# from ``PulseEnvelope.REGISTRY`` in jaqsi.
RX_DRAG: Tuple[float, float, float] = (
    0.326562746114197,
    0.4002767596709071,
    5.3228107728890315,
)
RX_DURATION: float = 3.141300761986467
RY_DRAG: Tuple[float, float, float] = (
    0.323287924190616,
    0.4065017233024265,
    7.00299644871222,
)
RY_DURATION: float = 3.139481229843545

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

    The Hamiltonian is ``coeff(t) * op`` acting on ``wires``, integrated from
    $0$ to ``duration``.  ``angle_fn`` maps the circuit's input parameter
    vector to the segment's scale factor; ``drag`` holds the envelope
    parameters of a driven rotation and is ``None`` for the
    constant-coefficient gates.
    """

    op: np.ndarray
    wires: Tuple[int, ...]
    duration: float
    angle_fn: Callable[[np.ndarray], float]
    drag: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class Channel:
    """Depolarizing channel of probability ``p`` on ``wire``.

    Sits between the segments of two gates, where the ``noise`` mode's circuit
    spec places it, which is also where jaqsi's pulse gates apply their noise.
    """

    wire: int
    p: float


def drag_env(t, drag: Tuple[float, float, float], xp=np):
    """Evaluate the drag envelope $\\Omega(t)$ of a driven rotation.

    Matches ``PulseEnvelope.drag`` evaluated at the moving pulse centre
    $t_c = t/2$.  Pass ``xp=jnp`` for the JAX-based backends.
    """
    amplitude, beta, sigma = drag
    gaussian = amplitude * xp.exp(-(t**2) / (8.0 * sigma**2))
    return gaussian * (1.0 - beta * t / (2.0 * sigma**2))


def make_coeff_fn(segment: Segment, params, xp=np) -> Callable:
    """Return the time-dependent coefficient $c(t)$ of *segment* at *params*.

    Constant-coefficient segments return their scale factor unchanged, so the
    callable is valid for every segment type.
    """
    angle = segment.angle_fn(params)
    if segment.drag is None:
        return lambda t: angle

    drag = segment.drag

    def coeff(t):
        return 0.5 * drag_env(t, drag, xp) * angle

    return coeff


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


def _ry(angle_fn: Callable[[np.ndarray], float], wire: int) -> Segment:
    """Driven $RY$ rotation with the calibrated drag envelope."""
    return Segment(
        op=PAULI_Y,
        wires=(wire,),
        duration=RY_DURATION,
        angle_fn=angle_fn,
        drag=RY_DRAG,
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


def _hadamard(wire: int) -> List[Segment]:
    """Hadamard as $RZ(\\pi)$, $RY(\\frac{\\pi}{2})$ and the correction phase."""
    return [
        _rz(lambda params: np.pi, wire),
        _ry(lambda params: np.pi / 2, wire),
        _correction(wire),
    ]


def _cnot(control: int, target: int) -> List[Segment]:
    """$CX$ as $H$, $CZ$, $H$ on the target."""
    return [*_hadamard(target), _cz(control, target), *_hadamard(target)]


def _crx(control: int, target: int, index: int) -> List[Segment]:
    """$CRX(\\phi)$ as two $CX$ interleaved with $RY(\\pm\\frac{\\phi}{2})$.

    *index* is the position of this gate's angle within the circuit's input
    parameter vector.
    """
    return [
        _rz(lambda params: np.pi / 2, target),
        _ry(lambda params: params[index] / 2, target),
        *_cnot(control, target),
        _ry(lambda params: -params[index] / 2, target),
        *_cnot(control, target),
        _rz(lambda params: -np.pi / 2, target),
    ]


def build_schedule(spec: CircuitSpec) -> List[Union[Segment, Channel]]:
    """Return the pulse schedule of *spec* in execution order.

    Only the Hadamard and $CRX$ decompositions are transcribed, so the
    ``crx_ring`` family is the only one with a pulse-level counterpart: its
    Hadamard layer contributes $3 n$ segments and every $CRX$ ring another
    $18 n$, i.e. $21 n$ at a depth of one.  A noisy spec adds a
    :class:`Channel` wherever it holds a ``DEPOL`` op, after the segments of
    the gate before it.
    """
    segments: List[Union[Segment, Channel]] = []
    for op in spec.ops:
        if op.gate == "H":
            segments.extend(_hadamard(op.wires[0]))
        elif op.gate == "CRX":
            segments.extend(_crx(op.wires[0], op.wires[1], op.index))
        elif op.gate == "DEPOL":
            segments.append(Channel(op.wires[0], spec.depolarizing))
        else:
            raise ValueError(f"Gate {op.gate!r} has no pulse transcription")
    return segments
