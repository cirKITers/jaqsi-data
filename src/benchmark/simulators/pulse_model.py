r"""Transcribe JAQSI pulse gates into a shared benchmark schedule.

Each gate becomes segments solving $\mathrm{d}U/\mathrm{d}t=-iH(t)U$.
Driven $RY$ uses JAQSI's calibrated Gaussian or DRAG envelope under the
rotating-wave approximation, with $H(t)=w(E(t)Y-Q(t)X)/2$ and
$Q(t)=-\beta\dot E(t)$ for DRAG ($Q=0$ for Gaussian). $RZ$, $CZ$, and the
Hadamard correction use constant Hamiltonians. Other adapters integrate
these same segments. Module import requires only NumPy.
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
    """Describe one segment of a local pulse Hamiltonian.

    The in-phase operator and optional DRAG quadrature act on ``wires`` over
    ``duration``. ``angle_fn`` reads the circuit parameter; ``envelope`` is
    set only for driven rotations.
    """

    op: np.ndarray
    wires: Tuple[int, ...]
    duration: float
    angle_fn: Callable[[np.ndarray], float]
    envelope: Optional[Tuple[float, ...]] = None
    quad_op: Optional[np.ndarray] = None

    @property
    def ops(self) -> Tuple[np.ndarray, ...]:
        """Return Hamiltonian operators, with the DRAG quadrature last."""
        return (self.op,) if self.quad_op is None else (self.op, self.quad_op)


@dataclass(frozen=True)
class Channel:
    """Represent a depolarizing channel on one wire.

    The noise mode places it between gates, as jaqsi pulse gates do.
    """

    wire: int
    p: float


def ry_envelope(envelope: str) -> Tuple[Tuple[float, ...], float]:
    """Return JAQSI's calibrated driven-RY envelope and duration.

    Gaussian uses ``(A, sigma)``; DRAG uses ``(A, beta, sigma)``.
    """
    from jaqsi.pulses import PulseEnvelope

    *params, duration = (
        float(x) for x in PulseEnvelope.get(envelope)["defaults"]["RY"]
    )
    return tuple(params), duration


def in_phase_env(t, envelope: Tuple[float, ...], duration: float, xp=np):
    r"""Evaluate the driven rotation's lifted Gaussian envelope.

    $E(t)=A(g(t)-g(0))/(1-g(0))$ with
    $g(t)=\exp(-(t-T/2)^2/(2\sigma^2))$.
    Match JAQSI's Gaussian and DRAG in-phase terms, including zero amplitude
    at pulse edges. ``expm1`` keeps values accurate for wide envelopes and
    finite outside the pulse, where adaptive solvers may probe. Use
    ``xp=jnp`` for JAX backends.
    """
    amplitude, sigma = envelope[0], envelope[-1]
    # Exponents of $g(t)$ and $g(0)$, and their difference without cancellation.
    a = -((t - duration / 2) ** 2) / (2.0 * sigma**2)
    b = -((duration / 2) ** 2) / (2.0 * sigma**2)
    d = t * (duration - t) / (2.0 * sigma**2)
    rise = xp.expm1(xp.minimum(d, 0.0)) - xp.expm1(xp.minimum(-d, 0.0))
    return amplitude * xp.exp(xp.maximum(a, b)) * rise / -xp.expm1(b)


def quadrature_env(t, envelope: Tuple[float, float, float], duration: float, xp=np):
    r"""Evaluate the DRAG term $Q(t)=-\beta\dot E(t)$.

    Unlike the in-phase envelope, it need not vanish at pulse edges.
    """
    amplitude, beta, sigma = envelope
    center = duration / 2
    gauss = xp.exp(-((t - center) ** 2) / (2.0 * sigma**2))
    lift = -xp.expm1(-(center**2) / (2.0 * sigma**2))
    return amplitude * beta * (t - center) / sigma**2 * gauss / lift


def make_coeff_fns(segment: Segment, params, xp=np) -> List[Callable]:
    """Build coefficient functions for a segment and parameter vector.

    Return them in ``segment.ops`` order; constant terms retain their scale.
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
    """Embed an operator in a big-endian qubit register.

    Qubit zero is most significant, matching JAQSI and PennyLane.
    """
    rest = [q for q in range(n_qubits) if q not in wires]
    full = xp.kron(op, xp.eye(2 ** len(rest), dtype=op.dtype)) if rest else op

    order = list(wires) + rest
    perm = [order.index(q) for q in range(n_qubits)]
    tensor = full.reshape([2] * (2 * n_qubits))
    tensor = tensor.transpose(perm + [p + n_qubits for p in perm])
    return tensor.reshape(2**n_qubits, 2**n_qubits)


def apply_local(u, psi, wires: Tuple[int, ...], n_qubits: int, xp=np):
    """Apply a local unitary to a statevector by tensor contraction.

    Equivalent to embedding the unitary in the full big-endian register.
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
    r"""Apply local $U\rho U^\dagger$ to a density matrix.

    Contract ket and bra axes under the big-endian convention.
    """
    k = len(wires)
    u = u.reshape([2] * (2 * k))
    tensor = rho.reshape([2] * (2 * n_qubits))
    tensor = _contract_axes(u, tensor, list(wires), xp)
    tensor = _contract_axes(xp.conj(u), tensor, [w + n_qubits for w in wires], xp)
    return tensor.reshape(2**n_qubits, 2**n_qubits)


def depolarize(rho, p: float, wire: int, n_qubits: int, xp=np):
    r"""Apply a single-wire depolarizing channel to ``rho``.

    Use $(1-p)\rho + p(X\rho X+Y\rho Y+Z\rho Z)/3$, as in JAQSI,
    PennyLane, and Qulacs.
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
    """Project a final statevector into the requested mode.

    For density output, form the pure-state projector.
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
    r"""Build driven $RY$ segments with the calibrated envelope.

    The DRAG quadrature drives $-X$ at carrier phase $\pi/2$.
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
    r"""Build $CRX(\phi)$ from two $CX$ gates and two $RY$ rotations.

    ``index`` locates its angle in the input parameter vector.
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
    """Build ``spec``'s pulse segments and noise channels in order.

    ``envelope`` selects JAQSI Gaussian or DRAG driven rotations. Only the
    ``crx_ring`` family's Hadamard and $CRX$ gates are transcribed: its
    Hadamard layer uses $3n$ segments and each $CRX$ ring uses $18n$.
    Insert each ``DEPOL`` channel after its preceding gate's segments.
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
