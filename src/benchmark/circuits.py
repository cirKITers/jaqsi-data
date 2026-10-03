"""Define backend-independent benchmark circuits.

Each adapter reads the same :class:`CircuitSpec` operation list. Batched
``inputs`` have shape ``(batch, n_inputs)``; ``weights`` have shape
``(n_weights,)`` and are shared across the batch. Families without a data
encoding layer keep their parameters in ``inputs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Gates the adapters have to implement.  Kept deliberately small: every family
# below decomposes into these, and a simulator missing one of them would not be
# a fair comparison anyway.  ``DEPOL`` is the single-qubit depolarizing channel
# :func:`build_spec` inserts for the ``noise`` mode.
GATES = ("H", "RX", "RZ", "CRX", "CNOT", "DEPOL")


@dataclass(frozen=True)
class Op:
    """Represent one circuit operation.

    For rotations, ``source`` selects ``inputs`` or ``weights`` and ``index``
    locates the angle. Other gates use ``None`` and ``-1``.
    """

    gate: str
    wires: Tuple[int, ...]
    source: Optional[str] = None
    index: int = -1


@dataclass(frozen=True)
class CircuitSpec:
    """Describe a circuit at a fixed width and depth.

    ``depolarizing`` is the probability of each ``DEPOL`` operation, or zero
    for a circuit without noise.
    """

    family: str
    n_qubits: int
    n_layers: int
    n_inputs: int
    n_weights: int
    ops: Tuple[Op, ...]
    depolarizing: float = 0.0

    @property
    def trainable(self) -> str:
        """Return the parameter vector differentiated by ``grad``.

        Data-encoding families train ``weights``; other families train ``inputs``.
        """
        return "weights" if self.n_weights else "inputs"

    @property
    def n_trainable(self) -> int:
        return self.n_weights if self.n_weights else self.n_inputs


def _crx_ring(n_qubits: int, n_layers: int) -> Tuple[List[Op], int, int]:
    """Build a Hadamard layer followed by ``n_layers`` $CRX$ rings.

    Each $CRX$ has its own angle. This is the circuit transcribed by the pulse
    model.
    """
    ops = [Op("H", (i,)) for i in range(n_qubits)]
    k = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            ops.append(Op("CRX", (i, (i + 1) % n_qubits), "inputs", k))
            k += 1
    return ops, k, 0


def _hea(n_qubits: int, n_layers: int) -> Tuple[List[Op], int, int]:
    """Build the hardware-efficient ansatz with $RX$ data encoding.

    Each layer follows the Yao/Qulacs rotation and entangler pattern: $RZ$,
    $RX$, $RZ$ rotations followed by a $CNOT$ ring. The final layer omits its
    trailing $RZ$ because it cannot affect a
    Pauli-Z expectation and would have zero gradient. The first $RZ$ remains:
    the encoding layer has already rotated the initial state off the Z axis.
    """
    ops = [Op("RX", (i,), "inputs", i) for i in range(n_qubits)]
    k = 0
    for layer in range(n_layers):
        block = ("RZ", "RX") if layer == n_layers - 1 else ("RZ", "RX", "RZ")
        for i in range(n_qubits):
            for gate in block:
                ops.append(Op(gate, (i,), "weights", k))
                k += 1
        for i in range(n_qubits):
            ops.append(Op("CNOT", (i, (i + 1) % n_qubits)))
    return ops, n_qubits, k


FAMILIES: Dict[str, Callable[[int, int], Tuple[List[Op], int, int]]] = {
    "crx_ring": _crx_ring,
    "hea": _hea,
}

# Families the pulse-level adapters can run.  ``pulse_model`` only transcribes
# the Hadamard and $CRX$ decompositions.
PULSE_FAMILIES = frozenset({"crx_ring"})


def build_spec(
    family: str, n_qubits: int, n_layers: int, depolarizing: float = 0.0
) -> CircuitSpec:
    """Build a :class:`CircuitSpec` for the given family and size.

    Nonzero ``depolarizing`` adds a channel on every affected wire after each
    gate. Keeping channels in the operation list gives adapters the same
    noise sequence.
    """
    if family not in FAMILIES:
        raise ValueError(
            f"Unknown circuit family: {family!r}. Available: {sorted(FAMILIES)}"
        )
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}")

    ops, n_inputs, n_weights = FAMILIES[family](n_qubits, n_layers)
    if depolarizing:
        noisy = []
        for op in ops:
            noisy.append(op)
            noisy.extend(Op("DEPOL", (w,)) for w in op.wires)
        ops = noisy
    return CircuitSpec(
        family=family,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_inputs=n_inputs,
        n_weights=n_weights,
        ops=tuple(ops),
        depolarizing=depolarizing,
    )


def angle(op: Op, inputs, weights):
    """Read ``op``'s angle from its input or weight vector.

    The final-axis lookup also accepts batched parameters.
    """
    vector = inputs if op.source == "inputs" else weights
    return vector[..., op.index]
