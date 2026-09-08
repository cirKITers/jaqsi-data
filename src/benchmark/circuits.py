"""Backend-agnostic definition of the benchmark circuits.

Every adapter builds its circuit by walking the operation list of a
:class:`CircuitSpec` rather than restating the circuit in its own API, so all
simulators provably execute the same gate sequence with the same parameters.

Parameters are split into two flat vectors:

``inputs``
    Batched.  One vector per sample, so a batch of shape ``(B, n_inputs)``
    represents $B$ independent evaluations.
``weights``
    Shared across the batch.  A single vector of shape ``(n_weights,)``.

This is the split a QML workload has: a batch of data encoded into the circuit
against one set of trainable parameters.  Families without a data-encoding
layer leave ``weights`` empty and carry their parameters in ``inputs``, which
keeps the batch axis in the same place for every family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Gates the adapters have to implement.  Kept deliberately small: every family
# below decomposes into these, and a simulator missing one of them would not be
# a fair comparison anyway.
GATES = ("H", "RX", "RZ", "CRX", "CNOT")


@dataclass(frozen=True)
class Op:
    """One gate of a circuit.

    ``source`` names the parameter vector the rotation angle is read from
    (``"inputs"`` or ``"weights"``) and ``index`` is the position within it.
    Both are ``None``/``-1`` for the non-parametric gates.
    """

    gate: str
    wires: Tuple[int, ...]
    source: Optional[str] = None
    index: int = -1


@dataclass(frozen=True)
class CircuitSpec:
    """A benchmark circuit at a fixed qubit count and depth."""

    family: str
    n_qubits: int
    n_layers: int
    n_inputs: int
    n_weights: int
    ops: Tuple[Op, ...]

    @property
    def trainable(self) -> str:
        """Name of the parameter vector the gradient is taken with respect to.

        Families with a data-encoding layer train ``weights``; the ones that
        carry all their parameters in ``inputs`` train those instead.
        """
        return "weights" if self.n_weights else "inputs"

    @property
    def n_trainable(self) -> int:
        return self.n_weights if self.n_weights else self.n_inputs


def _crx_ring(n_qubits: int, n_layers: int) -> Tuple[List[Op], int, int]:
    """Hadamard layer followed by *n_layers* rings of $CRX$.

    The gate-level and pulse-level benchmarks both run this family; it is the
    only one ``pulse_model`` transcribes.  Every $CRX$ carries its own angle,
    so a ring of $n$ gates has $n$ independent parameters.
    """
    ops = [Op("H", (i,)) for i in range(n_qubits)]
    k = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            ops.append(Op("CRX", (i, (i + 1) % n_qubits), "inputs", k))
            k += 1
    return ops, k, 0


def _hea(n_qubits: int, n_layers: int) -> Tuple[List[Op], int, int]:
    """Hardware-efficient ansatz with an $RX$ data-encoding layer.

    One $RX$ per wire encodes the input, then *n_layers* blocks of rotations on
    every wire followed by a $CNOT$ ring.  This is the
    rotation-layer/entangler-ring circuit the Yao and Qulacs benchmarks use,
    with the encoding layer that makes the batch axis a data axis.

    Following that convention, the blocks are $RZ \\cdot RX \\cdot RZ$ except
    the last, which is $RZ \\cdot RX$.  A trailing $RZ$ could not change a
    Pauli-Z expectation value anyway: it only adds phases, and the $CNOT$ ring
    after it permutes the computational basis, so neither touches the $Z$-basis
    probabilities.  Keeping it would leave $n$ parameters with an exactly zero
    gradient.

    The leading $RZ$ of the first block *is* kept, unlike in Yao and Qulacs.
    They start from $\\lvert 0 \\rangle$, where a leading $RZ$ is a global
    phase; here the encoding layer has already rotated the state off the $Z$
    axis, so it acts non-trivially.
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


def build_spec(family: str, n_qubits: int, n_layers: int) -> CircuitSpec:
    """Return the :class:`CircuitSpec` of *family* at the given size."""
    if family not in FAMILIES:
        raise ValueError(
            f"Unknown circuit family: {family!r}. Available: {sorted(FAMILIES)}"
        )
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}")

    ops, n_inputs, n_weights = FAMILIES[family](n_qubits, n_layers)
    return CircuitSpec(
        family=family,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_inputs=n_inputs,
        n_weights=n_weights,
        ops=tuple(ops),
    )


def angle(op: Op, inputs, weights):
    """Return the rotation angle of *op*, read from the matching vector.

    Indexing with an ellipsis keeps the call valid for a single parameter
    vector and for a batch of them, which the adapters relying on their own
    parameter broadcasting need.
    """
    vector = inputs if op.source == "inputs" else weights
    return vector[..., op.index]
