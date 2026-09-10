"""Qulacs simulator benchmark adapter.

Uses Qulacs' local statevector / density-matrix simulation for fast
quantum circuit simulation (no external provider or API key required).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import jax.numpy as jnp

from qulacs import (
    DensityMatrix,
    Observable,
    ParametricQuantumCircuit,
    QuantumCircuit,
    QuantumState,
)
from qulacs.circuit import QuantumCircuitOptimizer
from qulacs.gate import DenseMatrix, DepolarizingNoise

from benchmark.circuits import CircuitSpec, Op
from benchmark.simulators.base import SimulatorBenchmark, Mode, _endian_reverse_indices

# Qulacs defines its rotations as $R_P(\theta) = e^{+i\theta P/2}$, the opposite
# sign of the convention JAQSI, PennyLane, Qiskit and Qibo use.  Every angle
# handed to a Qulacs rotation is negated, and gradients taken with respect to
# those angles are negated back.
_SIGN = -1.0


def _rx_matrix(angle: float) -> np.ndarray:
    """Return the 2×2 RX matrix using the standard convention.

    Standard (textbook / PennyLane) convention:
        RX(φ) = exp(-i φ/2 X) = [[cos(φ/2), -i·sin(φ/2)],
                                  [-i·sin(φ/2), cos(φ/2)]]
    """
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _make_crx_gate(control: int, target: int, angle: float):
    """Build a controlled-RX gate.

    Qulacs' special gates (like ``RX``) do not support
    ``add_control_qubit``, so we create a ``DenseMatrix`` gate from
    the explicit RX matrix and then attach the control qubit.
    """
    mat = _rx_matrix(angle)
    crx = DenseMatrix(target, mat)
    crx.add_control_qubit(control, 1)
    return crx


def _angle(op: Op, sample: np.ndarray, weights: np.ndarray) -> float:
    """Return the rotation angle of *op* as a Python float.

    Qulacs takes plain floats, so this is the numpy counterpart of
    :func:`benchmark.circuits.angle`, which keeps array semantics.
    """
    vector = sample if op.source == "inputs" else weights
    return float(vector[op.index])


def _build_circuit(
    spec: CircuitSpec, sample: np.ndarray, weights: np.ndarray
) -> QuantumCircuit:
    """Build the benchmark circuit of *spec* for one sample."""
    circuit = QuantumCircuit(spec.n_qubits)
    for op in spec.ops:
        if op.gate == "H":
            circuit.add_H_gate(op.wires[0])
        elif op.gate == "RX":
            circuit.add_RX_gate(op.wires[0], _SIGN * _angle(op, sample, weights))
        elif op.gate == "RZ":
            circuit.add_RZ_gate(op.wires[0], _SIGN * _angle(op, sample, weights))
        elif op.gate == "CRX":
            circuit.add_gate(
                _make_crx_gate(op.wires[0], op.wires[1], _angle(op, sample, weights))
            )
        elif op.gate == "CNOT":
            circuit.add_CNOT_gate(op.wires[0], op.wires[1])
        elif op.gate == "DEPOL":
            circuit.add_gate(DepolarizingNoise(op.wires[0], spec.depolarizing))
        else:
            raise ValueError(f"Unsupported gate: {op.gate!r}")
    return circuit


def _build_parametric_circuit(
    spec: CircuitSpec, sample: np.ndarray, weights: np.ndarray
) -> ParametricQuantumCircuit:
    """Build *spec* with the trainable rotations as parametric gates.

    ``ParametricQuantumCircuit.backprop`` returns one gradient per parametric
    gate, so only the trainable rotations are registered as parametric and the
    data-encoding ones stay fixed.  The gradients then come back in exactly the
    order of the trainable vector.
    """
    trainable = spec.trainable
    circuit = ParametricQuantumCircuit(spec.n_qubits)
    for op in spec.ops:
        if op.gate == "H":
            circuit.add_H_gate(op.wires[0])
        elif op.gate == "CNOT":
            circuit.add_CNOT_gate(op.wires[0], op.wires[1])
        elif op.gate in ("RX", "RZ"):
            theta = _SIGN * _angle(op, sample, weights)
            if op.source == trainable:
                adder = (
                    circuit.add_parametric_RX_gate
                    if op.gate == "RX"
                    else circuit.add_parametric_RZ_gate
                )
                adder(op.wires[0], theta)
            elif op.gate == "RX":
                circuit.add_RX_gate(op.wires[0], theta)
            else:
                circuit.add_RZ_gate(op.wires[0], theta)
        else:
            raise ValueError(f"Gate {op.gate!r} has no parametric Qulacs equivalent")
    return circuit


class QulacsBenchmark(SimulatorBenchmark):
    name = "qulacs"

    def __init__(self) -> None:
        self._spec: CircuitSpec | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0
        self._run_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None

    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        if mode != "grad":
            return True
        # backprop differentiates parametric rotation gates.  The controlled
        # rotations of ``crx_ring`` are built as dense matrices with an attached
        # control qubit, which Qulacs cannot register as parametric.
        return all(op.gate in ("H", "RX", "RZ", "CNOT") for op in spec.ops)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        self._spec = spec
        self._n_qubits = spec.n_qubits
        self._mode = mode
        self._run_fn = self._make_run_fn(spec, mode, optimal_config)

    # ------------------------------------------------------------------
    # Run function factory
    # ------------------------------------------------------------------
    def _make_run_fn(
        self, spec: CircuitSpec, mode: Mode, optimal_config: bool
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Return a callable that maps a batch of inputs to results."""

        n_qubits = spec.n_qubits

        # Pre-compute the endian-reversal index permutation once.
        perm = _endian_reverse_indices(n_qubits)

        def _build(sample: np.ndarray, weights: np.ndarray) -> QuantumCircuit:
            circuit = _build_circuit(spec, sample, weights)
            # The optimizer treats Qulacs' probabilistic noise gates as the
            # identity and fuses them away, so a noisy circuit runs unfused.
            if optimal_config and mode != "noise":
                # In-place gate fusion; numerically identical to the default.
                QuantumCircuitOptimizer().optimize_light(circuit)
            return circuit

        if mode == "grad":
            return self._make_grad_fn(spec)

        if mode == "state":

            def _run_state(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    circuit = _build(sample, w)
                    state = QuantumState(n_qubits)
                    circuit.update_quantum_state(state)
                    # Reverse qubit ordering: little-endian -> big-endian
                    results.append(state.get_vector()[perm])
                return jnp.array(np.stack(results))

            return _run_state

        elif mode == "probs":

            def _run_probs(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    circuit = _build(sample, w)
                    state = QuantumState(n_qubits)
                    circuit.update_quantum_state(state)
                    sv = state.get_vector()
                    # Reverse qubit ordering: little-endian -> big-endian
                    results.append((np.abs(sv) ** 2)[perm])
                return jnp.array(np.stack(results))

            return _run_probs

        elif mode == "expval":

            # Per-qubit Pauli-Z signs over the little-endian computational basis,
            # precomputed once. signs[i, k] is +1 when bit i of basis index k is 0
            # and -1 otherwise, so signs @ probabilities gives the per-qubit Z
            # expectation values in Qulacs qubit order.
            basis = np.arange(1 << n_qubits)
            signs = 1.0 - 2.0 * ((basis[None, :] >> np.arange(n_qubits)[:, None]) & 1)

            def _run_expval(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    circuit = _build(sample, w)
                    state = QuantumState(n_qubits)
                    circuit.update_quantum_state(state)
                    probs = np.abs(state.get_vector()) ** 2
                    results.append(signs @ probs)
                return jnp.array(np.stack(results))

            return _run_expval

        elif mode in ("density", "noise"):

            def _run_density(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    circuit = _build(sample, w)
                    dm = DensityMatrix(n_qubits)
                    circuit.update_quantum_state(dm)
                    # Reverse qubit ordering on both axes
                    results.append(dm.get_matrix()[np.ix_(perm, perm)])
                return jnp.array(np.stack(results))

            return _run_density

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

    def _make_grad_fn(
        self, spec: CircuitSpec
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Return the analytic gradient via ``ParametricQuantumCircuit.backprop``.

        The observable is the summed per-qubit Pauli-Z, matching the loss the
        other adapters differentiate.  Qulacs labels qubit 0 as the least
        significant bit, but a sum over all qubits is invariant under that
        relabelling, so no reordering is needed.
        """
        observable = Observable(spec.n_qubits)
        for i in range(spec.n_qubits):
            observable.add_operator(1.0, f"Z {i}")

        per_sample = spec.trainable == "inputs"

        def _run_grad(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
            w = np.asarray(weights)
            grads = []
            for sample in np.asarray(inputs):
                circuit = _build_parametric_circuit(spec, sample, w)
                grads.append(_SIGN * np.asarray(circuit.backprop(observable)))
            if per_sample:
                return jnp.array(np.stack(grads))
            return jnp.array(np.sum(grads, axis=0))

        return _run_grad

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)

    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)
