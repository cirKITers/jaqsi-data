"""Benchmark local Qulacs statevector and density matrix simulators."""

from __future__ import annotations

from typing import Callable, Tuple

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
    r"""Return the standard $RX(\phi)=\exp(-i\phi X/2)$ matrix."""
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _make_crx_gate(control: int, target: int, angle: float):
    """Build controlled RX from a dense matrix.

    Qulacs special RX gates do not support ``add_control_qubit``.
    """
    mat = _rx_matrix(angle)
    crx = DenseMatrix(target, mat)
    crx.add_control_qubit(control, 1)
    return crx


def _angle(op: Op, sample: np.ndarray, weights: np.ndarray) -> float:
    """Return an operation angle as a Python float.

    This is the NumPy counterpart of :func:`benchmark.circuits.angle`.
    """
    vector = sample if op.source == "inputs" else weights
    return float(vector[op.index])


def _build_circuit(
    spec: CircuitSpec, sample: np.ndarray, weights: np.ndarray
) -> QuantumCircuit:
    """Build ``spec`` for one input sample."""
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
    spec: CircuitSpec,
    sample: np.ndarray,
    weights: np.ndarray,
    sources: Tuple[str, ...] = (),
) -> ParametricQuantumCircuit:
    """Build a Qulacs parametric circuit from ``spec``.

    By default only trainable rotations are parametric, so ``backprop``
    returns gradients in trainable-vector order. Passing both vectors in
    ``sources`` makes all rotations parametric and supports updating the
    circuit for each sample.
    """
    sources = sources or (spec.trainable,)
    circuit = ParametricQuantumCircuit(spec.n_qubits)
    for op in spec.ops:
        if op.gate == "H":
            circuit.add_H_gate(op.wires[0])
        elif op.gate == "CNOT":
            circuit.add_CNOT_gate(op.wires[0], op.wires[1])
        elif op.gate == "DEPOL":
            circuit.add_gate(DepolarizingNoise(op.wires[0], spec.depolarizing))
        elif op.gate in ("RX", "RZ"):
            theta = _SIGN * _angle(op, sample, weights)
            if op.source in sources:
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
        """Build a function that executes a batch of inputs."""

        n_qubits = spec.n_qubits

        # Pre-compute the endian-reversal index permutation once.
        perm = _endian_reverse_indices(n_qubits)

        # Gate fusion needs concrete angles: the optimizer leaves parametric
        # gates alone so that they stay updatable.  A fused circuit is
        # therefore rebuilt for every sample inside the timed loop, which beats
        # updating the angles of a parametric circuit built once from ten
        # qubits on in expval and from four in density.
        if (
            optimal_config
            and mode == "noise"
            and all(op.gate != "CRX" for op in spec.ops)
        ):
            rotations = tuple(op for op in spec.ops if op.source is not None)
            circuit = _build_parametric_circuit(
                spec,
                np.zeros(spec.n_inputs),
                np.zeros(spec.n_weights),
                sources=("inputs", "weights"),
            )

            def _build(sample: np.ndarray, weights: np.ndarray) -> QuantumCircuit:
                for k, op in enumerate(rotations):
                    circuit.set_parameter(k, _SIGN * _angle(op, sample, weights))
                return circuit

        else:

            def _build(sample: np.ndarray, weights: np.ndarray) -> QuantumCircuit:
                circuit = _build_circuit(spec, sample, weights)
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
        """Differentiate the summed Pauli-Z observable with Qulacs backprop.

        Summing all qubits removes the need for endianness conversion. Rebuild
        the circuit per sample so backprop differentiates only trainable gates;
        making data-encoding gates parametric was 7–10% slower from ten qubits on.
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
