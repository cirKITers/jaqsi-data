"""Qiskit simulator benchmark adapter.

Uses Qiskit's local statevector / density-matrix simulation
(no external provider or API key required).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import jax.numpy as jnp

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import (
    DensityMatrix,
    SparsePauliOp,
    Statevector,
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error

from benchmark.circuits import CircuitSpec
from benchmark.simulators.base import SimulatorBenchmark, Mode, _endian_reverse_indices


def _build_circuit(
    spec: CircuitSpec,
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Return the parametric circuit of *spec* and its two parameter vectors."""
    x = ParameterVector("x", spec.n_inputs)
    w = ParameterVector("w", spec.n_weights)
    vectors = {"inputs": x, "weights": w}

    qc = QuantumCircuit(spec.n_qubits)
    for op in spec.ops:
        if op.gate == "H":
            qc.h(op.wires[0])
        elif op.gate == "RX":
            qc.rx(vectors[op.source][op.index], op.wires[0])
        elif op.gate == "RZ":
            qc.rz(vectors[op.source][op.index], op.wires[0])
        elif op.gate == "CRX":
            qc.crx(vectors[op.source][op.index], op.wires[0], op.wires[1])
        elif op.gate == "CNOT":
            qc.cx(op.wires[0], op.wires[1])
        elif op.gate == "DEPOL":
            # Appended as an instruction rather than through a NoiseModel: Aer
            # attaches model errors per basis gate after transpilation, which
            # splits CRX into several gates.  Qiskit's $\lambda$ is the
            # probability $p$ of the other frameworks at $\lambda = 4p/3$.
            qc.append(depolarizing_error(4 * spec.depolarizing / 3, 1), [op.wires[0]])
        else:
            raise ValueError(f"Unsupported gate: {op.gate!r}")
    return qc, x, w


class QiskitBenchmark(SimulatorBenchmark):
    name = "qiskit"

    def __init__(self) -> None:
        self._circuit: QuantumCircuit | None = None
        self._x: ParameterVector | None = None
        self._w: ParameterVector | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0
        self._run_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None

    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        # Qiskit's gradient interface lives in the separate qiskit-algorithms
        # package, which is not a dependency here; the forward measurements are
        # unaffected.
        return mode != "grad"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        self._n_qubits = spec.n_qubits
        self._mode = mode
        self._circuit, self._x, self._w = _build_circuit(spec)

        # Pre-build the run function for the chosen mode to avoid
        # repeated branching inside the hot loop.
        self._run_fn = self._make_run_fn(mode, spec.n_qubits, optimal_config)

    def _bindings(self, sample: np.ndarray, weights: np.ndarray) -> Dict:
        """Map one sample and the shared weights onto the circuit parameters."""
        binding = {self._x[i]: float(v) for i, v in enumerate(sample)}
        binding.update({self._w[i]: float(v) for i, v in enumerate(weights)})
        return binding

    def _make_run_fn(
        self, mode: Mode, n_qubits: int, optimal_config: bool
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Return a callable that maps a batch of inputs to results."""

        # Pre-compute the endian-reversal index permutation once.
        perm = _endian_reverse_indices(n_qubits)

        if optimal_config:
            return self._make_aer_run_fn(mode, n_qubits, perm)

        if mode == "state":

            def _run_state(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    bound = self._circuit.assign_parameters(self._bindings(sample, w))
                    sv = Statevector.from_instruction(bound)
                    # Reverse qubit ordering: little-endian → big-endian
                    results.append(sv.data[perm])
                return jnp.array(np.stack(results))

            return _run_state

        elif mode == "probs":

            def _run_probs(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    bound = self._circuit.assign_parameters(self._bindings(sample, w))
                    sv = Statevector.from_instruction(bound)
                    # Reverse qubit ordering: little-endian → big-endian
                    results.append(sv.probabilities()[perm])
                return jnp.array(np.stack(results))

            return _run_probs

        elif mode == "expval":

            # Build per-qubit Z observables.
            # Qiskit's Pauli label string is right-to-left: the rightmost
            # character corresponds to qubit 0.  To measure Z on
            # big-endian qubit *i* we place 'Z' at position i (from the
            # right).
            obs_list = []
            for i in range(n_qubits):
                label = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
                obs_list.append(SparsePauliOp(label))

            def _run_expval(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    bound = self._circuit.assign_parameters(self._bindings(sample, w))
                    sv = Statevector.from_instruction(bound)
                    evs = [float(sv.expectation_value(obs).real) for obs in obs_list]
                    results.append(evs)
                return jnp.array(np.array(results))

            return _run_expval

        elif mode in ("density", "noise"):

            def _run_density(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    bound = self._circuit.assign_parameters(self._bindings(sample, w))
                    dm = DensityMatrix.from_instruction(bound)
                    # Reverse qubit ordering on both axes
                    results.append(dm.data[np.ix_(perm, perm)])
                return jnp.array(np.stack(results))

            return _run_density

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

    def _make_aer_run_fn(
        self, mode: Mode, n_qubits: int, perm: np.ndarray
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Return a run function backed by the qiskit-aer C++ simulator.

        Aer uses the same little-endian basis order as quantum_info, so the
        endian-reversal permutation *perm* is applied identically.  The circuit
        is transpiled once outside the timing loop; parameter binding stays a
        per-sample loop to mirror the default path.
        """
        method = "density_matrix" if mode in ("density", "noise") else "statevector"
        sim = AerSimulator(method=method, precision="double")

        if mode == "state":
            qc = self._circuit.copy()
            qc.save_statevector(label="sv")
            qc = transpile(qc, sim)

            def _run_state(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    bound = qc.assign_parameters(self._bindings(sample, w))
                    sv = np.asarray(sim.run(bound).result().data(0)["sv"])
                    results.append(sv[perm])
                return jnp.array(np.stack(results))

            return _run_state

        elif mode == "probs":
            qc = self._circuit.copy()
            qc.save_probabilities(label="probs")
            qc = transpile(qc, sim)

            def _run_probs(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    bound = qc.assign_parameters(self._bindings(sample, w))
                    probs = np.asarray(sim.run(bound).result().data(0)["probs"])
                    results.append(probs[perm])
                return jnp.array(np.stack(results))

            return _run_probs

        elif mode == "expval":
            # Per-qubit Z observables; the label's rightmost character is
            # qubit 0, so the expval vector needs no reordering.
            qc = self._circuit.copy()
            for i in range(n_qubits):
                label = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
                qc.save_expectation_value(
                    SparsePauliOp(label), range(n_qubits), label=f"z{i}"
                )
            qc = transpile(qc, sim)

            def _run_expval(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    bound = qc.assign_parameters(self._bindings(sample, w))
                    data = sim.run(bound).result().data(0)
                    evs = [float(np.real(data[f"z{i}"])) for i in range(n_qubits)]
                    results.append(evs)
                return jnp.array(np.array(results))

            return _run_expval

        elif mode in ("density", "noise"):
            qc = self._circuit.copy()
            qc.save_density_matrix(label="dm")
            qc = transpile(qc, sim)

            def _run_density(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    bound = qc.assign_parameters(self._bindings(sample, w))
                    dm = np.asarray(sim.run(bound).result().data(0)["dm"])
                    results.append(dm[np.ix_(perm, perm)])
                return jnp.array(np.stack(results))

            return _run_density

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)

    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)