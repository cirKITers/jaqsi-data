"""Qiskit simulator benchmark adapter.

Uses Qiskit's local statevector / density-matrix simulation
(no external provider or API key required).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import jax.numpy as jnp

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import (
    DensityMatrix,
    SparsePauliOp,
    Statevector,
)
from qiskit_aer import AerSimulator

from benchmark.simulators.base import SimulatorBenchmark, Mode, _endian_reverse_indices


class QiskitBenchmark(SimulatorBenchmark):
    name = "qiskit"

    def __init__(self) -> None:
        self._circuit: QuantumCircuit | None = None
        self._param: Parameter | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0
        self._run_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, n_qubits: int, mode: Mode, *, optimal_config: bool = False) -> None:
        self._n_qubits = n_qubits
        self._mode = mode

        phi = Parameter("phi")
        self._param = phi

        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        for i in range(n_qubits):
            qc.crx(phi, i, (i + 1) % n_qubits)
        self._circuit = qc

        # Pre-build the run function for the chosen mode to avoid
        # repeated branching inside the hot loop.
        self._run_fn = self._make_run_fn(mode, n_qubits, optimal_config)

    def _make_run_fn(
        self, mode: Mode, n_qubits: int, optimal_config: bool
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a callable that maps a batch of phi values to results."""

        # Pre-compute the endian-reversal index permutation once.
        perm = _endian_reverse_indices(n_qubits)

        if optimal_config:
            return self._make_aer_run_fn(mode, n_qubits, perm)

        if mode == "state":

            def _run_state(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = self._circuit.assign_parameters({self._param: float(phi_val)})
                    sv = Statevector.from_instruction(bound)
                    # Reverse qubit ordering: little-endian → big-endian
                    results.append(sv.data[perm])
                return jnp.array(np.stack(results))

            return _run_state

        elif mode == "probs":

            def _run_probs(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = self._circuit.assign_parameters({self._param: float(phi_val)})
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

            def _run_expval(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = self._circuit.assign_parameters({self._param: float(phi_val)})
                    sv = Statevector.from_instruction(bound)
                    evs = [float(sv.expectation_value(obs).real) for obs in obs_list]
                    results.append(evs)
                return jnp.array(np.array(results))

            return _run_expval

        elif mode == "density":

            def _run_density(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = self._circuit.assign_parameters({self._param: float(phi_val)})
                    dm = DensityMatrix.from_instruction(bound)
                    # Reverse qubit ordering on both axes
                    results.append(dm.data[np.ix_(perm, perm)])
                return jnp.array(np.stack(results))

            return _run_density

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

    def _make_aer_run_fn(
        self, mode: Mode, n_qubits: int, perm: np.ndarray
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a run function backed by the qiskit-aer C++ simulator.

        Aer uses the same little-endian basis order as quantum_info, so the
        endian-reversal permutation *perm* is applied identically.  The circuit
        is transpiled once outside the timing loop; parameter binding stays a
        per-phi loop to mirror the default path.
        """
        method = "density_matrix" if mode == "density" else "statevector"
        sim = AerSimulator(method=method, precision="double")

        if mode == "state":
            qc = self._circuit.copy()
            qc.save_statevector(label="sv")
            qc = transpile(qc, sim)

            def _run_state(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = qc.assign_parameters({self._param: float(phi_val)})
                    sv = np.asarray(sim.run(bound).result().data(0)["sv"])
                    results.append(sv[perm])
                return jnp.array(np.stack(results))

            return _run_state

        elif mode == "probs":
            qc = self._circuit.copy()
            qc.save_probabilities(label="probs")
            qc = transpile(qc, sim)

            def _run_probs(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = qc.assign_parameters({self._param: float(phi_val)})
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

            def _run_expval(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = qc.assign_parameters({self._param: float(phi_val)})
                    data = sim.run(bound).result().data(0)
                    evs = [float(np.real(data[f"z{i}"])) for i in range(n_qubits)]
                    results.append(evs)
                return jnp.array(np.array(results))

            return _run_expval

        elif mode == "density":
            qc = self._circuit.copy()
            qc.save_density_matrix(label="dm")
            qc = transpile(qc, sim)

            def _run_density(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = qc.assign_parameters({self._param: float(phi_val)})
                    dm = np.asarray(sim.run(bound).result().data(0)["dm"])
                    results.append(dm[np.ix_(perm, perm)])
                return jnp.array(np.stack(results))

            return _run_density

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(phi)

    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(phi)