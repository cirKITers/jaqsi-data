"""Qiskit simulator benchmark adapter.

Uses Qiskit's local statevector / density-matrix simulation
(no external provider or API key required).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import jax.numpy as jnp

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import (
    DensityMatrix,
    SparsePauliOp,
    Statevector,
)

from benchmark.simulators.base import SimulatorBenchmark, Mode


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
    def setup(self, n_qubits: int, mode: Mode) -> None:
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
        self._run_fn = self._make_run_fn(mode, n_qubits)

    def _make_run_fn(
        self, mode: Mode, n_qubits: int
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a callable that maps a batch of phi values to results."""

        if mode == "state":

            def _run_state(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = self._circuit.assign_parameters({self._param: float(phi_val)})
                    sv = Statevector.from_instruction(bound)
                    results.append(sv.data)
                return jnp.array(np.stack(results))

            return _run_state

        elif mode == "probs":

            def _run_probs(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    bound = self._circuit.assign_parameters({self._param: float(phi_val)})
                    sv = Statevector.from_instruction(bound)
                    results.append(sv.probabilities())
                return jnp.array(np.stack(results))

            return _run_probs

        elif mode == "expval":

            # Build per-qubit Z observables
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
                    results.append(dm.data)
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