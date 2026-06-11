"""Qibo simulator benchmark adapter.

Uses Qibo's built-in numpy backend for local statevector / density-matrix
simulation (no external provider or API key required).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import jax.numpy as jnp

from qibo import Circuit, gates, set_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z

from benchmark.simulators.base import SimulatorBenchmark, Mode


class QiboBenchmark(SimulatorBenchmark):
    name = "qibo"

    def __init__(self) -> None:
        self._circuit: Circuit | None = None
        self._circuit_dm: Circuit | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0
        self._n_params: int = 0
        self._run_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None
        self._z_hams: List[SymbolicHamiltonian] = []

    # ------------------------------------------------------------------
    # Circuit builders
    # ------------------------------------------------------------------
    def _build_circuit(self, n_qubits: int, *, density_matrix: bool = False) -> Circuit:
        """Build the parametric benchmark circuit."""
        c = Circuit(n_qubits, density_matrix=density_matrix)
        for i in range(n_qubits):
            c.add(gates.H(i))
        for i in range(n_qubits):
            c.add(gates.CRX(i, (i + 1) % n_qubits, theta=0.0))
        return c

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, n_qubits: int, mode: Mode, *, optimal_config: bool = False) -> None:
        # Backend selection is global to the process but only affects Qibo.
        # Reset to numpy on the default path so a prior optimal run does not leak.
        if optimal_config:
            set_backend("qibojit", platform="numba")
        else:
            set_backend("numpy")

        self._n_qubits = n_qubits
        self._mode = mode
        self._n_params = n_qubits  # one CRX per qubit

        if mode == "density":
            self._circuit_dm = self._build_circuit(n_qubits, density_matrix=True)
            self._circuit = None
        else:
            self._circuit = self._build_circuit(n_qubits)
            self._circuit_dm = None

        if mode == "expval":
            self._z_hams = [
                SymbolicHamiltonian(Z(i), nqubits=n_qubits)
                for i in range(n_qubits)
            ]

        self._run_fn = self._make_run_fn(mode, n_qubits)

    # ------------------------------------------------------------------
    # Run function factory
    # ------------------------------------------------------------------
    def _make_run_fn(
        self, mode: Mode, n_qubits: int
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a callable that maps a batch of phi values to results."""

        if mode == "state":
            def _run_state(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    params = [float(phi_val)] * self._n_params
                    self._circuit.set_parameters(params)
                    result = self._circuit()
                    results.append(np.array(result.state()))
                return jnp.array(np.stack(results))
            return _run_state

        elif mode == "probs":
            def _run_probs(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    params = [float(phi_val)] * self._n_params
                    self._circuit.set_parameters(params)
                    result = self._circuit()
                    results.append(np.array(result.probabilities()))
                return jnp.array(np.stack(results))
            return _run_probs

        elif mode == "expval":
            def _run_expval(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    params = [float(phi_val)] * self._n_params
                    self._circuit.set_parameters(params)
                    result = self._circuit()
                    state = result.state()
                    evs = [
                        float(np.real(ham.expectation_from_state(state)))
                        for ham in self._z_hams
                    ]
                    results.append(evs)
                return jnp.array(np.array(results))
            return _run_expval

        elif mode == "density":
            def _run_density(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    params = [float(phi_val)] * self._n_params
                    self._circuit_dm.set_parameters(params)
                    result = self._circuit_dm()
                    results.append(np.array(result.state()))
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