"""Qulacs simulator benchmark adapter.

Uses Qulacs' local statevector / density-matrix simulation for fast
quantum circuit simulation (no external provider or API key required).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import jax.numpy as jnp

from qulacs import DensityMatrix, Observable, QuantumCircuit, QuantumState
from qulacs.gate import DenseMatrix

from benchmark.simulators.base import SimulatorBenchmark, Mode


def _endian_reverse_indices(n_qubits: int) -> np.ndarray:
    """Return an index array that maps Qulacs' little-endian basis order
    to big-endian order (used by JAQSI, PennyLane, Qibo).

    Qulacs labels qubit 0 as the *least*-significant bit, so basis state
    index ``b_{n-1}\u2026b_1 b_0`` in Qulacs corresponds to
    ``b_0 b_1 \u2026 b_{n-1}`` in big-endian convention.  This function
    returns a permutation that re-sorts a length-2**n vector from
    little-endian to big-endian.
    """
    N = 1 << n_qubits
    indices = np.zeros(N, dtype=int)
    for i in range(N):
        rev = int(f"{i:0{n_qubits}b}"[::-1], 2)
        indices[rev] = i
    return indices


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


def _build_circuit(n_qubits: int, phi: float) -> QuantumCircuit:
    """Build the parametric benchmark circuit for a single phi value.

    1. Hadamard on every qubit.
    2. CRX(phi) in a ring: qubit i → qubit (i+1) mod n.
    """
    circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        circuit.add_H_gate(i)
    for i in range(n_qubits):
        crx = _make_crx_gate(i, (i + 1) % n_qubits, phi)
        circuit.add_gate(crx)
    return circuit


class QulacsBenchmark(SimulatorBenchmark):
    name = "qulacs"

    def __init__(self) -> None:
        self._mode: Mode = "probs"
        self._n_qubits: int = 0
        self._run_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, n_qubits: int, mode: Mode) -> None:
        self._n_qubits = n_qubits
        self._mode = mode
        self._run_fn = self._make_run_fn(mode, n_qubits)

    # ------------------------------------------------------------------
    # Run function factory
    # ------------------------------------------------------------------
    def _make_run_fn(
        self, mode: Mode, n_qubits: int
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a callable that maps a batch of phi values to results."""

        # Pre-compute the endian-reversal index permutation once.
        perm = _endian_reverse_indices(n_qubits)

        if mode == "state":

            def _run_state(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    circuit = _build_circuit(n_qubits, float(phi_val))
                    state = QuantumState(n_qubits)
                    circuit.update_quantum_state(state)
                    # Reverse qubit ordering: little-endian -> big-endian
                    results.append(state.get_vector()[perm])
                return jnp.array(np.stack(results))

            return _run_state

        elif mode == "probs":

            def _run_probs(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    circuit = _build_circuit(n_qubits, float(phi_val))
                    state = QuantumState(n_qubits)
                    circuit.update_quantum_state(state)
                    sv = state.get_vector()
                    # Reverse qubit ordering: little-endian -> big-endian
                    results.append(np.abs(sv) ** 2)
                    results[-1] = results[-1][perm]
                return jnp.array(np.stack(results))

            return _run_probs

        elif mode == "expval":

            # Pre-build per-qubit Z observables in Qulacs qubit order.
            z_obs: List[Observable] = []
            for i in range(n_qubits):
                obs = Observable(n_qubits)
                obs.add_operator(1.0, f"Z {i}")
                z_obs.append(obs)

            def _run_expval(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    circuit = _build_circuit(n_qubits, float(phi_val))
                    state = QuantumState(n_qubits)
                    circuit.update_quantum_state(state)
                    evs = [
                        float(obs.get_expectation_value(state).real)
                        for obs in z_obs
                    ]
                    results.append(evs)
                return jnp.array(np.array(results))

            return _run_expval

        elif mode == "density":

            def _run_density(phi_batch: jnp.ndarray) -> jnp.ndarray:
                results = []
                for phi_val in np.asarray(phi_batch):
                    circuit = _build_circuit(n_qubits, float(phi_val))
                    dm = DensityMatrix(n_qubits)
                    circuit.update_quantum_state(dm)
                    # Reverse qubit ordering on both axes
                    results.append(dm.get_matrix()[np.ix_(perm, perm)])
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