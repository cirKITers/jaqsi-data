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


def _endian_reverse_indices(n_qubits: int) -> np.ndarray:
    """Return an index array that maps Qiskit's little-endian basis order
    to big-endian order (used by JAQSI, PennyLane, Qibo).

    Qiskit labels qubit 0 as the *least*-significant bit, so basis state
    index ``b_{n-1}…b_1 b_0`` in Qiskit corresponds to
    ``b_0 b_1 … b_{n-1}`` in big-endian convention.  This function
    returns a permutation that re-sorts a length-2**n vector from
    little-endian to big-endian.
    """
    N = 1 << n_qubits
    indices = np.zeros(N, dtype=int)
    for i in range(N):
        # Reverse the bit pattern of i (n_qubits wide)
        rev = int(f"{i:0{n_qubits}b}"[::-1], 2)
        indices[rev] = i
    return indices


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

        # Pre-compute the endian-reversal index permutation once.
        perm = _endian_reverse_indices(n_qubits)

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

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(phi)

    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(phi)