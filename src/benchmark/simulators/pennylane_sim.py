"""PennyLane simulator benchmark adapter."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import pennylane as qml

from benchmark.simulators.base import SimulatorBenchmark, Mode


class PennylaneBenchmark(SimulatorBenchmark):
    name = "pennylane"

    def __init__(self) -> None:
        self._circuit_fn: Callable | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, n_qubits: int, mode: Mode) -> None:
        self._n_qubits = n_qubits
        self._mode = mode

        dev = qml.device("default.qubit", wires=n_qubits)

        return_map: dict[str, Callable] = {
            "density": lambda: qml.density_matrix(wires=range(n_qubits)),
            "state": lambda: qml.state(),
            "probs": lambda: qml.probs(wires=range(n_qubits)),
            "expval": lambda: [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)],
        }

        @qml.qnode(dev, interface="jax")
        def circuit(phi):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.CRX(phi, wires=[i, (i + 1) % n_qubits])
            return return_map[mode]()

        self._circuit_fn = circuit

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._circuit_fn is not None
        return self._circuit_fn(phi)

    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._circuit_fn is not None
        return self._circuit_fn(phi)