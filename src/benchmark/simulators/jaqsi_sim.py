"""Jaqsi simulator benchmark adapter."""

from __future__ import annotations

import jax.numpy as jnp

from jaqsi import Gates, PauliZ, Script

from benchmark.simulators.base import SimulatorBenchmark, Mode


class JaqsiBenchmark(SimulatorBenchmark):
    name = "jaqsi"

    def __init__(self) -> None:
        self._script: Script | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, n_qubits: int, mode: Mode, *, optimal_config: bool = False) -> None:
        self._n_qubits = n_qubits
        self._mode = mode

        def circuit(phi: float) -> None:
            for i in range(n_qubits):
                Gates.H(wires=i)
            for i in range(n_qubits):
                Gates.CRX(w=phi, wires=[i, (i + 1) % n_qubits])

        self._script = Script(f=circuit)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _execute(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._script is not None
        return self._script.execute(
            type=self._mode,
            obs=[PauliZ(wires=i, record=False) for i in range(self._n_qubits)],
            args=(phi,),
            in_axes=(0,),
        )

    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        return self._execute(phi)

    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        return self._execute(phi)