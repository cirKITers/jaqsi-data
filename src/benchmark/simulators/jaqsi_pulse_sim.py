"""Jaqsi pulse-level simulator benchmark adapter."""

from __future__ import annotations

from jaqsi import Gates, PulseInformation, Script

from benchmark.simulators.base import Mode
from benchmark.simulators.jaqsi_sim import JaqsiBenchmark


class JaqsiPulseBenchmark(JaqsiBenchmark):
    """Runs the benchmark circuit through jaqsi's pulse-level gate set.

    Only the circuit construction differs from the gate-level adapter: each
    gate expands into the pulse schedule transcribed in
    :mod:`benchmark.simulators.pulse_model`, which the other pulse adapters
    integrate themselves.
    """

    name = "jaqsi_pulse"

    def setup(self, n_qubits: int, mode: Mode, *, optimal_config: bool = False) -> None:
        self._n_qubits = n_qubits
        self._mode = mode

        # PulseInformation keeps the envelope, RWA flag and frame in class-level
        # state; restore the shipped defaults the pulse model transcribes.
        PulseInformation.reset_defaults()

        def circuit(phi: float) -> None:
            for i in range(n_qubits):
                Gates.H(wires=i, pulse=True)
            for i in range(n_qubits):
                Gates.CRX(w=phi, wires=[i, (i + 1) % n_qubits], pulse=True)

        self._script = Script(f=circuit)
