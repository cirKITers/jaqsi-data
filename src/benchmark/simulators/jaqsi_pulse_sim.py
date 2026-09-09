"""Jaqsi pulse-level simulator benchmark adapter."""

from __future__ import annotations

from jaqsi import PulseInformation

from benchmark.circuits import PULSE_FAMILIES, CircuitSpec
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

    pulse = True

    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        # ``pulse_model`` only transcribes the Hadamard and $CRX$ pulse
        # decompositions, so the other families have no pulse-level reference
        # the remaining pulse adapters could be compared against.  The pulse
        # level measures forward simulation only.
        return spec.family in PULSE_FAMILIES and mode != "grad"

    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        # PulseInformation keeps the envelope, RWA flag and frame in class-level
        # state; restore the shipped defaults the pulse model transcribes.
        PulseInformation.reset_defaults()
        super().setup(spec, mode, optimal_config=optimal_config)
