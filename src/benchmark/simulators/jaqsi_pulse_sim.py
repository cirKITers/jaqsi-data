"""Jaqsi pulse-level simulator benchmark adapter."""

from __future__ import annotations

from jaqsi import Evolution, PulseInformation

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

    # Under the RWA every pulse of a single-quadrature envelope is a
    # single-term drive $f(t) H$, which jaqsi solves in closed form by
    # integrating only the scalar pulse area; DRAG rotations keep a second
    # term and always take the matrix ODE.  ``False`` integrates the matrix ODE
    # for every pulse, as the other pulse adapters do.  Set by the runner from
    # the ``closed_form`` config option.
    closed_form = True

    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        # ``pulse_model`` only transcribes the Hadamard and $CRX$ pulse
        # decompositions, so the other families have no pulse-level reference
        # the remaining pulse adapters could be compared against.  The pulse
        # level measures forward simulation only.  The noise mode's channels
        # need nothing pulse-specific: jaqsi applies them to the density
        # matrix between the pulse gates, as it does at gate level.
        return spec.family in PULSE_FAMILIES and mode != "grad"

    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        # PulseInformation keeps the envelope, RWA flag and frame in class-level
        # state; restore the shipped defaults with the envelope the pulse model
        # transcribes.
        PulseInformation.reset_defaults(envelope=self.envelope)
        # A solver default as well, so it is set on every setup and neither
        # configuration can leak into the other.
        Evolution.set_solver_defaults(closed_form=self.closed_form)
        super().setup(spec, mode, optimal_config=optimal_config)
