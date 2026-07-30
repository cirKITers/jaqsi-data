"""Cross-validation tests for the pulse-level simulator adapters.

Every pulse adapter integrates the same schedule, so all of them are compared
against jaqsi_pulse.  The tolerance reflects the accumulated ODE solver error
rather than machine precision, and the qubit counts are kept small because
pennylane spends roughly 190 ms on each of the 21 segments per qubit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.runner import REFERENCE_BY_LEVEL, SIMULATOR_REGISTRY, _level
from benchmark.simulators.dynamiqs_pulse_sim import DynamiqsPulseBenchmark
from benchmark.simulators.jaqsi_pulse_sim import JaqsiPulseBenchmark
from benchmark.simulators.pennylane_pulse_sim import PennylanePulseBenchmark
from benchmark.simulators.qutip_pulse_sim import QutipPulseBenchmark

# Enable 64-bit precision for JAX (matches the benchmark runner)
jax.config.update("jax_enable_x64", True)

# Matches ``precision`` in configs/pulse.yaml
PRECISION = 1e-6

N_QUBITS = 2
MODES = ["probs", "expval", "state", "density"]

# Pulse simulators under test (excluding jaqsi_pulse, which is the reference).
_OTHER_SIMULATORS = [
    pytest.param(QutipPulseBenchmark, id="qutip_pulse"),
    pytest.param(DynamiqsPulseBenchmark, id="dynamiqs_pulse"),
    pytest.param(PennylanePulseBenchmark, id="pennylane_pulse"),
]


@pytest.fixture(scope="module")
def reference():
    """Return jaqsi_pulse results for every mode, computed once."""
    sim = JaqsiPulseBenchmark()
    results = {}
    for mode in MODES:
        sim.setup(N_QUBITS, mode)
        results[mode] = np.asarray(sim.run(jnp.array([0.5])))
    return results


# ------------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------------

class TestPulseCrossValidation:
    """Compare every pulse simulator against jaqsi_pulse for every mode."""

    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("mode", MODES)
    def test_matches_reference(self, reference, sim_cls, mode):
        sim = sim_cls()
        sim.setup(N_QUBITS, mode)
        result = np.asarray(sim.run(jnp.array([0.5])))

        assert result.shape == reference[mode].shape
        np.testing.assert_allclose(
            result, reference[mode], atol=PRECISION,
            err_msg=f"{mode} mismatch: {sim.name} vs jaqsi_pulse",
        )

    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    def test_batch_matches_reference(self, sim_cls):
        batch = jnp.array([0.5, -1.2])
        ref = JaqsiPulseBenchmark()
        ref.setup(N_QUBITS, "state")
        sim = sim_cls()
        sim.setup(N_QUBITS, "state")

        np.testing.assert_allclose(
            np.asarray(sim.run(batch)),
            np.asarray(ref.run(batch)),
            atol=PRECISION,
            err_msg=f"state batch mismatch: {sim.name} vs jaqsi_pulse",
        )


class TestPulseOptimalConfigEquivalence:
    """optimal_config must not change numerical results, only performance."""

    @pytest.mark.parametrize("mode", MODES)
    def test_pennylane_optimal_matches_default(self, mode):
        phi = jnp.array([0.5])

        default_sim = PennylanePulseBenchmark()
        default_sim.setup(N_QUBITS, mode, optimal_config=False)
        default_out = np.asarray(default_sim.run(phi))

        optimal_sim = PennylanePulseBenchmark()
        optimal_sim.setup(N_QUBITS, mode, optimal_config=True)
        optimal_out = np.asarray(optimal_sim.run(phi))

        # Both paths integrate the same ODEs with the same solver, so they agree
        # far more tightly than the cross-simulator tolerance.
        np.testing.assert_allclose(default_out, optimal_out, atol=1e-8)


# ------------------------------------------------------------------
# Registry wiring
# ------------------------------------------------------------------

class TestPulseRegistry:
    """The runner resolves pulse adapters and their reference correctly."""

    @pytest.mark.parametrize(
        "name",
        ["jaqsi_pulse", "pennylane_pulse", "qutip_pulse", "dynamiqs_pulse"],
    )
    def test_registered(self, name):
        assert name in SIMULATOR_REGISTRY
        assert _level(name) == "pulse"

    def test_gate_simulators_keep_gate_level(self):
        for name in ["jaqsi", "pennylane", "qiskit", "qibo", "qulacs"]:
            assert _level(name) == "gate"

    def test_reference_per_level(self):
        assert REFERENCE_BY_LEVEL["pulse"] == "jaqsi_pulse"
        assert REFERENCE_BY_LEVEL["gate"] == "jaqsi"

    def test_config_accepts_pulse_simulators(self):
        from benchmark.config import load_config

        cfg = load_config(overrides=["simulators=[jaqsi_pulse,qutip_pulse]"])
        assert cfg.simulators == ["jaqsi_pulse", "qutip_pulse"]
