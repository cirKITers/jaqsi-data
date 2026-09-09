"""Cross-validation tests for the pulse-level simulator adapters.

Every pulse adapter integrates the same schedule, so all of them are compared
against jaqsi_pulse, in both the default and the optimized configuration: the
two differ in the dimension of the integrated ODE and are benchmarked
separately, so both have to reproduce the reference.  The tolerance reflects
the accumulated ODE solver error rather than machine precision, and the qubit
counts are kept small because pennylane spends roughly 190 ms on each of the
21 segments per qubit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.circuits import build_spec
from benchmark.runner import REFERENCE_BY_LEVEL, SIMULATOR_REGISTRY, _level
from benchmark.simulators.dynamiqs_pulse_sim import DynamiqsPulseBenchmark
from benchmark.simulators.jaqsi_pulse_sim import JaqsiPulseBenchmark
from benchmark.simulators.pennylane_pulse_sim import PennylanePulseBenchmark
from benchmark.simulators.qutip_pulse_sim import QutipPulseBenchmark

# Enable 64-bit precision for JAX (matches the benchmark runner)
jax.config.update("jax_enable_x64", True)

# Tighter than ``precision`` in configs/pulse.yaml, which has to cover the
# solver error accumulated over a full sweep; these tests run at two qubits,
# where the deviation stays around $10^{-7}$.
PRECISION = 1e-6

N_QUBITS = 2
MODES = ["probs", "expval", "state", "density"]

# The pulse model only transcribes the Hadamard and CRX decompositions.
SPEC = build_spec("crx_ring", N_QUBITS, 1)


def _inputs(batch_size: int = 1) -> jnp.ndarray:
    """Return a reproducible batch of CRX angle vectors."""
    return jax.random.uniform(
        jax.random.PRNGKey(3),
        (batch_size, SPEC.n_inputs),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )


# crx_ring carries all its parameters in ``inputs``.
WEIGHTS = jnp.zeros(0)

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
        sim.setup(SPEC, mode)
        results[mode] = np.asarray(sim.run(_inputs(), WEIGHTS))
    return results


# ------------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------------

class TestPulseCrossValidation:
    """Compare every pulse simulator against jaqsi_pulse for every mode."""

    @pytest.mark.parametrize("optimal", [False, True], ids=["default", "optimal"])
    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("mode", MODES)
    def test_matches_reference(self, reference, sim_cls, mode, optimal):
        sim = sim_cls()
        sim.setup(SPEC, mode, optimal_config=optimal)
        result = np.asarray(sim.run(_inputs(), WEIGHTS))

        assert result.shape == reference[mode].shape
        np.testing.assert_allclose(
            result, reference[mode], atol=PRECISION,
            err_msg=f"{mode} mismatch: {sim.name} vs jaqsi_pulse",
        )

    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    def test_batch_matches_reference(self, sim_cls):
        batch = _inputs(batch_size=2)
        ref = JaqsiPulseBenchmark()
        ref.setup(SPEC, "state")
        sim = sim_cls()
        sim.setup(SPEC, "state")

        np.testing.assert_allclose(
            np.asarray(sim.run(batch, WEIGHTS)),
            np.asarray(ref.run(batch, WEIGHTS)),
            atol=PRECISION,
            err_msg=f"state batch mismatch: {sim.name} vs jaqsi_pulse",
        )


# Each simulator paired with the tolerance its two configurations agree to.
# PennyLane and dynamiqs run the same integrator either way, so they agree far
# more tightly than the cross-simulator tolerance.  QuTiP switches between
# ``sesolve`` and ``propagator``, which are different routines, so its two
# configurations only agree to the solver-limited tolerance.
_OPTIMAL_CONFIG_SIMULATORS = [
    pytest.param(QutipPulseBenchmark, PRECISION, id="qutip_pulse"),
    pytest.param(DynamiqsPulseBenchmark, 1e-8, id="dynamiqs_pulse"),
    pytest.param(PennylanePulseBenchmark, 1e-8, id="pennylane_pulse"),
]


class TestPulseOptimalConfigEquivalence:
    """optimal_config must not change numerical results, only performance."""

    @pytest.mark.parametrize("sim_cls, tolerance", _OPTIMAL_CONFIG_SIMULATORS)
    @pytest.mark.parametrize("mode", MODES)
    def test_optimal_matches_default(self, sim_cls, tolerance, mode):
        inputs = _inputs()

        default_sim = sim_cls()
        default_sim.setup(SPEC, mode, optimal_config=False)
        default_out = np.asarray(default_sim.run(inputs, WEIGHTS))

        optimal_sim = sim_cls()
        optimal_sim.setup(SPEC, mode, optimal_config=True)
        optimal_out = np.asarray(optimal_sim.run(inputs, WEIGHTS))

        np.testing.assert_allclose(default_out, optimal_out, atol=tolerance)


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


class TestPulseSupport:
    """The pulse adapters only accept what the pulse model transcribes."""

    @pytest.mark.parametrize(
        "sim_cls", _OTHER_SIMULATORS + [pytest.param(JaqsiPulseBenchmark, id="jaqsi_pulse")]
    )
    def test_rejects_families_without_a_transcription(self, sim_cls):
        assert not sim_cls().supports(build_spec("hea", 2, 1), "state")

    @pytest.mark.parametrize(
        "sim_cls", _OTHER_SIMULATORS + [pytest.param(JaqsiPulseBenchmark, id="jaqsi_pulse")]
    )
    def test_rejects_the_gradient_mode(self, sim_cls):
        assert not sim_cls().supports(SPEC, "grad")

    @pytest.mark.parametrize(
        "sim_cls", _OTHER_SIMULATORS + [pytest.param(JaqsiPulseBenchmark, id="jaqsi_pulse")]
    )
    def test_accepts_the_pulse_family(self, sim_cls):
        assert sim_cls().supports(SPEC, "state")
