"""Tests for simulator benchmark adapters.

These tests verify that every simulator adapter produces outputs with the
correct shape and basic mathematical properties (normalisation, valid
ranges, hermiticity, etc.).  They are simulator-agnostic: each test is
parametrised over all available backends.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable 64-bit precision for JAX (matches the benchmark runner).  Without it
# the parameter arrays are float32 and the normalisation checks below fail on
# rounding alone.
jax.config.update("jax_enable_x64", True)

from benchmark.circuits import build_spec
from benchmark.simulators.base import BenchmarkResult
from benchmark.simulators.pennylane_sim import (
    PennylaneBenchmark,
    PennylaneLightningBenchmark,
)
from benchmark.simulators.qiskit_sim import QiskitBenchmark
from benchmark.simulators.qibo_sim import QiboBenchmark
from benchmark.simulators.qulacs_sim import QulacsBenchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(spec, batch_size: int = 1):
    """Return a deterministic ``(inputs, weights)`` pair for *spec*."""
    inputs = jnp.full((batch_size, spec.n_inputs), 0.5)
    weights = jnp.full((spec.n_weights,), 0.3)
    return inputs, weights


def _run(sim, spec, batch_size: int = 1):
    """Execute *sim* on *spec* with the deterministic parameters."""
    return sim.run(*_make_params(spec, batch_size))


# All simulator classes under test.
_ALL_SIMULATORS = [
    pytest.param(PennylaneBenchmark, id="pennylane"),
    pytest.param(PennylaneLightningBenchmark, id="pennylane_lightning"),
    pytest.param(QiskitBenchmark, id="qiskit"),
    pytest.param(QiboBenchmark, id="qibo"),
    pytest.param(QulacsBenchmark, id="qulacs"),
]

# Both circuit families, exercised at a depth greater than one so that the
# layer index takes part in the parameter bookkeeping.
_FAMILIES = ["crx_ring", "hea"]


# ---------------------------------------------------------------------------
# Module-level import & name
# ---------------------------------------------------------------------------

class TestSimulatorImport:
    """Verify that every simulator is importable via the lazy __init__."""

    def test_lazy_import_pennylane(self):
        from benchmark.simulators import PennylaneBenchmark
        assert PennylaneBenchmark is not None

    def test_lazy_import_qiskit(self):
        from benchmark.simulators import QiskitBenchmark
        assert QiskitBenchmark is not None

    def test_lazy_import_qibo(self):
        from benchmark.simulators import QiboBenchmark
        assert QiboBenchmark is not None

    def test_lazy_import_qulacs(self):
        from benchmark.simulators import QulacsBenchmark
        assert QulacsBenchmark is not None

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_has_name(self, sim_cls):
        sim = sim_cls()
        assert isinstance(sim.name, str)
        assert len(sim.name) > 0


# ---------------------------------------------------------------------------
# Mode: state
# ---------------------------------------------------------------------------

class TestStateMode:

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_state_returns_correct_shape(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 2, 2)
        sim.setup(spec, "state")
        result = _run(sim, spec)
        assert result.shape == (1, 2**spec.n_qubits)

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_state_batch(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 2, 2)
        sim.setup(spec, "state")
        result = _run(sim, spec, batch_size=3)
        assert result.shape == (3, 2**spec.n_qubits)

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_state_is_normalized(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 3, 2)
        sim.setup(spec, "state")
        result = _run(sim, spec)
        norm = float(jnp.sum(jnp.abs(result[0]) ** 2))
        assert norm == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Mode: probs
# ---------------------------------------------------------------------------

class TestProbsMode:

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_probs_returns_correct_shape(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 2, 2)
        sim.setup(spec, "probs")
        result = _run(sim, spec)
        assert result.shape == (1, 2**spec.n_qubits)

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_probs_sum_to_one(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 3, 2)
        sim.setup(spec, "probs")
        result = _run(sim, spec)
        assert float(jnp.sum(result[0])) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_probs_non_negative(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 2, 2)
        sim.setup(spec, "probs")
        result = _run(sim, spec, batch_size=2)
        assert jnp.all(result >= 0)


# ---------------------------------------------------------------------------
# Mode: expval
# ---------------------------------------------------------------------------

class TestExpvalMode:

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_expval_returns_correct_shape(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 3, 2)
        sim.setup(spec, "expval")
        result = jnp.asarray(_run(sim, spec))
        # PennyLane returns (n_obs, batch); others return (batch, n_obs).
        if sim.name.startswith("pennylane"):
            assert result.shape == (spec.n_qubits, 1)
        else:
            assert result.shape == (1, spec.n_qubits)

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_expval_in_valid_range(self, sim_cls, family):
        """Z expectation values must be in [-1, 1]."""
        sim = sim_cls()
        spec = build_spec(family, 2, 2)
        sim.setup(spec, "expval")
        result = jnp.asarray(_run(sim, spec))
        assert jnp.all(result >= -1.0 - 1e-10)
        assert jnp.all(result <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Mode: density
# ---------------------------------------------------------------------------

class TestDensityMode:

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_density_returns_correct_shape(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 2, 2)
        dim = 2**spec.n_qubits
        sim.setup(spec, "density")
        result = _run(sim, spec)
        assert result.shape == (1, dim, dim)

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_density_trace_is_one(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 2, 2)
        sim.setup(spec, "density")
        result = _run(sim, spec)
        trace = float(jnp.trace(result[0]).real)
        assert trace == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_density_is_hermitian(self, sim_cls, family):
        sim = sim_cls()
        spec = build_spec(family, 2, 2)
        sim.setup(spec, "density")
        result = _run(sim, spec)
        dm = result[0]
        np.testing.assert_allclose(dm, dm.conj().T, atol=1e-10)


# ---------------------------------------------------------------------------
# Unsupported mode
# ---------------------------------------------------------------------------

class TestUnsupportedMode:

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_unsupported_mode_raises(self, sim_cls):
        """An invalid mode must raise during setup or, at latest, during run."""
        sim = sim_cls()
        spec = build_spec("hea", 2, 1)
        with pytest.raises(Exception):
            sim.setup(spec, "invalid_mode")
            # Some simulators defer the error to run time.
            _run(sim, spec)


# ---------------------------------------------------------------------------
# Integration with timing harness
# ---------------------------------------------------------------------------

class TestHarnessIntegration:
    """Verify every simulator works with the base-class benchmark() method."""

    @staticmethod
    def _make_sweep(spec, n_iters: int = 3, batch_size: int = 1):
        inputs = jnp.ones((n_iters + 1, batch_size, spec.n_inputs)) * 0.5
        weights = jnp.ones((n_iters + 1, spec.n_weights)) * 0.3
        return inputs, weights

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_benchmark_returns_result(self, sim_cls):
        sim = sim_cls()
        spec = build_spec("hea", 2, 2)
        inputs, weights = self._make_sweep(spec)
        result = sim.benchmark(
            spec=spec, mode="probs", all_inputs=inputs, all_weights=weights
        )
        assert isinstance(result, BenchmarkResult)
        assert result.simulator == sim.name
        assert result.mode == "probs"
        assert result.n_qubits == 2
        assert result.circuit == "hea"
        assert result.n_layers == 2
        assert result.mean_ms >= 0
        assert result.std_ms >= 0

    @pytest.mark.parametrize("do_warmup", [True, False])
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_benchmark_warmup(self, sim_cls, do_warmup):
        sim = sim_cls()
        spec = build_spec("hea", 2, 2)
        inputs, weights = self._make_sweep(spec)
        result = sim.benchmark(
            spec=spec,
            mode="state",
            all_inputs=inputs,
            all_weights=weights,
            do_warmup=do_warmup,
        )
        assert result.raw_output is not None


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Verify all simulators are accepted by the configuration system."""

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_all_simulators_in_config(self, sim_cls):
        from benchmark.config import ALL_SIMULATORS
        assert sim_cls().name in ALL_SIMULATORS

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_config_accepts_each_simulator(self, sim_cls):
        from benchmark.config import load_config
        name = sim_cls().name
        cfg = load_config(overrides=[f"simulators=[{name}]"])
        assert cfg.simulators == [name]


# ---------------------------------------------------------------------------
# Optimal-config equivalence
# ---------------------------------------------------------------------------

class TestOptimalConfigEquivalence:
    """optimal_config must not change numerical results, only performance."""

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    @pytest.mark.parametrize("mode", ["probs", "expval", "state", "density"])
    def test_optimal_matches_default(self, sim_cls, mode, family):
        spec = build_spec(family, 2, 2)
        inputs, weights = _make_params(spec)

        default_sim = sim_cls()
        default_sim.setup(spec, mode, optimal_config=False)
        default_out = np.asarray(default_sim.run(inputs, weights))

        optimal_sim = sim_cls()
        optimal_sim.setup(spec, mode, optimal_config=True)
        optimal_out = np.asarray(optimal_sim.run(inputs, weights))

        np.testing.assert_allclose(default_out, optimal_out, atol=1e-8)

    @pytest.mark.parametrize("mode", ["probs", "expval", "state", "density"])
    def test_parallel_aer_experiments_match_default(self, mode, monkeypatch):
        """Aer runs the batch's experiments in parallel, one per thread."""
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        spec = build_spec("hea", 3, 2)
        # Distinct samples, so a reordered batch shows.
        inputs = jnp.linspace(0.1, 2.0, 4 * spec.n_inputs).reshape(4, -1)
        weights = jnp.full((spec.n_weights,), 0.3)

        default_sim = QiskitBenchmark()
        default_sim.setup(spec, mode, optimal_config=False)
        optimal_sim = QiskitBenchmark()
        optimal_sim.setup(spec, mode, optimal_config=True)

        np.testing.assert_allclose(
            np.asarray(default_sim.run(inputs, weights)),
            np.asarray(optimal_sim.run(inputs, weights)),
            atol=1e-8,
        )


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

class TestQiboThreading:
    """Qibo is the one backend that has to be pinned after construction.

    ``optimal_config`` selects qibojit in every mode and the default
    configuration numpy, so that flag decides which backend is under test.
    """

    def test_numpy_backend_accepts_a_multi_thread_pinning(self, monkeypatch):
        """The numpy backend rejects set_threads above one, so it is skipped.

        Without that guard, every multi-threaded run of the default
        configuration fails during setup rather than producing a measurement.
        """
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        sim = QiboBenchmark()
        spec = build_spec("hea", 2, 1)
        sim.setup(spec, "expval", optimal_config=False)
        result = _run(sim, spec)
        assert result.shape == (1, spec.n_qubits)

    @pytest.mark.parametrize("threads", [1, 2])
    def test_qibojit_backend_is_pinned(self, monkeypatch, threads):
        """qibojit ignores the environment, so the adapter pins it directly."""
        import qibo

        monkeypatch.setenv("OMP_NUM_THREADS", str(threads))
        sim = QiboBenchmark()
        sim.setup(build_spec("hea", 2, 1), "density", optimal_config=True)
        assert qibo.get_threads() == threads

    def test_state_vector_modes_use_qibojit(self, monkeypatch):
        """At one thread qibojit is ahead of numpy in the state-vector modes too."""
        import qibo

        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        sim = QiboBenchmark()
        sim.setup(build_spec("hea", 2, 1), "expval", optimal_config=True)
        assert qibo.get_backend().name == "qibojit"
