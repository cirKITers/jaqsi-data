"""Tests for simulator benchmark adapters.

These tests verify that every simulator adapter produces outputs with the
correct shape and basic mathematical properties (normalisation, valid
ranges, hermiticity, etc.).  They are simulator-agnostic: each test is
parametrised over all available backends.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.simulators.base import BenchmarkResult
from benchmark.simulators.pennylane_sim import PennylaneBenchmark
from benchmark.simulators.qiskit_sim import QiskitBenchmark
from benchmark.simulators.qibo_sim import QiboBenchmark
from benchmark.simulators.qulacs_sim import QulacsBenchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phi_batch(batch_size: int = 1) -> jnp.ndarray:
    """Return a small batch of phi values."""
    return jnp.array([0.5] * batch_size)


# All simulator classes under test.
_ALL_SIMULATORS = [
    pytest.param(PennylaneBenchmark, id="pennylane"),
    pytest.param(QiskitBenchmark, id="qiskit"),
    pytest.param(QiboBenchmark, id="qibo"),
    pytest.param(QulacsBenchmark, id="qulacs"),
]


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

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_state_returns_correct_shape(self, sim_cls):
        sim = sim_cls()
        n_qubits = 2
        sim.setup(n_qubits, "state")
        result = sim.run(_make_phi_batch(batch_size=1))
        assert result.shape == (1, 2**n_qubits)

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_state_batch(self, sim_cls):
        sim = sim_cls()
        n_qubits = 2
        sim.setup(n_qubits, "state")
        result = sim.run(_make_phi_batch(batch_size=3))
        assert result.shape == (3, 2**n_qubits)

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_state_is_normalized(self, sim_cls):
        sim = sim_cls()
        sim.setup(3, "state")
        result = sim.run(_make_phi_batch(batch_size=1))
        norm = float(jnp.sum(jnp.abs(result[0]) ** 2))
        assert norm == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Mode: probs
# ---------------------------------------------------------------------------

class TestProbsMode:

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_probs_returns_correct_shape(self, sim_cls):
        sim = sim_cls()
        n_qubits = 2
        sim.setup(n_qubits, "probs")
        result = sim.run(_make_phi_batch(batch_size=1))
        assert result.shape == (1, 2**n_qubits)

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_probs_sum_to_one(self, sim_cls):
        sim = sim_cls()
        sim.setup(3, "probs")
        result = sim.run(_make_phi_batch(batch_size=1))
        assert float(jnp.sum(result[0])) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_probs_non_negative(self, sim_cls):
        sim = sim_cls()
        sim.setup(2, "probs")
        result = sim.run(_make_phi_batch(batch_size=2))
        assert jnp.all(result >= 0)


# ---------------------------------------------------------------------------
# Mode: expval
# ---------------------------------------------------------------------------

class TestExpvalMode:

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_expval_returns_correct_shape(self, sim_cls):
        sim = sim_cls()
        n_qubits = 3
        sim.setup(n_qubits, "expval")
        result = jnp.asarray(sim.run(_make_phi_batch(batch_size=1)))
        # PennyLane returns (n_obs, batch); others return (batch, n_obs).
        if sim.name == "pennylane":
            assert result.shape == (n_qubits, 1)
        else:
            assert result.shape == (1, n_qubits)

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_expval_in_valid_range(self, sim_cls):
        """Z expectation values must be in [-1, 1]."""
        sim = sim_cls()
        sim.setup(2, "expval")
        result = jnp.asarray(sim.run(_make_phi_batch(batch_size=1)))
        assert jnp.all(result >= -1.0 - 1e-10)
        assert jnp.all(result <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Mode: density
# ---------------------------------------------------------------------------

class TestDensityMode:

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_density_returns_correct_shape(self, sim_cls):
        sim = sim_cls()
        n_qubits = 2
        dim = 2**n_qubits
        sim.setup(n_qubits, "density")
        result = sim.run(_make_phi_batch(batch_size=1))
        assert result.shape == (1, dim, dim)

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_density_trace_is_one(self, sim_cls):
        sim = sim_cls()
        sim.setup(2, "density")
        result = sim.run(_make_phi_batch(batch_size=1))
        trace = float(jnp.trace(result[0]).real)
        assert trace == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_density_is_hermitian(self, sim_cls):
        sim = sim_cls()
        sim.setup(2, "density")
        result = sim.run(_make_phi_batch(batch_size=1))
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
        with pytest.raises(Exception):
            sim.setup(2, "invalid_mode")
            # Some simulators defer the error to run time.
            sim.run(_make_phi_batch(batch_size=1))


# ---------------------------------------------------------------------------
# Integration with timing harness
# ---------------------------------------------------------------------------

class TestHarnessIntegration:
    """Verify every simulator works with the base-class benchmark() method."""

    @staticmethod
    def _make_phis(n_iters: int = 3, batch_size: int = 1) -> jnp.ndarray:
        return jnp.ones((n_iters + 1, batch_size)) * 0.5

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_benchmark_returns_result(self, sim_cls):
        sim = sim_cls()
        result = sim.benchmark(n_qubits=2, mode="probs", all_phis=self._make_phis())
        assert isinstance(result, BenchmarkResult)
        assert result.simulator == sim.name
        assert result.mode == "probs"
        assert result.n_qubits == 2
        assert result.mean_ms >= 0
        assert result.std_ms >= 0

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_benchmark_warmup(self, sim_cls):
        sim = sim_cls()
        result = sim.benchmark(
            n_qubits=2, mode="state", all_phis=self._make_phis(), do_warmup=True,
        )
        assert result.raw_output is not None

    @pytest.mark.parametrize("sim_cls", _ALL_SIMULATORS)
    def test_benchmark_no_warmup(self, sim_cls):
        sim = sim_cls()
        result = sim.benchmark(
            n_qubits=2, mode="state", all_phis=self._make_phis(), do_warmup=False,
        )
        assert result.raw_output is not None


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Verify all simulators are accepted by the configuration system."""

    def test_all_simulators_in_config(self):
        from benchmark.config import ALL_SIMULATORS
        for sim_cls in [PennylaneBenchmark, QiskitBenchmark, QiboBenchmark, QulacsBenchmark]:
            assert sim_cls().name in ALL_SIMULATORS

    def test_config_accepts_each_simulator(self):
        from benchmark.config import load_config
        for sim_cls in [PennylaneBenchmark, QiskitBenchmark, QiboBenchmark, QulacsBenchmark]:
            name = sim_cls().name
            cfg = load_config(overrides=[f"simulators=[{name}]"])
            assert cfg.simulators == [name]