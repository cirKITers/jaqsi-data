"""Tests for the Qulacs simulator benchmark adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.simulators.base import BenchmarkResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phi_batch(batch_size: int = 1) -> jnp.ndarray:
    """Return a small batch of phi values."""
    return jnp.array([0.5] * batch_size)


# ---------------------------------------------------------------------------
# Module-level import & name
# ---------------------------------------------------------------------------

class TestQulacsBenchmarkImport:
    """Verify that QulacsBenchmark is importable via the lazy __init__."""

    def test_lazy_import(self):
        from benchmark.simulators import QulacsBenchmark
        assert QulacsBenchmark is not None

    def test_name(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        assert sim.name == "qulacs"


# ---------------------------------------------------------------------------
# Circuit building
# ---------------------------------------------------------------------------

class TestBuildCircuit:
    """Test the _build_circuit helper builds the expected gate sequence."""

    def test_circuit_gate_count(self):
        from benchmark.simulators.qulacs_sim import _build_circuit
        n_qubits = 3
        circuit = _build_circuit(n_qubits, phi=0.5)
        # n_qubits H gates + n_qubits CRX gates = 2 * n_qubits
        assert circuit.get_gate_count() == 2 * n_qubits

    def test_circuit_qubit_count(self):
        from benchmark.simulators.qulacs_sim import _build_circuit
        circuit = _build_circuit(4, phi=1.0)
        assert circuit.get_qubit_count() == 4


# ---------------------------------------------------------------------------
# CRX gate construction
# ---------------------------------------------------------------------------

class TestMakeCrxGate:
    """Test that _make_crx_gate produces a valid controlled-RX gate."""

    def test_crx_gate_returns_gate_object(self):
        from benchmark.simulators.qulacs_sim import _make_crx_gate
        gate = _make_crx_gate(control=0, target=1, angle=0.5)
        # Should be a DenseMatrix gate with a 2x2 matrix (excludes control)
        mat = gate.get_matrix()
        assert mat.shape == (2, 2)

    def test_crx_identity_at_zero_angle(self):
        """CRX(0) should be the identity on the target qubit."""
        from benchmark.simulators.qulacs_sim import _make_crx_gate
        gate = _make_crx_gate(control=0, target=1, angle=0.0)
        mat = gate.get_matrix()
        np.testing.assert_allclose(mat, np.eye(2), atol=1e-12)

    def test_crx_has_control_qubit(self):
        from benchmark.simulators.qulacs_sim import _make_crx_gate
        gate = _make_crx_gate(control=0, target=1, angle=0.5)
        assert gate.get_control_index_list() == [0]


# ---------------------------------------------------------------------------
# Mode: state
# ---------------------------------------------------------------------------

class TestStateMode:

    def test_state_returns_correct_shape(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        n_qubits = 2
        sim.setup(n_qubits, "state")
        result = sim.run(_make_phi_batch(batch_size=1))
        assert result.shape == (1, 2**n_qubits)

    def test_state_batch(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        n_qubits = 2
        sim.setup(n_qubits, "state")
        result = sim.run(_make_phi_batch(batch_size=3))
        assert result.shape == (3, 2**n_qubits)

    def test_state_is_normalized(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        sim.setup(3, "state")
        result = sim.run(_make_phi_batch(batch_size=1))
        norm = float(jnp.sum(jnp.abs(result[0]) ** 2))
        assert norm == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Mode: probs
# ---------------------------------------------------------------------------

class TestProbsMode:

    def test_probs_returns_correct_shape(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        n_qubits = 2
        sim.setup(n_qubits, "probs")
        result = sim.run(_make_phi_batch(batch_size=1))
        assert result.shape == (1, 2**n_qubits)

    def test_probs_sum_to_one(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        sim.setup(3, "probs")
        result = sim.run(_make_phi_batch(batch_size=1))
        assert float(jnp.sum(result[0])) == pytest.approx(1.0, abs=1e-10)

    def test_probs_non_negative(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        sim.setup(2, "probs")
        result = sim.run(_make_phi_batch(batch_size=2))
        assert jnp.all(result >= 0)


# ---------------------------------------------------------------------------
# Mode: expval
# ---------------------------------------------------------------------------

class TestExpvalMode:

    def test_expval_returns_correct_shape(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        n_qubits = 3
        sim.setup(n_qubits, "expval")
        result = sim.run(_make_phi_batch(batch_size=1))
        # (batch, n_qubits)
        assert result.shape == (1, n_qubits)

    def test_expval_in_valid_range(self):
        """Z expectation values must be in [-1, 1]."""
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        sim.setup(2, "expval")
        result = sim.run(_make_phi_batch(batch_size=1))
        assert jnp.all(result >= -1.0 - 1e-10)
        assert jnp.all(result <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Mode: density
# ---------------------------------------------------------------------------

class TestDensityMode:

    def test_density_returns_correct_shape(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        n_qubits = 2
        dim = 2**n_qubits
        sim.setup(n_qubits, "density")
        result = sim.run(_make_phi_batch(batch_size=1))
        assert result.shape == (1, dim, dim)

    def test_density_trace_is_one(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        sim.setup(2, "density")
        result = sim.run(_make_phi_batch(batch_size=1))
        trace = float(jnp.trace(result[0]).real)
        assert trace == pytest.approx(1.0, abs=1e-10)

    def test_density_is_hermitian(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        sim.setup(2, "density")
        result = sim.run(_make_phi_batch(batch_size=1))
        dm = result[0]
        np.testing.assert_allclose(dm, dm.conj().T, atol=1e-10)


# ---------------------------------------------------------------------------
# Unsupported mode
# ---------------------------------------------------------------------------

class TestUnsupportedMode:

    def test_unsupported_mode_raises(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        with pytest.raises(ValueError, match="Unsupported mode"):
            sim.setup(2, "invalid_mode")


# ---------------------------------------------------------------------------
# Integration with timing harness
# ---------------------------------------------------------------------------

class TestHarnessIntegration:
    """Verify QulacsBenchmark works with the base-class benchmark() method."""

    def _make_phis(self, n_iters: int = 3, batch_size: int = 1) -> jnp.ndarray:
        return jnp.ones((n_iters + 1, batch_size)) * 0.5

    def test_benchmark_returns_result(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        result = sim.benchmark(n_qubits=2, mode="probs", all_phis=self._make_phis())
        assert isinstance(result, BenchmarkResult)
        assert result.simulator == "qulacs"
        assert result.mode == "probs"
        assert result.n_qubits == 2
        assert result.mean_ms >= 0
        assert result.std_ms >= 0

    def test_benchmark_warmup(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        # Should not raise
        result = sim.benchmark(
            n_qubits=2, mode="state", all_phis=self._make_phis(), do_warmup=True
        )
        assert result.raw_output is not None

    def test_benchmark_no_warmup(self):
        from benchmark.simulators.qulacs_sim import QulacsBenchmark
        sim = QulacsBenchmark()
        result = sim.benchmark(
            n_qubits=2, mode="state", all_phis=self._make_phis(), do_warmup=False
        )
        assert result.raw_output is not None


# ---------------------------------------------------------------------------
# Config & runner awareness
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Verify that 'qulacs' is accepted as a valid simulator name."""

    def test_qulacs_in_all_simulators(self):
        from benchmark.config import ALL_SIMULATORS
        assert "qulacs" in ALL_SIMULATORS

    def test_config_accepts_qulacs(self):
        from benchmark.config import load_config
        cfg = load_config(overrides=["simulators=[qulacs]"])
        assert cfg.simulators == ["qulacs"]

    def test_default_config_includes_qulacs(self):
        from benchmark.config import load_config
        cfg = load_config()
        assert "qulacs" in cfg.simulators