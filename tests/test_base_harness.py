"""Tests for the SimulatorBenchmark base-class timing harness."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from benchmark.simulators.base import BenchmarkResult, Mode, SimulatorBenchmark


class DummySimulator(SimulatorBenchmark):
    """Minimal concrete simulator for testing the shared harness."""

    name = "dummy"

    def __init__(self) -> None:
        self.setup_calls: list[tuple[int, Mode]] = []
        self.warmup_calls: int = 0
        self.run_calls: int = 0

    def setup(self, n_qubits: int, mode: Mode) -> None:
        self.setup_calls.append((n_qubits, mode))

    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        self.warmup_calls += 1
        return phi

    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        self.run_calls += 1
        return phi * 2.0


class TestBenchmarkHarness:
    def _make_phis(self, n_iters: int = 5, batch_size: int = 1) -> jnp.ndarray:
        return jnp.ones((n_iters + 1, batch_size))

    def test_returns_benchmark_result(self):
        sim = DummySimulator()
        result = sim.benchmark(n_qubits=2, mode="probs", all_phis=self._make_phis())
        assert isinstance(result, BenchmarkResult)

    def test_result_fields(self):
        sim = DummySimulator()
        phis = self._make_phis(n_iters=5, batch_size=3)
        result = sim.benchmark(n_qubits=4, mode="expval", all_phis=phis)
        assert result.simulator == "dummy"
        assert result.mode == "expval"
        assert result.n_qubits == 4
        assert result.batch_size == 3
        assert result.n_iters == 5
        assert result.mean_ms >= 0
        assert result.std_ms >= 0

    def test_setup_called_once(self):
        sim = DummySimulator()
        sim.benchmark(n_qubits=2, mode="probs", all_phis=self._make_phis())
        assert sim.setup_calls == [(2, "probs")]

    def test_warmup_called_once(self):
        sim = DummySimulator()
        sim.benchmark(n_qubits=2, mode="probs", all_phis=self._make_phis(n_iters=3))
        assert sim.warmup_calls == 1

    def test_warmup_skipped(self):
        sim = DummySimulator()
        sim.benchmark(
            n_qubits=2, mode="probs", all_phis=self._make_phis(), do_warmup=False
        )
        assert sim.warmup_calls == 0

    def test_run_called_n_iters_times(self):
        sim = DummySimulator()
        n_iters = 7
        sim.benchmark(n_qubits=2, mode="state", all_phis=self._make_phis(n_iters=n_iters))
        assert sim.run_calls == n_iters

    def test_raw_output_is_last_run(self):
        sim = DummySimulator()
        phis = self._make_phis(n_iters=3, batch_size=1)
        result = sim.benchmark(n_qubits=2, mode="probs", all_phis=phis)
        # DummySimulator.run returns phi * 2.0; the last iteration uses phis[2]
        expected = phis[2] * 2.0
        np.testing.assert_allclose(result.raw_output, expected)