"""Tests for the SimulatorBenchmark base-class timing harness."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from benchmark.circuits import CircuitSpec, build_spec
from benchmark.simulators.base import BenchmarkResult, Mode, SimulatorBenchmark


class DummySimulator(SimulatorBenchmark):
    """Minimal concrete simulator for testing the shared harness."""

    name = "dummy"

    def __init__(self) -> None:
        self.setup_calls: list[tuple[CircuitSpec, Mode]] = []
        self.warmup_calls: int = 0
        self.run_calls: int = 0

    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        self.setup_calls.append((spec, mode))

    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        self.warmup_calls += 1
        return inputs

    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        self.run_calls += 1
        return inputs * 2.0


class TestBenchmarkHarness:
    def _make_sweep(
        self, spec: CircuitSpec, n_iters: int = 5, batch_size: int = 1
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        inputs = jnp.ones((n_iters + 1, batch_size, spec.n_inputs))
        weights = jnp.ones((n_iters + 1, spec.n_weights))
        return inputs, weights

    def _benchmark(self, sim, spec, mode="probs", **kwargs):
        n_iters = kwargs.pop("n_iters", 5)
        batch_size = kwargs.pop("batch_size", 1)
        inputs, weights = self._make_sweep(spec, n_iters, batch_size)
        return sim.benchmark(
            spec=spec, mode=mode, all_inputs=inputs, all_weights=weights, **kwargs
        )

    def test_returns_benchmark_result(self):
        sim = DummySimulator()
        result = self._benchmark(sim, build_spec("hea", 2, 1))
        assert isinstance(result, BenchmarkResult)

    def test_result_fields(self):
        sim = DummySimulator()
        spec = build_spec("hea", 4, 3)
        result = self._benchmark(sim, spec, mode="expval", n_iters=5, batch_size=3)
        assert result.simulator == "dummy"
        assert result.mode == "expval"
        assert result.n_qubits == 4
        assert result.circuit == "hea"
        assert result.n_layers == 3
        assert result.batch_size == 3
        assert result.n_iters == 5
        assert result.mean_ms >= 0
        assert result.std_ms >= 0

    def test_setup_called_once(self):
        sim = DummySimulator()
        spec = build_spec("hea", 2, 1)
        self._benchmark(sim, spec)
        assert sim.setup_calls == [(spec, "probs")]

    def test_warmup_called_once(self):
        sim = DummySimulator()
        self._benchmark(sim, build_spec("hea", 2, 1), n_iters=3)
        assert sim.warmup_calls == 1

    def test_warmup_skipped(self):
        sim = DummySimulator()
        self._benchmark(sim, build_spec("hea", 2, 1), do_warmup=False)
        assert sim.warmup_calls == 0

    def test_run_called_n_iters_times(self):
        sim = DummySimulator()
        n_iters = 7
        self._benchmark(sim, build_spec("hea", 2, 1), mode="state", n_iters=n_iters)
        assert sim.run_calls == n_iters

    def test_raw_output_is_last_run(self):
        sim = DummySimulator()
        spec = build_spec("hea", 2, 1)
        inputs, weights = self._make_sweep(spec, n_iters=3, batch_size=1)
        result = sim.benchmark(
            spec=spec, mode="probs", all_inputs=inputs, all_weights=weights
        )
        # DummySimulator.run returns inputs * 2.0; the last iteration uses index 2
        expected = inputs[2] * 2.0
        np.testing.assert_allclose(result.raw_output, expected)

    def test_supports_defaults_to_true(self):
        sim = DummySimulator()
        spec = build_spec("hea", 2, 1)
        assert sim.supports(spec, "probs")
        assert sim.supports(spec, "grad")
