"""Exercise partial recovery without running a full simulator sweep."""

import csv
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from benchmark import runner
from benchmark.config import load_config
from benchmark.simulators.base import BenchmarkResult


@pytest.fixture
def adapters(monkeypatch):
    seen = {}

    class Reference:
        name = "jaqsi"

        def supports(self, spec, mode):
            return True

        def setup(self, spec, mode, **kwargs):
            self.mode = mode

        def run(self, inputs, weights):
            return inputs + weights[: inputs.shape[-1]]

        def benchmark(self, spec, mode, all_inputs, all_weights, **kwargs):
            seen[(self.name, spec.n_qubits, mode)] = (
                np.asarray(all_inputs),
                np.asarray(all_weights),
            )
            self.setup(spec, mode)
            return BenchmarkResult(
                simulator=self.name,
                mode=mode,
                n_qubits=spec.n_qubits,
                circuit=spec.family,
                n_layers=spec.n_layers,
                batch_size=all_inputs.shape[1],
                n_iters=all_inputs.shape[0] - 1,
                threads=1,
                mean_ms=1.0,
                std_ms=0.0,
                raw_output=self.run(all_inputs[-2], all_weights[-2]),
            )

    class Other(Reference):
        name = "qiskit"
        mismatch = False

        def run(self, inputs, weights):
            output = super().run(inputs, weights)
            return output + 1 if self.mismatch else output

    module = SimpleNamespace(Reference=Reference, Other=Other)
    monkeypatch.setattr(
        runner,
        "SIMULATOR_REGISTRY",
        {"jaqsi": ("fake", "Reference"), "qiskit": ("fake", "Other")},
    )
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: (
            module if name == "benchmark.simulators.fake" else original_import(name)
        ),
    )
    monkeypatch.setattr(runner, "record_provenance", lambda *args, **kwargs: None)
    return seen, Other


def config(tmp_path, **overrides):
    values = {
        "output.dir": str(tmp_path),
        "output.identifier": "resume",
        "simulators": "[jaqsi,qiskit]",
        "modes": "[expval]",
        "qubits.min": 2,
        "qubits.max": 2,
        "execution.n_iters": 2,
        "execution.batch_size": 2,
        "circuit.layers": "[1]",
    }
    values.update(overrides)
    return load_config(overrides=[f"{k}={v}" for k, v in values.items()])


def test_invalid_row_is_not_saved_and_is_revalidated_on_resume(tmp_path, adapters):
    seen, other = adapters
    cfg = config(tmp_path)
    other.mismatch = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Results mismatch"):
            runner.run_benchmarks(cfg)
        with runner._csv_path(cfg).open() as f:
            assert [r["simulator"] for r in csv.DictReader(f)] == ["jaqsi"]
    original = seen[("jaqsi", 2, "expval")]
    resumed = seen[("qiskit", 2, "expval")]
    for a, b in zip(original, resumed):
        np.testing.assert_array_equal(a, b)
    other.mismatch = False
    runner.run_benchmarks(cfg)
    assert len(runner._load_completed(runner._csv_path(cfg))) == 2


def test_case_inputs_survive_skips_and_mode_changes(tmp_path, adapters):
    seen, _ = adapters
    cfg = config(tmp_path, **{"qubits.max": 3})
    runner.run_benchmarks(cfg)
    original = seen[("jaqsi", 3, "expval")]
    # Resume into a wider sweep with the earlier width already completed.
    cfg.qubits.max = 4
    runner.run_benchmarks(cfg)
    extended = seen[("jaqsi", 4, "expval")]
    cfg = config(
        tmp_path / "separate",
        **{"qubits.min": 3, "qubits.max": 4, "modes": "[probs,expval]"},
    )
    runner.run_benchmarks(cfg)
    for n, expected in [(3, original), (4, extended)]:
        for a, b in zip(expected, seen[("jaqsi", n, "expval")]):
            np.testing.assert_array_equal(a, b)


def test_unselected_reference_still_validates(tmp_path, adapters):
    _, other = adapters
    other.mismatch = True
    cfg = config(tmp_path, simulators="[qiskit]")
    with pytest.raises(RuntimeError, match="Results mismatch"):
        runner.run_benchmarks(cfg)
    assert not runner._load_completed(runner._csv_path(cfg))


def test_validation_rejects_broadcastable_shape_mismatch():
    ref = BenchmarkResult(
        simulator="jaqsi",
        raw_output=jnp.ones((2, 4)),
        mode="probs",
        n_qubits=2,
        circuit="hea",
        n_layers=1,
        batch_size=2,
        n_iters=1,
        threads=1,
        mean_ms=1.0,
        std_ms=0.0,
    )
    other = replace(ref, simulator="qiskit", raw_output=jnp.ones((1, 4)))
    with pytest.raises(RuntimeError, match="shapes"):
        runner._validate_results(ref, other, 1e-8)
