"""Tests for benchmark.config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from benchmark.config import load_config, BenchmarkConfig


class TestLoadConfigDefaults:
    """Loading with no arguments should produce sensible defaults."""

    def test_returns_benchmark_config(self):
        cfg = load_config()
        assert isinstance(cfg, BenchmarkConfig)

    def test_default_qubit_range(self):
        cfg = load_config()
        assert cfg.qubits.min == 2
        assert cfg.qubits.max == 16

    def test_default_modes(self):
        cfg = load_config()
        assert cfg.modes == ["probs", "expval", "state", "density"]

    def test_identifier_auto_generated(self):
        cfg = load_config()
        # Should be a 14-digit timestamp string
        assert cfg.output.identifier is not None
        assert len(cfg.output.identifier) == 14
        assert cfg.output.identifier.isdigit()


class TestLoadConfigOverrides:
    """CLI-style dot-list overrides should take precedence."""

    def test_override_seed(self):
        cfg = load_config(overrides=["seed=42"])
        assert cfg.seed == 42

    def test_override_qubits(self):
        cfg = load_config(overrides=["qubits.min=5", "qubits.max=8"])
        assert cfg.qubits.min == 5
        assert cfg.qubits.max == 8

    def test_override_identifier(self):
        cfg = load_config(overrides=["output.identifier=myrun"])
        assert cfg.output.identifier == "myrun"

    def test_override_modes(self):
        cfg = load_config(overrides=["modes=[probs,expval]"])
        assert cfg.modes == ["probs", "expval"]


class TestLoadConfigSimulators:
    """The `simulators` field should filter which backends are run."""

    def test_default_simulators(self):
        cfg = load_config()
        assert cfg.simulators == ["yaqsi", "pennylane", "qiskit", "qibo", "qulacs"]

    def test_override_simulators(self):
        cfg = load_config(overrides=["simulators=[yaqsi,pennylane]"])
        assert cfg.simulators == ["yaqsi", "pennylane"]

    def test_single_simulator(self):
        cfg = load_config(overrides=["simulators=[qiskit]"])
        assert cfg.simulators == ["qiskit"]

    def test_unknown_simulator_raises(self):
        with pytest.raises(ValueError, match="Unknown simulator"):
            load_config(overrides=["simulators=[yaqsi,fake_sim]"])

    def test_simulators_from_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(
            "simulators:\n"
            "  - yaqsi\n"
            "  - qibo\n"
        )
        cfg = load_config(config_path=str(yaml_file))
        assert cfg.simulators == ["yaqsi", "qibo"]


class TestLoadConfigYaml:
    """A custom YAML file should be loadable and mergeable."""

    def test_custom_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
                seed: 999
                qubits:
                  min: 4
                  max: 6
                execution:
                  n_iters: 5
            """)
        )
        cfg = load_config(config_path=str(yaml_file))
        assert cfg.seed == 999
        assert cfg.qubits.min == 4
        assert cfg.qubits.max == 6
        assert cfg.execution.n_iters == 5
        # Defaults still applied for fields not in the YAML
        assert cfg.warmup is True

    def test_yaml_plus_overrides(self, tmp_path: Path):
        yaml_file = tmp_path / "base.yaml"
        yaml_file.write_text("seed: 111\n")
        cfg = load_config(config_path=str(yaml_file), overrides=["seed=222"])
        # Override wins
        assert cfg.seed == 222

    def test_nonexistent_yaml_falls_back_to_defaults(self):
        cfg = load_config(config_path="/nonexistent/path.yaml")
        # Should still work with structured defaults
        assert isinstance(cfg, BenchmarkConfig)
        assert cfg.seed == 1000