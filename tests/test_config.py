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

    def test_default_qubit_range_is_ascending(self):
        cfg = load_config()
        assert cfg.qubits.min >= 1
        assert cfg.qubits.max >= cfg.qubits.min

    def test_default_modes_are_valid(self):
        """The shipped modes are tuned between runs, so check the invariant.

        Pinning the exact list makes this fail every time a mode is commented
        out of the config, which says nothing about correctness.
        """
        from benchmark.simulators.base import Mode
        import typing

        cfg = load_config()
        valid = set(typing.get_args(Mode))
        assert cfg.modes
        assert set(cfg.modes) <= valid
        assert "grad" in cfg.modes

    def test_default_optimal_config(self):
        cfg = load_config()
        assert cfg.optimal_config is True

    def test_default_circuit_is_a_valid_depth_sweep(self):
        from benchmark.circuits import FAMILIES

        cfg = load_config()
        assert cfg.circuit.family in FAMILIES
        assert cfg.circuit.layers
        assert all(n >= 1 for n in cfg.circuit.layers)

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

    def test_override_optimal_config(self):
        cfg = load_config(overrides=["optimal_config=true"])
        assert cfg.optimal_config is True


class TestLoadConfigSimulators:
    """The `simulators` field should filter which backends are run."""

    def test_default_simulators_are_known(self):
        """The shipped list is tuned between runs, so check the invariant.

        jaqsi has to be present because every other simulator is
        cross-validated against it.  pennylane_psr is deliberately absent: the
        parameter-shift rule is far too slow to run at every size.
        """
        from benchmark.config import ALL_SIMULATORS

        cfg = load_config()
        assert set(cfg.simulators) <= set(ALL_SIMULATORS)
        assert "jaqsi" in cfg.simulators
        assert "pennylane_psr" not in cfg.simulators

    def test_parameter_shift_is_available(self):
        cfg = load_config(overrides=["simulators=[jaqsi,pennylane_psr]"])
        assert cfg.simulators == ["jaqsi", "pennylane_psr"]

    def test_override_simulators(self):
        cfg = load_config(overrides=["simulators=[jaqsi,pennylane]"])
        assert cfg.simulators == ["jaqsi", "pennylane"]

    def test_single_simulator(self):
        cfg = load_config(overrides=["simulators=[qiskit]"])
        assert cfg.simulators == ["qiskit"]

    def test_unknown_simulator_raises(self):
        with pytest.raises(ValueError, match="Unknown simulator"):
            load_config(overrides=["simulators=[jaqsi,fake_sim]"])

    def test_simulators_from_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(
            "simulators:\n"
            "  - jaqsi\n"
            "  - qibo\n"
        )
        cfg = load_config(config_path=str(yaml_file))
        assert cfg.simulators == ["jaqsi", "qibo"]


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


class TestCircuitConfig:
    """The circuit family and depth sweep are validated at load time."""

    def test_override_family_and_layers(self):
        cfg = load_config(
            overrides=["circuit.family=crx_ring", "circuit.layers=[1,3]"]
        )
        assert cfg.circuit.family == "crx_ring"
        assert cfg.circuit.layers == [1, 3]

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown circuit family"):
            load_config(overrides=["circuit.family=nope"])

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError, match="circuit.layers"):
            load_config(overrides=["circuit.layers=[]"])

    def test_non_positive_layers_raises(self):
        with pytest.raises(ValueError, match="circuit.layers"):
            load_config(overrides=["circuit.layers=[0]"])


class TestThreadsConfig:
    """The thread count is validated at load time and pins the whole process."""

    def test_default_thread_count_is_positive(self):
        """The shipped count is tuned between runs, so check the invariant."""
        cfg = load_config()
        assert isinstance(cfg.threads, int)
        assert cfg.threads >= 1

    def test_override(self):
        cfg = load_config(overrides=["threads=8"])
        assert cfg.threads == 8

    @pytest.mark.parametrize("threads", [0, -4])
    def test_non_positive_raises(self, threads):
        with pytest.raises(ValueError, match="threads must be at least 1"):
            load_config(overrides=[f"threads={threads}"])
