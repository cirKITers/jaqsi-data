"""Tests for the CLI entry point (benchmark.__main__)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from benchmark.__main__ import main


@pytest.fixture(autouse=True)
def _keep_affinity(monkeypatch):
    """main() pins the whole test process, which caps numba for later tests."""
    monkeypatch.setattr("benchmark.threads.pin_threads", lambda threads: None)


class TestCLIVisualize:
    """--visualize-only should load CSV and generate plots without running benchmarks."""

    def test_visualize_only(self, sample_csv: Path, tmp_path: Path):
        with patch("benchmark.visualize.plot_ratio") as mock_ratio, \
             patch("benchmark.visualize.plot_absolute") as mock_abs, \
             patch("benchmark.visualize.print_summary") as mock_summary:
            main(["--visualize-only", str(sample_csv)])

        mock_ratio.assert_called_once()
        mock_abs.assert_called_once()
        mock_summary.assert_called_once()

    def test_visualize_only_missing_csv(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            main(["--visualize-only", str(tmp_path / "nope.csv")])


class TestCLIConfig:
    """Config overrides should propagate correctly through the CLI."""

    def test_overrides_reach_runner(self, tmp_path: Path):
        with patch("benchmark.runner.run_benchmarks") as mock_run, \
             patch("benchmark.visualize.load_results") as mock_load, \
             patch("benchmark.visualize.plot_ratio"), \
             patch("benchmark.visualize.plot_absolute"), \
             patch("benchmark.visualize.print_summary"):
            # Make run_benchmarks return a fake CSV path
            fake_csv = tmp_path / "fake.csv"
            fake_csv.touch()
            mock_run.return_value = fake_csv
            mock_load.return_value = {}

            main([
                f"output.dir={tmp_path}",
                "qubits.min=3",
                "qubits.max=4",
                "execution.n_iters=2",
            ])

        mock_run.assert_called_once()
        cfg = mock_run.call_args[0][0]
        assert cfg.qubits.min == 3
        assert cfg.qubits.max == 4
        assert cfg.execution.n_iters == 2

    def test_no_plot_flag(self, tmp_path: Path):
        with patch("benchmark.runner.run_benchmarks") as mock_run, \
             patch("benchmark.visualize.plot_ratio") as mock_ratio, \
             patch("benchmark.visualize.plot_absolute") as mock_abs:
            fake_csv = tmp_path / "fake.csv"
            fake_csv.touch()
            mock_run.return_value = fake_csv

            main(["--no-plot", f"output.dir={tmp_path}"])

        mock_ratio.assert_not_called()
        mock_abs.assert_not_called()