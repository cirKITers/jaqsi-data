"""Tests for benchmark.visualize."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.visualize import (
    ModeResults,
    _compute_ratio_with_error,
    load_results,
    plot_ratio,
    plot_absolute,
)


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------

class TestLoadResults:
    def test_loads_correct_modes(self, sample_csv: Path):
        results = load_results(sample_csv)
        assert set(results.keys()) == {"probs", "expval"}

    def test_qubit_sizes_ascending(self, sample_csv: Path):
        results = load_results(sample_csv)
        for mr in results.values():
            assert mr.qubit_sizes == sorted(mr.qubit_sizes)

    def test_values_parsed(self, sample_csv: Path):
        results = load_results(sample_csv)
        probs = results["probs"]
        assert probs.qubit_sizes == [2, 3]
        assert probs.ys_mean_ms[0] == pytest.approx(1.5)
        assert probs.pl_mean_ms[0] == pytest.approx(3.0)

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nope.csv")

    def test_partial_csv_skips_incomplete(self, tmp_path: Path):
        """If a (n_qubits, mode) pair has only one simulator, skip it."""
        import csv
        from benchmark.runner import CSV_COLUMNS

        p = tmp_path / "partial.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            # Only yaqsi for probs@2 — no pennylane counterpart
            w.writerow((2, "probs", "yaqsi", "1.0", "0.1", 1, 10))
        results = load_results(p)
        # No complete pair → empty
        assert len(results) == 0 or all(
            len(mr.qubit_sizes) == 0 for mr in results.values()
        )


# ---------------------------------------------------------------------------
# _compute_ratio_with_error
# ---------------------------------------------------------------------------

class TestComputeRatio:
    def test_simple_ratio(self):
        mr = ModeResults(
            qubit_sizes=[2],
            ys_mean_ms=[2.0],
            ys_std_ms=[0.0],
            pl_mean_ms=[4.0],
            pl_std_ms=[0.0],
        )
        ratios, errors = _compute_ratio_with_error(mr)
        assert ratios == [pytest.approx(2.0)]
        assert errors == [pytest.approx(0.0)]

    def test_error_propagation(self):
        mr = ModeResults(
            qubit_sizes=[2],
            ys_mean_ms=[10.0],
            ys_std_ms=[1.0],
            pl_mean_ms=[20.0],
            pl_std_ms=[2.0],
        )
        ratios, errors = _compute_ratio_with_error(mr)
        r = 20.0 / 10.0
        expected_err = r * ((2.0 / 20.0) ** 2 + (1.0 / 10.0) ** 2) ** 0.5
        assert ratios[0] == pytest.approx(r)
        assert errors[0] == pytest.approx(expected_err)


# ---------------------------------------------------------------------------
# Plotting (smoke tests — just ensure no exceptions)
# ---------------------------------------------------------------------------

class TestPlotting:
    def test_plot_ratio_saves_file(self, sample_csv: Path, tmp_path: Path):
        results = load_results(sample_csv)
        out = tmp_path / "ratio.png"
        plot_ratio(results, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_absolute_saves_file(self, sample_csv: Path, tmp_path: Path):
        results = load_results(sample_csv)
        out = tmp_path / "absolute.png"
        plot_absolute(results, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_ratio_empty_results(self, tmp_path: Path):
        """Plotting with no data should not crash."""
        out = tmp_path / "empty.png"
        plot_ratio({}, output_path=out)
        # File is created but may be a blank figure
        assert out.exists()

    def test_plot_absolute_empty_results(self, tmp_path: Path):
        """Empty results should not crash."""
        plot_absolute({}, output_path=tmp_path / "empty.png")