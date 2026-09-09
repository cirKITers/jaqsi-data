"""Tests for benchmark.visualize."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.visualize import (
    ModeResults,
    SimTimings,
    _compute_ratio_with_error,
    load_results,
    plot_ratio,
    plot_absolute,
    plot_infidelity,
)


def _write_pulse_csv(path: Path) -> Path:
    """Write a minimal pulse-level CSV carrying an infidelity column."""
    import csv
    from benchmark.runner import CSV_COLUMNS

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for q in (2, 3):
            # expval leaves the column empty, state fills it (including a zero).
            w.writerow(("crx_ring", 1, q, "expval", "jaqsi_pulse",
                        "1.0", "0.1", 1, 10, 1, ""))
            w.writerow(("crx_ring", 1, q, "expval", "qutip_pulse",
                        "4.0", "0.2", 1, 10, 1, ""))
            w.writerow(("crx_ring", 1, q, "state", "jaqsi_pulse",
                        "1.0", "0.1", 1, 10, 1,
                        "0.000000e+00" if q == 2 else "2.220446e-16"))
            w.writerow(("crx_ring", 1, q, "state", "qutip_pulse",
                        "4.0", "0.2", 1, 10, 1, "1.1e-14"))
    return path


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

    def test_all_simulators_present(self, sample_csv: Path):
        results = load_results(sample_csv)
        for mr in results.values():
            assert set(mr.simulators.keys()) == {"jaqsi", "pennylane", "qiskit", "qibo"}

    def test_values_parsed(self, sample_csv: Path):
        results = load_results(sample_csv)
        probs = results["probs"]
        assert probs.qubit_sizes == [2, 3]
        assert probs.simulators["jaqsi"].mean_ms[0] == pytest.approx(1.5)
        assert probs.simulators["pennylane"].mean_ms[0] == pytest.approx(3.0)
        assert probs.simulators["qiskit"].mean_ms[0] == pytest.approx(4.0)
        assert probs.simulators["qibo"].mean_ms[0] == pytest.approx(3.5)

    def test_infidelity_absent_column(self, sample_csv: Path):
        """A gate-level CSV without the column yields None everywhere."""
        results = load_results(sample_csv)
        st = results["probs"].simulators["jaqsi"]
        assert st.infidelity == [None] * len(st.qubit_sizes)

    def test_infidelity_parsed(self, tmp_path: Path):
        results = load_results(_write_pulse_csv(tmp_path / "pulse.csv"))
        assert results["expval"].simulators["jaqsi_pulse"].infidelity == [None, None]
        assert results["state"].simulators["qutip_pulse"].infidelity == [
            pytest.approx(1.1e-14),
            pytest.approx(1.1e-14),
        ]

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nope.csv")

    def test_sweep_is_narrowed_to_one_slice(self, tmp_path: Path):
        """Rows from other depths must not collide with the plotted slice."""
        import csv
        from benchmark.runner import CSV_COLUMNS

        p = tmp_path / "sweep.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            for layers, mean in ((1, "1.0"), (4, "9.0")):
                w.writerow(("hea", layers, 2, "probs", "jaqsi", mean, "0.1", 1, 10, 1))
                w.writerow(("hea", layers, 3, "probs", "jaqsi", mean, "0.1", 1, 10, 1))

        # Without a filter the first slice in the file wins.
        results = load_results(p)
        assert results["probs"].simulators["jaqsi"].mean_ms == [1.0, 1.0]

        # An explicit filter selects any other slice.
        results = load_results(p, n_layers=4)
        assert results["probs"].simulators["jaqsi"].mean_ms == [9.0, 9.0]

    def test_partial_csv_skips_incomplete(self, tmp_path: Path):
        """If a (n_qubits, mode) pair is missing a simulator that exists
        elsewhere in the file, that row should be skipped."""
        import csv
        from benchmark.runner import CSV_COLUMNS

        p = tmp_path / "partial.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            # probs@2: only jaqsi — incomplete
            w.writerow(("hea", 1, 2, "probs", "jaqsi", "1.0", "0.1", 1, 10, 1))
            # probs@3: both simulators — complete
            w.writerow(("hea", 1, 3, "probs", "jaqsi", "2.0", "0.2", 1, 10, 1))
            w.writerow(("hea", 1, 3, "probs", "pennylane", "4.0", "0.3", 1, 10, 1))
        results = load_results(p)
        # Only probs@3 should be included (probs@2 is incomplete)
        assert "probs" in results
        assert results["probs"].qubit_sizes == [3]


# ---------------------------------------------------------------------------
# _compute_ratio_with_error
# ---------------------------------------------------------------------------

class TestComputeRatio:
    def test_simple_ratio(self):
        ratios, errors = _compute_ratio_with_error(
            ref_mean=[2.0], ref_std=[0.0],
            other_mean=[4.0], other_std=[0.0],
        )
        assert ratios == [pytest.approx(2.0)]
        assert errors == [pytest.approx(0.0)]

    def test_error_propagation(self):
        ratios, errors = _compute_ratio_with_error(
            ref_mean=[10.0], ref_std=[1.0],
            other_mean=[20.0], other_std=[2.0],
        )
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
        out = tmp_path / "ratio.pgf"
        plot_ratio(results, output_path=out)
        # PNG is always written; PDF is intentionally dropped.
        png = out.with_suffix(".png")
        assert png.exists()
        assert png.stat().st_size > 0
        assert not out.with_suffix(".pdf").exists()

    def test_plot_absolute_saves_file(self, sample_csv: Path, tmp_path: Path):
        results = load_results(sample_csv)
        out = tmp_path / "absolute.pgf"
        plot_absolute(results, output_path=out)
        # PNG is always written; PDF is intentionally dropped.
        png = out.with_suffix(".png")
        assert png.exists()
        assert png.stat().st_size > 0
        assert not out.with_suffix(".pdf").exists()

    def test_plot_infidelity_saves_file(self, tmp_path: Path):
        results = load_results(_write_pulse_csv(tmp_path / "pulse.csv"))
        out = tmp_path / "infidelity.pgf"
        plot_infidelity(results, output_path=out)
        png = out.with_suffix(".png")
        assert png.exists()
        assert png.stat().st_size > 0

    def test_plot_infidelity_no_data(self, sample_csv: Path, tmp_path: Path):
        """A gate-level result file has no infidelity, so no figure is written."""
        results = load_results(sample_csv)
        out = tmp_path / "infidelity.pgf"
        plot_infidelity(results, output_path=out)
        assert not out.with_suffix(".png").exists()

    def test_plot_ratio_empty_results(self, tmp_path: Path):
        """Plotting with no data should not crash."""
        plot_ratio({}, output_path=tmp_path / "empty.pdf")

    def test_plot_absolute_empty_results(self, tmp_path: Path):
        """Empty results should not crash."""
        plot_absolute({}, output_path=tmp_path / "empty.pdf")