"""Tests for benchmark.runner CSV helpers and recovery logic."""


from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import pytest

from benchmark.runner import (
    CSV_COLUMNS,
    _append_row,
    _csv_path,
    _ensure_csv,
    _load_completed,
    _validate_results,
)
from benchmark.simulators.base import BenchmarkResult
from benchmark.config import load_config


# ---------------------------------------------------------------------------
# _csv_path
# ---------------------------------------------------------------------------

class TestCsvPath:
    def test_path_format(self):
        cfg = load_config(overrides=["output.dir=out", "output.identifier=abc123"])
        assert _csv_path(cfg) == Path("out/benchmarks-abc123.csv")


# ---------------------------------------------------------------------------
# _ensure_csv
# ---------------------------------------------------------------------------

class TestEnsureCsv:
    def test_creates_file_with_header(self, tmp_path: Path):
        p = tmp_path / "sub" / "results.csv"
        _ensure_csv(p)
        assert p.exists()
        with open(p) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_COLUMNS

    def test_idempotent(self, tmp_path: Path):
        p = tmp_path / "results.csv"
        _ensure_csv(p)
        # Write a data row
        with open(p, "a", newline="") as f:
            csv.writer(f).writerow(["2", "probs", "yaqsi", "1.0", "0.1", "1", "10"])
        # Call again — should NOT truncate
        _ensure_csv(p)
        with open(p) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 data row


# ---------------------------------------------------------------------------
# _append_row / _load_completed
# ---------------------------------------------------------------------------

class TestAppendAndLoadCompleted:
    def _make_result(self, sim: str = "yaqsi", mode: str = "probs", n_qubits: int = 2):
        return BenchmarkResult(
            simulator=sim,
            mode=mode,
            n_qubits=n_qubits,
            batch_size=1,
            n_iters=10,
            mean_ms=1.5,
            std_ms=0.1,
            raw_output=jnp.array([0.0]),
        )

    def test_round_trip(self, tmp_path: Path):
        p = tmp_path / "results.csv"
        _ensure_csv(p)

        r1 = self._make_result("yaqsi", "probs", 2)
        r2 = self._make_result("pennylane", "probs", 2)
        _append_row(p, r1)
        _append_row(p, r2)

        completed = _load_completed(p)
        assert (2, "probs", "yaqsi") in completed
        assert (2, "probs", "pennylane") in completed
        assert len(completed) == 2

    def test_load_completed_empty_file(self, tmp_path: Path):
        p = tmp_path / "results.csv"
        _ensure_csv(p)
        assert _load_completed(p) == set()

    def test_load_completed_nonexistent(self, tmp_path: Path):
        p = tmp_path / "nope.csv"
        assert _load_completed(p) == set()


# ---------------------------------------------------------------------------
# _validate_results
# ---------------------------------------------------------------------------

class TestValidateResults:
    def test_matching_results_pass(self):
        arr = jnp.array([[0.5, 0.3]])
        r1 = BenchmarkResult("yaqsi", "probs", 2, 1, 10, 1.0, 0.1, arr)
        r2 = BenchmarkResult("pennylane", "probs", 2, 1, 10, 2.0, 0.2, arr)
        # Should not raise
        _validate_results(r1, r2, precision=1e-8)

    def test_mismatched_results_raise(self):
        r1 = BenchmarkResult("yaqsi", "probs", 2, 1, 10, 1.0, 0.1, jnp.array([1.0]))
        r2 = BenchmarkResult("pennylane", "probs", 2, 1, 10, 2.0, 0.2, jnp.array([0.0]))
        with pytest.raises(RuntimeError, match="Results mismatch"):
            _validate_results(r1, r2, precision=1e-8)

    def test_expval_transposed_pennylane(self):
        """expval mode: PL is (n_obs, batch), Yaqsi is (batch, n_obs)."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])       # (1, 3)  batch=1, 3 obs
        pl_arr = jnp.array([[0.1], [0.2], [0.3]])    # (3, 1)  PL convention
        r1 = BenchmarkResult("yaqsi", "expval", 3, 1, 10, 1.0, 0.1, ys_arr)
        r2 = BenchmarkResult("pennylane", "expval", 3, 1, 10, 2.0, 0.2, pl_arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_expval_not_transposed_qiskit(self):
        """expval mode: Qiskit uses (batch, n_obs) like Yaqsi — no transpose."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])       # (1, 3)
        qk_arr = jnp.array([[0.1, 0.2, 0.3]])       # (1, 3)  same layout
        r1 = BenchmarkResult("yaqsi", "expval", 3, 1, 10, 1.0, 0.1, ys_arr)
        r2 = BenchmarkResult("qiskit", "expval", 3, 1, 10, 2.0, 0.2, qk_arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_qiskit_matching_results_pass(self):
        arr = jnp.array([[0.5, 0.3]])
        r1 = BenchmarkResult("yaqsi", "probs", 2, 1, 10, 1.0, 0.1, arr)
        r2 = BenchmarkResult("qiskit", "probs", 2, 1, 10, 3.0, 0.3, arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_qibo_matching_results_pass(self):
        arr = jnp.array([[0.5, 0.3]])
        r1 = BenchmarkResult("yaqsi", "probs", 2, 1, 10, 1.0, 0.1, arr)
        r2 = BenchmarkResult("qibo", "probs", 2, 1, 10, 2.5, 0.2, arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_expval_not_transposed_qibo(self):
        """expval mode: Qibo uses (batch, n_obs) like Yaqsi — no transpose."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])
        qb_arr = jnp.array([[0.1, 0.2, 0.3]])
        r1 = BenchmarkResult("yaqsi", "expval", 3, 1, 10, 1.0, 0.1, ys_arr)
        r2 = BenchmarkResult("qibo", "expval", 3, 1, 10, 2.0, 0.2, qb_arr)
        _validate_results(r1, r2, precision=1e-8)