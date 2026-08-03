"""Tests for benchmark.runner CSV helpers and recovery logic."""


from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import pytest

from benchmark.runner import (
    CSV_COLUMNS,
    GATE_COUNTERPART,
    _append_row,
    _csv_path,
    _ensure_csv,
    _infidelity,
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
            csv.writer(f).writerow(["2", "probs", "jaqsi", "1.0", "0.1", "1", "10"])
        # Call again — should NOT truncate
        _ensure_csv(p)
        with open(p) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 data row


# ---------------------------------------------------------------------------
# _append_row / _load_completed
# ---------------------------------------------------------------------------

class TestAppendAndLoadCompleted:
    def _make_result(self, sim: str = "jaqsi", mode: str = "probs", n_qubits: int = 2):
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

        r1 = self._make_result("jaqsi", "probs", 2)
        r2 = self._make_result("pennylane", "probs", 2)
        _append_row(p, r1)
        _append_row(p, r2)

        completed = _load_completed(p)
        assert (2, "probs", "jaqsi") in completed
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
        r1 = BenchmarkResult("jaqsi", "probs", 2, 1, 10, 1.0, 0.1, arr)
        r2 = BenchmarkResult("pennylane", "probs", 2, 1, 10, 2.0, 0.2, arr)
        # Should not raise
        _validate_results(r1, r2, precision=1e-8)

    def test_mismatched_results_raise(self):
        r1 = BenchmarkResult("jaqsi", "probs", 2, 1, 10, 1.0, 0.1, jnp.array([1.0]))
        r2 = BenchmarkResult("pennylane", "probs", 2, 1, 10, 2.0, 0.2, jnp.array([0.0]))
        with pytest.raises(RuntimeError, match="Results mismatch"):
            _validate_results(r1, r2, precision=1e-8)

    def test_expval_transposed_pennylane(self):
        """expval mode: PL is (n_obs, batch), Jaqsi is (batch, n_obs)."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])       # (1, 3)  batch=1, 3 obs
        pl_arr = jnp.array([[0.1], [0.2], [0.3]])    # (3, 1)  PL convention
        r1 = BenchmarkResult("jaqsi", "expval", 3, 1, 10, 1.0, 0.1, ys_arr)
        r2 = BenchmarkResult("pennylane", "expval", 3, 1, 10, 2.0, 0.2, pl_arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_expval_not_transposed_qiskit(self):
        """expval mode: Qiskit uses (batch, n_obs) like Jaqsi — no transpose."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])       # (1, 3)
        qk_arr = jnp.array([[0.1, 0.2, 0.3]])       # (1, 3)  same layout
        r1 = BenchmarkResult("jaqsi", "expval", 3, 1, 10, 1.0, 0.1, ys_arr)
        r2 = BenchmarkResult("qiskit", "expval", 3, 1, 10, 2.0, 0.2, qk_arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_qiskit_matching_results_pass(self):
        arr = jnp.array([[0.5, 0.3]])
        r1 = BenchmarkResult("jaqsi", "probs", 2, 1, 10, 1.0, 0.1, arr)
        r2 = BenchmarkResult("qiskit", "probs", 2, 1, 10, 3.0, 0.3, arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_qibo_matching_results_pass(self):
        arr = jnp.array([[0.5, 0.3]])
        r1 = BenchmarkResult("jaqsi", "probs", 2, 1, 10, 1.0, 0.1, arr)
        r2 = BenchmarkResult("qibo", "probs", 2, 1, 10, 2.5, 0.2, arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_expval_not_transposed_qibo(self):
        """expval mode: Qibo uses (batch, n_obs) like Jaqsi — no transpose."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])
        qb_arr = jnp.array([[0.1, 0.2, 0.3]])
        r1 = BenchmarkResult("jaqsi", "expval", 3, 1, 10, 1.0, 0.1, ys_arr)
        r2 = BenchmarkResult("qibo", "expval", 3, 1, 10, 2.0, 0.2, qb_arr)
        _validate_results(r1, r2, precision=1e-8)


# ---------------------------------------------------------------------------
# _infidelity
# ---------------------------------------------------------------------------

class TestInfidelity:
    """Pulse results are scored against the gate-level circuit they implement."""

    _PSI = jnp.array([[0.6, 0.8j, 0.0, 0.0]])

    @staticmethod
    def _density(psi):
        return jnp.einsum("bi,bj->bij", psi, jnp.conj(psi))

    def test_identical_state_is_zero(self):
        assert _infidelity(self._PSI, self._PSI, "state") == pytest.approx(0.0, abs=1e-15)

    def test_identical_density_is_zero(self):
        rho = self._density(self._PSI)
        assert _infidelity(rho, rho, "density") == pytest.approx(0.0, abs=1e-15)

    def test_identical_probs_is_zero(self):
        probs = jnp.abs(self._PSI) ** 2
        assert _infidelity(probs, probs, "probs") == pytest.approx(0.0, abs=1e-15)

    def test_orthogonal_states_are_one(self):
        orthogonal = jnp.array([[0.0, 0.0, 1.0, 0.0]])
        assert _infidelity(self._PSI, orthogonal, "state") == pytest.approx(1.0)

    def test_expval_has_no_fidelity(self):
        assert _infidelity(self._PSI, self._PSI, "expval") is None

    def test_global_phase_is_ignored(self):
        phased = self._PSI * jnp.exp(1j * 0.7)
        assert _infidelity(phased, self._PSI, "state") == pytest.approx(0.0, abs=1e-15)

    def test_norm_drift_does_not_go_negative(self):
        """The ODE solvers do not preserve the norm exactly."""
        drifted = self._PSI * (1.0 + 1.0e-7)
        assert _infidelity(drifted, self._PSI, "state") >= 0.0

    def test_matches_analytic_overlap(self):
        theta = 0.3
        rotated = jnp.array([[jnp.cos(theta), jnp.sin(theta)]], dtype=complex)
        zero = jnp.array([[1.0, 0.0]], dtype=complex)
        expected = float(jnp.sin(theta) ** 2)
        assert _infidelity(rotated, zero, "state") == pytest.approx(expected)
        # Pure states: the density-matrix fidelity reduces to the same overlap.
        assert _infidelity(
            self._density(rotated), self._density(zero), "density"
        ) == pytest.approx(expected)

    def test_reports_worst_case_over_batch(self):
        theta = 0.3
        rotated = jnp.array([[jnp.cos(theta), jnp.sin(theta)]], dtype=complex)
        zero = jnp.array([[1.0, 0.0]], dtype=complex)
        batch = jnp.concatenate([zero, rotated], axis=0)
        reference = jnp.concatenate([zero, zero], axis=0)
        assert _infidelity(batch, reference, "state") == pytest.approx(
            float(jnp.sin(theta) ** 2)
        )

    def test_every_pulse_simulator_has_a_counterpart(self):
        from benchmark.runner import SIMULATOR_REGISTRY, _level

        pulse = {name for name in SIMULATOR_REGISTRY if _level(name) == "pulse"}
        assert pulse == set(GATE_COUNTERPART)
        for gate_name in GATE_COUNTERPART.values():
            assert _level(gate_name) == "gate"
            assert gate_name in SIMULATOR_REGISTRY


class TestInfidelityColumn:
    """The infidelity reaches the CSV, and is blank for gate-level rows."""

    def test_written_for_pulse_rows(self, tmp_path: Path):
        path = tmp_path / "b.csv"
        _ensure_csv(path)
        result = BenchmarkResult(
            "jaqsi_pulse", "state", 2, 1, 10, 1.0, 0.1, jnp.array([1.0])
        )
        result.infidelity = 1.25e-14
        _append_row(path, result)

        with open(path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        assert float(row["infidelity"]) == pytest.approx(1.25e-14)

    def test_blank_for_gate_rows(self, tmp_path: Path):
        path = tmp_path / "b.csv"
        _ensure_csv(path)
        _append_row(
            path,
            BenchmarkResult("jaqsi", "state", 2, 1, 10, 1.0, 0.1, jnp.array([1.0])),
        )

        with open(path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        assert row["infidelity"] == ""