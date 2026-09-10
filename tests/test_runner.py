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
    run_benchmarks,
)
from benchmark.simulators.base import BenchmarkResult
from benchmark.config import load_config


def _result(
    simulator: str,
    mode: str,
    n_qubits: int,
    raw_output,
    *,
    circuit: str = "hea",
    n_layers: int = 1,
    batch_size: int = 1,
    n_iters: int = 10,
    threads: int = 1,
    mean_ms: float = 1.0,
    std_ms: float = 0.1,
) -> BenchmarkResult:
    """Build a BenchmarkResult, defaulting the fields a test does not care about."""
    return BenchmarkResult(
        simulator=simulator,
        mode=mode,
        n_qubits=n_qubits,
        circuit=circuit,
        n_layers=n_layers,
        batch_size=batch_size,
        n_iters=n_iters,
        threads=threads,
        mean_ms=mean_ms,
        std_ms=std_ms,
        raw_output=raw_output,
    )


# ---------------------------------------------------------------------------
# _csv_path
# ---------------------------------------------------------------------------

class TestCsvPath:
    def test_path_carries_identifier_and_commit(self):
        """The commit tags the file so results identify the code that made them."""
        from benchmark.runner import _git_commit

        cfg = load_config(overrides=["output.dir=out", "output.identifier=abc123"])
        path = _csv_path(cfg)
        assert path.parent == Path("out")
        assert path.name.startswith("benchmarks-abc123-")
        assert path.name.endswith(".csv")
        assert _git_commit() in path.name

    def test_commit_is_a_short_sha_or_unknown(self):
        from benchmark.runner import _git_commit

        commit = _git_commit()
        base = commit[: -len("-dirty")] if commit.endswith("-dirty") else commit
        assert commit == "unknown" or len(base) >= 7


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
            csv.writer(f).writerow(
                ["hea", "1", "2", "probs", "jaqsi", "1.0", "0.1", "1", "10", "1"]
            )
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
        return _result(sim, mode, n_qubits, jnp.array([0.0]), mean_ms=1.5)

    def test_round_trip(self, tmp_path: Path):
        p = tmp_path / "results.csv"
        _ensure_csv(p)

        r1 = self._make_result("jaqsi", "probs", 2)
        r2 = self._make_result("pennylane", "probs", 2)
        _append_row(p, r1)
        _append_row(p, r2)

        completed = _load_completed(p)
        assert ("hea", 1, 2, 1, "1", "probs", "jaqsi") in completed
        assert ("hea", 1, 2, 1, "1", "probs", "pennylane") in completed
        assert len(completed) == 2

    def test_sweeps_get_separate_entries(self, tmp_path: Path):
        """Depth, family, batch and thread sweeps must not collapse onto one key."""
        p = tmp_path / "results.csv"
        _ensure_csv(p)

        _append_row(p, _result("jaqsi", "probs", 2, jnp.array([0.0]), n_layers=1))
        _append_row(p, _result("jaqsi", "probs", 2, jnp.array([0.0]), n_layers=4))
        _append_row(
            p, _result("jaqsi", "probs", 2, jnp.array([0.0]), circuit="crx_ring")
        )
        _append_row(p, _result("jaqsi", "probs", 2, jnp.array([0.0]), batch_size=64))
        _append_row(p, _result("jaqsi", "probs", 2, jnp.array([0.0]), threads=16))

        completed = _load_completed(p)
        assert len(completed) == 5
        assert ("hea", 4, 2, 1, "1", "probs", "jaqsi") in completed
        assert ("crx_ring", 1, 2, 1, "1", "probs", "jaqsi") in completed
        assert ("hea", 1, 2, 64, "1", "probs", "jaqsi") in completed
        assert ("hea", 1, 2, 1, "16", "probs", "jaqsi") in completed

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
        r1 = _result("jaqsi", "probs", 2, arr)
        r2 = _result("pennylane", "probs", 2, arr)
        # Should not raise
        _validate_results(r1, r2, precision=1e-8)

    def test_mismatched_results_raise(self):
        r1 = _result("jaqsi", "probs", 2, jnp.array([1.0]))
        r2 = _result("pennylane", "probs", 2, jnp.array([0.0]))
        with pytest.raises(RuntimeError, match="Results mismatch"):
            _validate_results(r1, r2, precision=1e-8)

    def test_expval_transposed_pennylane(self):
        """expval mode: PL is (n_obs, batch), Jaqsi is (batch, n_obs)."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])       # (1, 3)  batch=1, 3 obs
        pl_arr = jnp.array([[0.1], [0.2], [0.3]])    # (3, 1)  PL convention
        r1 = _result("jaqsi", "expval", 3, ys_arr)
        r2 = _result("pennylane", "expval", 3, pl_arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_expval_transposed_pennylane_adjoint(self):
        """The lightning adapter broadcasts the same way as default.qubit."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])
        pl_arr = jnp.array([[0.1], [0.2], [0.3]])
        r1 = _result("jaqsi", "expval", 3, ys_arr)
        r2 = _result("pennylane_adjoint", "expval", 3, pl_arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_expval_not_transposed_pulse_pennylane(self):
        """The pulse adapter stacks per sample, so it needs no transpose."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])
        pl_arr = jnp.array([[0.1, 0.2, 0.3]])
        r1 = _result("jaqsi_pulse", "expval", 3, ys_arr, circuit="crx_ring")
        r2 = _result("pennylane_pulse", "expval", 3, pl_arr, circuit="crx_ring")
        _validate_results(r1, r2, precision=1e-8)

    def test_expval_not_transposed_qiskit(self):
        """expval mode: Qiskit uses (batch, n_obs) like Jaqsi — no transpose."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])       # (1, 3)
        qk_arr = jnp.array([[0.1, 0.2, 0.3]])       # (1, 3)  same layout
        r1 = _result("jaqsi", "expval", 3, ys_arr)
        r2 = _result("qiskit", "expval", 3, qk_arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_qiskit_matching_results_pass(self):
        arr = jnp.array([[0.5, 0.3]])
        r1 = _result("jaqsi", "probs", 2, arr)
        r2 = _result("qiskit", "probs", 2, arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_qibo_matching_results_pass(self):
        arr = jnp.array([[0.5, 0.3]])
        r1 = _result("jaqsi", "probs", 2, arr)
        r2 = _result("qibo", "probs", 2, arr)
        _validate_results(r1, r2, precision=1e-8)

    def test_expval_not_transposed_qibo(self):
        """expval mode: Qibo uses (batch, n_obs) like Jaqsi — no transpose."""
        ys_arr = jnp.array([[0.1, 0.2, 0.3]])
        qb_arr = jnp.array([[0.1, 0.2, 0.3]])
        r1 = _result("jaqsi", "expval", 3, ys_arr)
        r2 = _result("qibo", "expval", 3, qb_arr)
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
        result = _result(
            "jaqsi_pulse", "state", 2, jnp.array([1.0]), circuit="crx_ring"
        )
        result.infidelity = 1.25e-14
        _append_row(path, result)

        with open(path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        assert float(row["infidelity"]) == pytest.approx(1.25e-14)

    def test_blank_for_gate_rows(self, tmp_path: Path):
        path = tmp_path / "b.csv"
        _ensure_csv(path)
        _append_row(path, _result("jaqsi", "state", 2, jnp.array([1.0])))

        with open(path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        assert row["infidelity"] == ""


class TestNoiseMode:
    """Only the noise mode is handed a circuit with depolarizing channels."""

    def test_channels_reach_the_noise_mode_only(self, tmp_path: Path, monkeypatch):
        from benchmark.simulators.jaqsi_sim import JaqsiBenchmark

        seen = {}
        setup = JaqsiBenchmark.setup

        def recording_setup(self, spec, mode, **kwargs):
            seen[mode] = spec.depolarizing
            return setup(self, spec, mode, **kwargs)

        monkeypatch.setattr(JaqsiBenchmark, "setup", recording_setup)
        cfg = load_config(
            overrides=[
                f"output.dir={tmp_path}",
                "simulators=[jaqsi]",
                "modes=[expval,noise]",
                "qubits.min=2",
                "qubits.max=2",
                "circuit.layers=[1]",
                "execution.n_iters=1",
                "execution.batch_size=1",
            ]
        )
        run_benchmarks(cfg)
        assert seen == {"expval": 0.0, "noise": cfg.depolarizing}