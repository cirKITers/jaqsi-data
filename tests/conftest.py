"""Shared fixtures for the benchmark test suite."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmark.runner import CSV_COLUMNS


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for benchmark output."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a minimal benchmark CSV with three simulators and return its path."""
    csv_path = tmp_path / "benchmarks-test123.csv"
    rows = [
        # (circuit, n_layers, n_qubits, mode, simulator, mean_ms, std_ms,
        #  batch_size, n_iters, threads)
        ("hea", 1, 2, "probs", "pennylane", "3.000000", "0.200000", 1, 10, 1),
        ("hea", 1, 2, "probs", "qibo", "3.500000", "0.250000", 1, 10, 1),
        ("hea", 1, 2, "probs", "qiskit", "4.000000", "0.300000", 1, 10, 1),
        ("hea", 1, 2, "probs", "jaqsi", "1.500000", "0.100000", 1, 10, 1),
        ("hea", 1, 2, "expval", "pennylane", "2.400000", "0.160000", 1, 10, 1),
        ("hea", 1, 2, "expval", "qibo", "2.800000", "0.180000", 1, 10, 1),
        ("hea", 1, 2, "expval", "qiskit", "3.600000", "0.240000", 1, 10, 1),
        ("hea", 1, 2, "expval", "jaqsi", "1.200000", "0.080000", 1, 10, 1),
        ("hea", 1, 3, "probs", "pennylane", "5.000000", "0.300000", 1, 10, 1),
        ("hea", 1, 3, "probs", "qibo", "5.500000", "0.350000", 1, 10, 1),
        ("hea", 1, 3, "probs", "qiskit", "6.500000", "0.400000", 1, 10, 1),
        ("hea", 1, 3, "probs", "jaqsi", "2.000000", "0.150000", 1, 10, 1),
        ("hea", 1, 3, "expval", "pennylane", "4.500000", "0.250000", 1, 10, 1),
        ("hea", 1, 3, "expval", "qibo", "5.000000", "0.300000", 1, 10, 1),
        ("hea", 1, 3, "expval", "qiskit", "5.400000", "0.350000", 1, 10, 1),
        ("hea", 1, 3, "expval", "jaqsi", "1.800000", "0.120000", 1, 10, 1),
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    return csv_path