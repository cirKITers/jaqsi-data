"""Benchmark runner with CSV-based recovery support."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import jax
import jax.numpy as jnp

from benchmark.config import BenchmarkConfig
from benchmark.simulators.base import BenchmarkResult, Mode, SimulatorBenchmark

logger = logging.getLogger(__name__)

# CSV column order
CSV_COLUMNS = [
    "n_qubits",
    "mode",
    "simulator",
    "mean_ms",
    "std_ms",
    "batch_size",
    "n_iters",
]


def _csv_path(cfg: BenchmarkConfig) -> Path:
    return Path(cfg.output.dir) / f"benchmarks-{cfg.output.identifier}.csv"


def _ensure_csv(path: Path) -> None:
    """Create the CSV file with a header if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        logger.info(f"Created new results file: {path}")


def _load_completed(path: Path) -> Set[Tuple[int, str, str]]:
    """Return the set of ``(n_qubits, mode, simulator)`` tuples already
    present in *path* so we can skip them on a resumed run."""
    completed: Set[Tuple[int, str, str]] = set()
    if not path.exists():
        return completed
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((int(row["n_qubits"]), row["mode"], row["simulator"]))
    return completed


def _append_row(path: Path, result: BenchmarkResult) -> None:
    """Append a single result row to the CSV, flushing immediately."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                result.n_qubits,
                result.mode,
                result.simulator,
                f"{result.mean_ms:.6f}",
                f"{result.std_ms:.6f}",
                result.batch_size,
                result.n_iters,
            ]
        )


def _validate_results(
    ref: BenchmarkResult,
    other: BenchmarkResult,
    precision: float,
) -> None:
    """Raise ``RuntimeError`` when *other* diverges from *ref*.

    The *ref* result is treated as the reference (typically jaqsi).
    PennyLane returns expval as ``(n_obs, batch)`` while jaqsi and the
    Qiskit adapter both use ``(batch, n_obs)`` layout.
    """
    ref_arr = jnp.asarray(ref.raw_output)
    oth_arr = jnp.asarray(other.raw_output)

    # PennyLane returns expval transposed relative to everyone else
    if ref.mode == "expval" and other.simulator == "pennylane":
        oth_arr = oth_arr.T

    if not jnp.allclose(ref_arr, oth_arr, atol=precision):
        raise RuntimeError(
            f"Results mismatch ({ref.simulator} vs {other.simulator}) "
            f"for {ref.n_qubits} qubits, mode={ref.mode}:\n"
            f"  {ref.simulator:>10}  shape={ref_arr.shape}\n"
            f"  {other.simulator:>10}  shape={oth_arr.shape}\n"
            f"  max|Δ| = {jnp.max(jnp.abs(ref_arr - oth_arr))}"
        )
    logger.info(f"  ✓ {ref.simulator} ≈ {other.simulator}")


def run_benchmarks(cfg: BenchmarkConfig) -> Path:
    """Execute the full benchmark suite described by *cfg*.

    Already-completed ``(n_qubits, mode, simulator)`` combinations found in
    the output CSV are skipped, enabling seamless recovery after a crash.

    Returns the path to the CSV results file.
    """
    jax.config.update("jax_enable_x64", True)

    csv_file = _csv_path(cfg)
    _ensure_csv(csv_file)
    completed = _load_completed(csv_file)
    if completed:
        logger.info(
            f"Resuming run {cfg.output.identifier}: "
            f"{len(completed)} result(s) already recorded."
        )

    rng = jax.random.PRNGKey(cfg.seed)
    qubit_sizes = list(range(cfg.qubits.min, cfg.qubits.max + 1))

    # Late imports to avoid hard dependency on optional backends at module level
    from benchmark.simulators.jaqsi_sim import JaqsiBenchmark
    from benchmark.simulators.pennylane_sim import PennylaneBenchmark
    from benchmark.simulators.qiskit_sim import QiskitBenchmark
    from benchmark.simulators.qibo_sim import QiboBenchmark
    from benchmark.simulators.qulacs_sim import QulacsBenchmark

    _all_simulators: Dict[str, SimulatorBenchmark] = {
        "jaqsi": JaqsiBenchmark(),
        "pennylane": PennylaneBenchmark(),
        "qiskit": QiskitBenchmark(),
        "qibo": QiboBenchmark(),
        "qulacs": QulacsBenchmark(),
    }

    simulators: List[SimulatorBenchmark] = [
        _all_simulators[name] for name in cfg.simulators
    ]

    for n_qubits in qubit_sizes:
        for mode in cfg.modes:
            mode: Mode  # type: ignore[no-redef]

            # Check if *all* simulators are already done for this combo
            all_done = all(
                (n_qubits, mode, sim.name) in completed for sim in simulators
            )
            if all_done:
                logger.info(
                    f"[skip] n_qubits={n_qubits}, mode={mode} — already complete"
                )
                continue

            # Generate random parameters (shared across simulators for fairness)
            rng, subkey = jax.random.split(rng)
            all_phis = jax.random.uniform(
                subkey,
                shape=(cfg.execution.n_iters + 1, cfg.execution.batch_size),
                minval=-jnp.pi,
                maxval=jnp.pi,
            )

            sim_results: Dict[str, BenchmarkResult] = {}

            for sim in simulators:
                key = (n_qubits, mode, sim.name)
                if key in completed:
                    logger.info(f"[skip] {sim.name} n_qubits={n_qubits}, mode={mode}")
                    continue

                logger.info(
                    f"[run]  {sim.name} n_qubits={n_qubits}, mode={mode} "
                    f"(iters={cfg.execution.n_iters}, batch={cfg.execution.batch_size})"
                )
                result = sim.benchmark(
                    n_qubits=n_qubits,
                    mode=mode,
                    all_phis=all_phis,
                    do_warmup=cfg.warmup,
                )
                logger.info(
                    f"  {result.mean_ms:.2f} ± {result.std_ms:.2f} ms"
                )

                _append_row(csv_file, result)
                completed.add(key)
                sim_results[sim.name] = result

            # Cross-validate all simulators against jaqsi (reference)
            if "jaqsi" in sim_results:
                for other_name, other_res in sim_results.items():
                    if other_name == "jaqsi":
                        continue
                    _validate_results(
                        sim_results["jaqsi"],
                        other_res,
                        cfg.precision,
                    )

    logger.info(f"All benchmarks complete. Results in {csv_file}")
    return csv_file