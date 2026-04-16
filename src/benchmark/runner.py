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
    res_ys: BenchmarkResult,
    res_pl: BenchmarkResult,
    precision: float,
) -> None:
    """Raise ``RuntimeError`` when yaqsi and pennylane outputs diverge."""
    ys_arr = jnp.asarray(res_ys.raw_output)
    pl_arr = jnp.asarray(res_pl.raw_output)

    # PennyLane returns expval as (n_obs, batch); Yaqsi returns (batch, n_obs)
    if res_ys.mode == "expval":
        pl_arr = pl_arr.T

    if not jnp.allclose(ys_arr, pl_arr, atol=precision):
        raise RuntimeError(
            f"Results mismatch for {res_ys.n_qubits} qubits, mode={res_ys.mode}:\n"
            f"  Yaqsi  shape={ys_arr.shape}\n"
            f"  PL     shape={pl_arr.shape}\n"
            f"  max|Δ| = {jnp.max(jnp.abs(ys_arr - pl_arr))}"
        )
    logger.info("  ✓ Results match")


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
    from benchmark.simulators.yaqsi_sim import YaqsiBenchmark
    from benchmark.simulators.pennylane_sim import PennylaneBenchmark

    simulators: List[SimulatorBenchmark] = [
        YaqsiBenchmark(),
        PennylaneBenchmark(),
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

            # Cross-validate if both were run in this session
            if "yaqsi" in sim_results and "pennylane" in sim_results:
                _validate_results(
                    sim_results["yaqsi"],
                    sim_results["pennylane"],
                    cfg.precision,
                )

    logger.info(f"All benchmarks complete. Results in {csv_file}")
    return csv_file