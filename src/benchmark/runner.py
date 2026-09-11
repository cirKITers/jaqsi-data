"""Benchmark runner with CSV-based recovery support."""

from __future__ import annotations

import csv
import importlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp

from benchmark.circuits import CircuitSpec, build_spec
from benchmark.config import BenchmarkConfig
from benchmark.simulators.base import BenchmarkResult, Mode, SimulatorBenchmark

logger = logging.getLogger(__name__)

# CSV column order
CSV_COLUMNS = [
    "circuit",
    "n_layers",
    "n_qubits",
    "mode",
    "simulator",
    "mean_ms",
    "std_ms",
    "batch_size",
    "n_iters",
    "threads",
    "infidelity",
]

# Cross-validation tolerance for the gradient mode.  Reverse-mode AD, the
# adjoint method and Qulacs' backprop accumulate their sums in different
# orders, so gradients agree to a looser bound than the forward results.
GRAD_PRECISION = 1.0e-6

# Maps a simulator name to the module and class implementing its adapter.
# Only the requested adapters are imported, so a missing optional backend does
# not break runs that do not use it.
SIMULATOR_REGISTRY: Dict[str, Tuple[str, str]] = {
    "jaqsi": ("jaqsi_sim", "JaqsiBenchmark"),
    "pennylane": ("pennylane_sim", "PennylaneBenchmark"),
    "pennylane_adjoint": ("pennylane_sim", "PennylaneAdjointBenchmark"),
    "pennylane_psr": ("pennylane_sim", "PennylanePsrBenchmark"),
    "qiskit": ("qiskit_sim", "QiskitBenchmark"),
    "qibo": ("qibo_sim", "QiboBenchmark"),
    "qulacs": ("qulacs_sim", "QulacsBenchmark"),
    "jaqsi_pulse": ("jaqsi_pulse_sim", "JaqsiPulseBenchmark"),
    "pennylane_pulse": ("pennylane_pulse_sim", "PennylanePulseBenchmark"),
    "qutip_pulse": ("qutip_pulse_sim", "QutipPulseBenchmark"),
    "dynamiqs_pulse": ("dynamiqs_pulse_sim", "DynamiqsPulseBenchmark"),
}

# Reference simulator each simulation level is cross-validated against.  Pulse
# results carry the ODE solver error and are therefore never compared to the
# exact gate-level results.
REFERENCE_BY_LEVEL: Dict[str, str] = {"gate": "jaqsi", "pulse": "jaqsi_pulse"}

# Gate-level simulation each pulse simulator is measured against.  QuTiP and
# dynamiqs have no gate-level adapter here, so they fall back to the gate
# reference; the gate-level simulators agree to an infidelity of order
# $10^{-15}$, which makes the choice immaterial.
GATE_COUNTERPART: Dict[str, str] = {
    "jaqsi_pulse": "jaqsi",
    "pennylane_pulse": "pennylane",
    "qutip_pulse": "jaqsi",
    "dynamiqs_pulse": "jaqsi",
}

# Modes whose output defines a state or a distribution, and hence a fidelity.
FIDELITY_MODES = frozenset({"state", "density", "probs", "noise"})

# Gate-level PennyLane adapters batch through PennyLane's own parameter
# broadcasting, which returns expval as ``(n_obs, batch)`` where every other
# adapter returns ``(batch, n_obs)``.
PENNYLANE_BROADCAST = frozenset({"pennylane", "pennylane_adjoint"})


def _level(simulator: str) -> str:
    """Return the simulation level (``gate`` or ``pulse``) of *simulator*."""
    return "pulse" if simulator.endswith("_pulse") else "gate"


def _sqrtm_psd(a: jnp.ndarray) -> jnp.ndarray:
    """Return the principal square root of the Hermitian PSD matrices *a*."""
    w, v = jnp.linalg.eigh(a)
    root = jnp.sqrt(jnp.clip(w, 0.0))[..., None, :]
    return (v * root) @ jnp.conj(jnp.swapaxes(v, -1, -2))


def _infidelity(pulse_output, gate_output, mode: str) -> Optional[float]:
    """Return $1 - F$ between a pulse result and its gate-level counterpart.

    The fidelity is normalised by the norms of both operands, so that the
    result measures the deviation in state rather than the norm drift the ODE
    solvers accumulate.  Returns the worst case over the batch, or ``None``
    for modes that define no state.  Normalisation bounds the fidelity by one,
    so the result is clipped at zero to absorb rounding at that bound.
    """
    pulse = jnp.asarray(pulse_output)
    gate = jnp.asarray(gate_output)

    if mode == "state":
        overlap = jnp.abs(jnp.sum(jnp.conj(gate) * pulse, axis=-1)) ** 2
        norms = jnp.sum(jnp.abs(gate) ** 2, axis=-1) * jnp.sum(
            jnp.abs(pulse) ** 2, axis=-1
        )
    elif mode == "density":
        # The schedule is unitary, so both operands are pure and the Uhlmann
        # fidelity reduces to $\\mathrm{tr}(\\rho\\sigma)$.
        overlap = jnp.real(jnp.einsum("...ij,...ji->...", gate, pulse))
        norms = jnp.real(
            jnp.trace(gate, axis1=-2, axis2=-1) * jnp.trace(pulse, axis1=-2, axis2=-1)
        )
    elif mode == "noise":
        # The channels make both operands mixed, so this takes the full
        # Uhlmann fidelity $\\lVert\\sqrt{\\rho}\\sqrt{\\sigma}\\rVert_1^2$, which
        # the ``density`` case above is the pure-state limit of.  It sums the
        # singular values of $\\sqrt{\\rho}\\sqrt{\\sigma}$ rather than the roots
        # of the eigenvalues of $\\sqrt{\\rho}\\sigma\\sqrt{\\rho}$: the noisy
        # states have eigenvalues down to $10^{-11}$, where those roots turn
        # rounding errors into an infidelity of $10^{-8}$ for identical inputs
        # at eight qubits.
        product = _sqrtm_psd(gate) @ _sqrtm_psd(pulse)
        overlap = jnp.sum(jnp.linalg.svd(product, compute_uv=False), axis=-1) ** 2
        norms = jnp.real(
            jnp.trace(gate, axis1=-2, axis2=-1) * jnp.trace(pulse, axis1=-2, axis2=-1)
        )
    elif mode == "probs":
        # Classical fidelity of the measurement distributions.
        gate_p = jnp.clip(jnp.real(gate), 0.0)
        pulse_p = jnp.clip(jnp.real(pulse), 0.0)
        overlap = jnp.sum(jnp.sqrt(gate_p * pulse_p), axis=-1) ** 2
        norms = jnp.sum(gate_p, axis=-1) * jnp.sum(pulse_p, axis=-1)
    else:
        return None

    return max(0.0, float(jnp.max(1.0 - overlap / norms)))


def _gate_output(
    name: str,
    spec: CircuitSpec,
    mode: Mode,
    inputs: jnp.ndarray,
    weights: jnp.ndarray,
    optimal_config: bool,
) -> jnp.ndarray:
    """Execute the gate-level simulator *name* once, outside the timing loop."""
    module_name, class_name = SIMULATOR_REGISTRY[name]
    module = importlib.import_module(f"benchmark.simulators.{module_name}")
    sim = getattr(module, class_name)()
    sim.setup(spec, mode, optimal_config=optimal_config)
    return sim.run(inputs, weights)


def _git_commit() -> str:
    """Return the short commit this benchmark ran from, ``unknown`` if unclear.

    A ``-dirty`` suffix marks modified *tracked* files; untracked ones are
    ignored, since a results file being written into the tree is the normal
    case and does not change the code that produced it.
    """
    root = Path(__file__).resolve().parent.parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not commit:
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def _csv_path(cfg: BenchmarkConfig) -> Path:
    """Return the results path, tagged with the commit the run came from.

    The commit is part of the file name rather than a column so that a results
    file identifies its own provenance once it is copied off the machine, and
    so that a run started after a code change lands in its own file instead of
    resuming into results from different code.
    """
    name = f"benchmarks-{cfg.output.identifier}-{_git_commit()}.csv"
    return Path(cfg.output.dir) / name


def _ensure_csv(path: Path) -> None:
    """Create the CSV file with a header if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        logger.info(f"Created new results file: {path}")


def _load_completed(path: Path) -> Set[Tuple[str, int, int, int, str, str, str]]:
    """Return the set of ``(circuit, n_layers, n_qubits, batch_size, threads,
    mode, simulator)`` tuples already present in *path* so we can skip them on
    a resumed run.

    ``batch_size`` and ``threads`` are part of the key so that a batch or
    thread sweep can be appended to one results file by re-running with the
    same ``output.identifier`` and a different setting.
    """
    completed: Set[Tuple[str, int, int, int, str, str, str]] = set()
    if not path.exists():
        return completed
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(
                (
                    row["circuit"],
                    int(row["n_layers"]),
                    int(row["n_qubits"]),
                    int(row["batch_size"]),
                    row.get("threads", ""),
                    row["mode"],
                    row["simulator"],
                )
            )
    return completed


def _append_row(path: Path, result: BenchmarkResult) -> None:
    """Append a single result row to the CSV, flushing immediately."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                result.circuit,
                result.n_layers,
                result.n_qubits,
                result.mode,
                result.simulator,
                f"{result.mean_ms:.6f}",
                f"{result.std_ms:.6f}",
                result.batch_size,
                result.n_iters,
                "" if result.threads is None else result.threads,
                "" if result.infidelity is None else f"{result.infidelity:.6e}",
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
    if ref.mode == "expval" and other.simulator in PENNYLANE_BROADCAST:
        oth_arr = oth_arr.T

    # rtol is disabled so that *precision* is the whole tolerance rather than
    # being widened by numpy's default relative term.
    if not jnp.allclose(ref_arr, oth_arr, atol=precision, rtol=0.0):
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

    Already-completed ``(circuit, n_layers, n_qubits, batch_size, threads,
    mode, simulator)`` combinations found in the output CSV are skipped,
    enabling seamless recovery after a crash.

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
    simulators: List[SimulatorBenchmark] = []
    for name in cfg.simulators:
        module_name, class_name = SIMULATOR_REGISTRY[name]
        module = importlib.import_module(f"benchmark.simulators.{module_name}")
        simulators.append(getattr(module, class_name)())

    for n_layers in cfg.circuit.layers:
        for n_qubits in qubit_sizes:
            for mode in cfg.modes:
                mode: Mode  # type: ignore[no-redef]

                # Only the noise mode carries depolarizing channels.
                spec = build_spec(
                    cfg.circuit.family,
                    n_qubits,
                    n_layers,
                    depolarizing=cfg.depolarizing if mode == "noise" else 0.0,
                )

                active: List[SimulatorBenchmark] = []
                for sim in simulators:
                    if sim.supports(spec, mode):
                        active.append(sim)
                    else:
                        logger.info(
                            f"[skip] {sim.name} does not support "
                            f"circuit={spec.family}, mode={mode}"
                        )

                # Check if *all* supported simulators are already done
                batch_size = cfg.execution.batch_size
                threads = str(cfg.threads)
                all_done = all(
                    (
                        spec.family,
                        n_layers,
                        n_qubits,
                        batch_size,
                        threads,
                        mode,
                        sim.name,
                    )
                    in completed
                    for sim in active
                )
                if all_done:
                    logger.info(
                        f"[skip] circuit={spec.family}, layers={n_layers}, "
                        f"n_qubits={n_qubits}, mode={mode} — already complete"
                    )
                    continue

                # Generate random parameters, shared across simulators for
                # fairness.  Inputs carry the batch axis; weights are shared
                # across the batch and redrawn every iteration.
                rng, input_key, weight_key = jax.random.split(rng, 3)
                all_inputs = jax.random.uniform(
                    input_key,
                    shape=(
                        cfg.execution.n_iters + 1,
                        cfg.execution.batch_size,
                        spec.n_inputs,
                    ),
                    minval=-jnp.pi,
                    maxval=jnp.pi,
                )
                all_weights = jax.random.uniform(
                    weight_key,
                    shape=(cfg.execution.n_iters + 1, spec.n_weights),
                    minval=-jnp.pi,
                    maxval=jnp.pi,
                )

                sim_results: Dict[str, BenchmarkResult] = {}

                # The recorded output is the one from the last timed iteration,
                # so the gate-level comparison has to use the same parameters.
                last_inputs = all_inputs[cfg.execution.n_iters - 1]
                last_weights = all_weights[cfg.execution.n_iters - 1]
                gate_outputs: Dict[str, jnp.ndarray] = {}

                for sim in active:
                    key = (
                        spec.family,
                        n_layers,
                        n_qubits,
                        batch_size,
                        threads,
                        mode,
                        sim.name,
                    )
                    if key in completed:
                        logger.info(
                            f"[skip] {sim.name} layers={n_layers}, "
                            f"n_qubits={n_qubits}, mode={mode}"
                        )
                        continue

                    logger.info(
                        f"[run]  {sim.name} circuit={spec.family}, "
                        f"layers={n_layers}, n_qubits={n_qubits}, mode={mode} "
                        f"(iters={cfg.execution.n_iters}, "
                        f"batch={cfg.execution.batch_size})"
                    )
                    result = sim.benchmark(
                        spec=spec,
                        mode=mode,
                        all_inputs=all_inputs,
                        all_weights=all_weights,
                        do_warmup=cfg.warmup,
                        optimal_config=cfg.optimal_config,
                    )
                    logger.info(
                        f"  {result.mean_ms:.2f} ± {result.std_ms:.2f} ms"
                    )

                    # Report how far the pulse simulation drifts from the
                    # gate-level circuit it implements, which the calibrated
                    # pulse parameters and the ODE solver accuracy both feed
                    # into.
                    if _level(sim.name) == "pulse" and mode in FIDELITY_MODES:
                        gate_name = GATE_COUNTERPART[sim.name]
                        if gate_name not in gate_outputs:
                            gate_outputs[gate_name] = _gate_output(
                                gate_name,
                                spec,
                                mode,
                                last_inputs,
                                last_weights,
                                cfg.optimal_config,
                            )
                        result.infidelity = _infidelity(
                            result.raw_output, gate_outputs[gate_name], mode
                        )
                        logger.info(
                            f"  infidelity vs {gate_name}: {result.infidelity:.3e}"
                        )

                    _append_row(csv_file, result)
                    completed.add(key)
                    sim_results[sim.name] = result

                # Cross-validate each simulation level against its jaqsi reference
                precision = GRAD_PRECISION if mode == "grad" else cfg.precision
                for level, ref_name in REFERENCE_BY_LEVEL.items():
                    if ref_name not in sim_results:
                        continue
                    for other_name, other_res in sim_results.items():
                        if other_name == ref_name or _level(other_name) != level:
                            continue
                        _validate_results(
                            sim_results[ref_name],
                            other_res,
                            precision,
                        )

    logger.info(f"All benchmarks complete. Results in {csv_file}")
    return csv_file