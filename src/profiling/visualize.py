"""Visualization for JAQSI profiling results.

Generates publication-quality plots from the profiling summary data,
suitable for inclusion in papers or technical reports.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Styling – publication-quality defaults
# ------------------------------------------------------------------

# Two-column text width and single-column width (inches) of the IEEE
# conference paper, used to size figures for natural-scale inclusion.
TEXTWIDTH_IN = 7.16
COLWIDTH_IN = 3.45

# Paper-grade rcParams: serif fonts, subtle grid, no top/right spines, and a
# PGF backend configured for the pdflatex build of the paper.
PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.5,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
}

# Per-mode colours – Paul Tol "bright" qualitative scheme (colourblind safe)
MODE_COLORS: Dict[str, str] = {
    "probs": "#4477AA",    # blue
    "expval": "#EE6677",   # red
    "state": "#228833",    # green
    "density": "#AA3377",  # purple
}

MODE_MARKERS: Dict[str, str] = {
    "probs": "o",
    "expval": "s",
    "state": "^",
    "density": "D",
}


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    """Write the figure as PGF (for the paper) and PNG (for quick inspection).

    PDF output is intentionally dropped. PGF is attempted after the PNG so a
    missing LaTeX toolchain (e.g. on a headless compute node) degrades to a
    PNG-only result with a warning instead of raising.
    """
    base = Path(output_path).with_suffix("")
    png_path = base.with_suffix(".png")
    pgf_path = base.with_suffix(".pgf")
    fig.savefig(str(png_path))
    try:
        fig.savefig(str(pgf_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "PGF export failed for %s (LaTeX required): %s", pgf_path, exc
        )
        logger.info("Figure saved to %s", png_path)
    else:
        logger.info("Figure saved to %s and %s", pgf_path, png_path)


def _set_integer_xticks(ax: plt.Axes, qubit_sizes: List[int]) -> None:
    """Force x-axis to show only integer tick values."""
    ax.set_xticks(qubit_sizes)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(qubit_sizes))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FixedFormatter([str(q) for q in qubit_sizes])
    )


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class ProfilingTimings:
    """Timing and memory data for a single measurement mode across qubit counts."""

    qubit_sizes: List[int] = field(default_factory=list)
    wall_time_s: List[float] = field(default_factory=list)
    avg_time_s: List[float] = field(default_factory=list)
    profile_runs: List[int] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    trace_dirs: List[str] = field(default_factory=list)
    # Memory metrics (bytes)
    jax_peak_mem_bytes: List[int] = field(default_factory=list)
    rss_delta_bytes: List[int] = field(default_factory=list)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_profiling_results(results: list[dict]) -> Dict[str, ProfilingTimings]:
    """Organise a list of profiling result dicts by mode.

    Parameters
    ----------
    results : list[dict]
        The list returned by ``JaqsiProfiler.run_all()``.

    Returns
    -------
    dict
        Mapping from mode name → :class:`ProfilingTimings`.
    """
    by_mode: Dict[str, ProfilingTimings] = {}
    for r in results:
        mode = r["mode"]
        if mode not in by_mode:
            by_mode[mode] = ProfilingTimings()
        pt = by_mode[mode]
        pt.qubit_sizes.append(r["n_qubits"])
        pt.wall_time_s.append(r["wall_time_s"])
        pt.avg_time_s.append(r["avg_time_per_run_s"])
        pt.profile_runs.append(r["profile_runs"])
        pt.batch_sizes.append(r["batch_size"])
        pt.trace_dirs.append(r["trace_dir"])
        # Memory metrics (gracefully handle missing keys for old CSVs)
        pt.jax_peak_mem_bytes.append(int(r.get("jax_peak_mem_bytes", 0)))
        rss_before = int(r.get("rss_before_bytes", 0))
        rss_after = int(r.get("rss_after_bytes", 0))
        pt.rss_delta_bytes.append(rss_after - rss_before)
    return by_mode


def load_profiling_csv(csv_path: str | Path) -> Dict[str, ProfilingTimings]:
    """Load profiling results from a CSV file.

    The CSV must have columns: ``mode``, ``n_qubits``, ``batch_size``,
    ``profile_runs``, ``wall_time_s``, ``avg_time_per_run_s``, ``trace_dir``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "mode": row["mode"],
                "n_qubits": int(row["n_qubits"]),
                "batch_size": int(row["batch_size"]),
                "profile_runs": int(row["profile_runs"]),
                "wall_time_s": float(row["wall_time_s"]),
                "avg_time_per_run_s": float(row["avg_time_per_run_s"]),
                "trace_dir": row.get("trace_dir", ""),
                # Memory columns (may be absent in older CSVs)
                "jax_peak_mem_bytes": int(row.get("jax_peak_mem_bytes", 0)),
                "rss_before_bytes": int(row.get("rss_before_bytes", 0)),
                "rss_after_bytes": int(row.get("rss_after_bytes", 0)),
            })
    return load_profiling_results(rows)


# ------------------------------------------------------------------
# CSV export
# ------------------------------------------------------------------

PROFILING_CSV_COLUMNS = [
    "circuit",
    "n_layers",
    "mode",
    "n_qubits",
    "batch_size",
    "profile_runs",
    "wall_time_s",
    "avg_time_per_run_s",
    "trace_dir",
    "jax_mem_before_bytes",
    "jax_mem_after_bytes",
    "jax_peak_mem_bytes",
    "rss_before_bytes",
    "rss_after_bytes",
]


def save_profiling_csv(
    results: list[dict],
    csv_path: str | Path,
) -> Path:
    """Write profiling results to a CSV file for later re-plotting.

    Parameters
    ----------
    results : list[dict]
        The list returned by ``JaqsiProfiler.run_all()``.
    csv_path : str or Path
        Output file path.

    Returns
    -------
    Path
        The path the CSV was written to.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILING_CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "circuit": r.get("circuit", ""),
                "n_layers": r.get("n_layers", ""),
                "mode": r["mode"],
                "n_qubits": r["n_qubits"],
                "batch_size": r["batch_size"],
                "profile_runs": r["profile_runs"],
                "wall_time_s": f"{r['wall_time_s']:.6f}",
                "avg_time_per_run_s": f"{r['avg_time_per_run_s']:.6f}",
                "trace_dir": r.get("trace_dir", ""),
                "jax_mem_before_bytes": r.get("jax_mem_before_bytes", 0),
                "jax_mem_after_bytes": r.get("jax_mem_after_bytes", 0),
                "jax_peak_mem_bytes": r.get("jax_peak_mem_bytes", 0),
                "rss_before_bytes": r.get("rss_before_bytes", 0),
                "rss_after_bytes": r.get("rss_after_bytes", 0),
            })
    logger.info("Profiling CSV saved to %s", csv_path)
    return csv_path


# ------------------------------------------------------------------
# Plotting – publication quality
# ------------------------------------------------------------------

def plot_scaling(
    results: Dict[str, ProfilingTimings],
    *,
    title: str = "JAQSI Execution Time vs Qubit Count",
    output_path: Optional[str | Path] = None,
    show: bool = False,
    use_avg: bool = True,
    log_y: bool = True,
) -> None:
    """Plot execution time vs qubit count — one line per mode.

    This is the primary figure for a paper: it shows how JAQSI scales
    across measurement modes as qubit count grows.

    Parameters
    ----------
    results : dict
        Mapping of mode → :class:`ProfilingTimings` (from
        :func:`load_profiling_results` or :func:`load_profiling_csv`).
    title : str
        Figure suptitle.
    output_path : str or Path, optional
        If given, the figure is written to this base path as PGF and PNG.
    show : bool
        Call ``plt.show()`` after rendering.
    use_avg : bool
        If *True* plot per-run average; otherwise plot total wall time.
    log_y : bool
        Use logarithmic y-axis.
    """
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(COLWIDTH_IN, COLWIDTH_IN * 0.78))

        for mode, pt in results.items():
            y_vals = pt.avg_time_s if use_avg else pt.wall_time_s
            y_ms = [v * 1000.0 for v in y_vals]  # convert to ms

            ax.plot(
                pt.qubit_sizes,
                y_ms,
                label=mode,
                color=MODE_COLORS[mode],
                marker=MODE_MARKERS[mode],
                linestyle="-",
                linewidth=2,
                markersize=6,
                alpha=0.9,
            )

        ax.set_xlabel("n Qubits")
        ylabel = "Avg time per run (ms)" if use_avg else "Total wall time (ms)"
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        if results:
            any_pt = next(iter(results.values()))
            _set_integer_xticks(ax, any_pt.qubit_sizes)
        ax.legend(framealpha=0.9)
        ax.set_title(title)

        plt.tight_layout()
        if output_path is not None:
            _save_figure(fig, output_path)
        if show:
            plt.show()
        plt.close(fig)


def plot_per_mode(
    results: Dict[str, ProfilingTimings],
    *,
    output_path: Optional[str | Path] = None,
    show: bool = False,
    log_y: bool = True,
) -> None:
    """One subplot per measurement mode showing scaling behaviour.

    This is useful when modes have vastly different magnitudes and
    sharing a y-axis compresses the smaller modes.

    Parameters
    ----------
    results : dict
        Mapping of mode → :class:`ProfilingTimings`.
    output_path : str or Path, optional
        If given, the figure is written to this base path as PGF and PNG.
    show : bool
        Call ``plt.show()`` after rendering.
    log_y : bool
        Use logarithmic y-axis.
    """
    n_modes = len(results)
    if n_modes == 0:
        logger.warning("No results to plot.")
        return

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            1,
            n_modes,
            figsize=(TEXTWIDTH_IN, max(2.6, TEXTWIDTH_IN / n_modes * 0.8)),
            squeeze=False,
        )
        axes = axes.flatten()

        for ax, (mode, pt) in zip(axes, results.items()):
            y_ms = [v * 1000.0 for v in pt.avg_time_s]

            ax.plot(
                pt.qubit_sizes,
                y_ms,
                color=MODE_COLORS[mode],
                marker=MODE_MARKERS[mode],
                linestyle="-",
                linewidth=2,
                markersize=6,
                alpha=0.9,
            )
            ax.set_xlabel("n Qubits")
            ax.set_ylabel("Avg time per run (ms)")
            ax.set_title(mode)
            if log_y:
                ax.set_yscale("log")
            _set_integer_xticks(ax, pt.qubit_sizes)
            ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()

        if output_path is not None:
            _save_figure(fig, output_path)
        if show:
            plt.show()
        plt.close(fig)


def plot_mode_comparison_bar(
    results: Dict[str, ProfilingTimings],
    *,
    qubit_count: Optional[int] = None,
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Bar chart comparing modes at a fixed qubit count.

    If *qubit_count* is ``None`` the largest available qubit count is
    used.  This figure is ideal for a paper table/figure that answers
    "which measurement mode is cheapest at scale?".

    Parameters
    ----------
    results : dict
        Mapping of mode → :class:`ProfilingTimings`.
    qubit_count : int, optional
        The qubit count to compare.  Defaults to the largest available.
    title : str, optional
        Figure title.
    output_path : str or Path, optional
        If given, the figure is written to this base path as PGF and PNG.
    show : bool
        Call ``plt.show()`` after rendering.
    """
    if not results:
        logger.warning("No results to plot.")
        return

    # Determine the qubit count to use
    if qubit_count is None:
        all_qubits = set()
        for pt in results.values():
            all_qubits.update(pt.qubit_sizes)
        qubit_count = max(all_qubits)

    modes: List[str] = []
    times_ms: List[float] = []

    for mode, pt in results.items():
        if qubit_count in pt.qubit_sizes:
            idx = pt.qubit_sizes.index(qubit_count)
            modes.append(mode)
            times_ms.append(pt.avg_time_s[idx] * 1000.0)

    if not modes:
        logger.warning("No data found for qubit_count=%d", qubit_count)
        return

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(COLWIDTH_IN, COLWIDTH_IN * 0.85))

        colors = [MODE_COLORS[m] for m in modes]
        x_pos = list(range(len(modes)))

        bars = ax.bar(x_pos, times_ms, color=colors, alpha=0.85, edgecolor="white")

        # Value labels on top of bars
        for bar, val in zip(bars, times_ms):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(modes)
        ax.set_ylabel("Avg time per run (ms)")
        ax.set_title(
            title or f"JAQSI Mode Comparison — {qubit_count} qubits"
        )
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        plt.tight_layout()
        if output_path is not None:
            _save_figure(fig, output_path)
        if show:
            plt.show()
        plt.close(fig)


def plot_memory(
    results: Dict[str, ProfilingTimings],
    *,
    title: str = "JAQSI Peak Memory vs Qubit Count",
    output_path: Optional[str | Path] = None,
    show: bool = False,
    log_y: bool = True,
) -> None:
    """Plot peak JAX device memory vs qubit count — one line per mode.

    This figure shows how JAQSI's memory footprint scales with the
    number of qubits for each measurement mode.  It is especially
    useful for spotting modes with exponential memory growth (e.g.
    density-matrix simulation).

    Parameters
    ----------
    results : dict
        Mapping of mode → :class:`ProfilingTimings` (from
        :func:`load_profiling_results` or :func:`load_profiling_csv`).
    title : str
        Figure suptitle.
    output_path : str or Path, optional
        If given, the figure is written to this base path as PGF and PNG.
    show : bool
        Call ``plt.show()`` after rendering.
    log_y : bool
        Use logarithmic y-axis (recommended given exponential scaling).
    """
    # Check that memory data is available
    has_data = any(
        any(v > 0 for v in pt.jax_peak_mem_bytes)
        for pt in results.values()
    )
    if not has_data:
        logger.warning(
            "No JAX memory data found — skipping memory plot. "
            "Re-run profiling to collect memory metrics."
        )
        return

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(COLWIDTH_IN, COLWIDTH_IN * 0.78))

        for mode, pt in results.items():
            mem_mb = [b / (1024 * 1024) for b in pt.jax_peak_mem_bytes]

            # Skip modes where all values are zero (no data)
            if all(v == 0 for v in mem_mb):
                continue

            ax.plot(
                pt.qubit_sizes,
                mem_mb,
                label=mode,
                color=MODE_COLORS[mode],
                marker=MODE_MARKERS[mode],
                linestyle="-",
                linewidth=2,
                markersize=6,
                alpha=0.9,
            )

        ax.set_xlabel("n Qubits")
        ax.set_ylabel("Peak JAX memory (MB)")
        if log_y:
            ax.set_yscale("log")
        if results:
            any_pt = next(iter(results.values()))
            _set_integer_xticks(ax, any_pt.qubit_sizes)
        ax.legend(framealpha=0.9)
        ax.set_title(title)

        plt.tight_layout()
        if output_path is not None:
            _save_figure(fig, output_path)
        if show:
            plt.show()
        plt.close(fig)


def print_profiling_summary(results: Dict[str, ProfilingTimings]) -> None:
    """Print a human-readable summary table to the logger."""
    header = (
        f"{'Mode':<10} {'Qubits':>6} {'Batch':>6} {'Runs':>5} "
        f"{'Avg (ms)':>10} {'JAX Peak (MB)':>14}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for mode, pt in results.items():
        for i, q in enumerate(pt.qubit_sizes):
            avg_ms = pt.avg_time_s[i] * 1000.0
            peak_mb = pt.jax_peak_mem_bytes[i] / (1024 * 1024)
            logger.info(
                f"{mode:<10} {q:>6} {pt.batch_sizes[i]:>6} "
                f"{pt.profile_runs[i]:>5} {avg_ms:>10.3f} {peak_mb:>14.3f}"
            )