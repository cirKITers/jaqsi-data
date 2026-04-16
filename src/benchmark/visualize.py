"""Visualization and result aggregation for benchmark CSV files."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

@dataclass
class ModeResults:
    """Aggregated timing data for a single measurement mode."""

    qubit_sizes: List[int] = field(default_factory=list)
    ys_mean_ms: List[float] = field(default_factory=list)
    ys_std_ms: List[float] = field(default_factory=list)
    pl_mean_ms: List[float] = field(default_factory=list)
    pl_std_ms: List[float] = field(default_factory=list)


def load_results(csv_path: str | Path) -> Dict[str, ModeResults]:
    """Parse a benchmark CSV into per-mode result containers.

    Returns a dict mapping mode name → ``ModeResults``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    # First pass: collect rows keyed by (n_qubits, mode)
    raw: Dict[tuple, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["n_qubits"]), row["mode"])
            if key not in raw:
                raw[key] = {}
            raw[key][row["simulator"]] = {
                "mean_ms": float(row["mean_ms"]),
                "std_ms": float(row["std_ms"]),
            }

    # Second pass: organise by mode
    modes_seen: Dict[str, ModeResults] = {}
    # Sort keys to guarantee ascending qubit order
    for (n_qubits, mode) in sorted(raw.keys()):
        if mode not in modes_seen:
            modes_seen[mode] = ModeResults()
        mr = modes_seen[mode]
        entry = raw[(n_qubits, mode)]
        if "yaqsi" in entry and "pennylane" in entry:
            mr.qubit_sizes.append(n_qubits)
            mr.ys_mean_ms.append(entry["yaqsi"]["mean_ms"])
            mr.ys_std_ms.append(entry["yaqsi"]["std_ms"])
            mr.pl_mean_ms.append(entry["pennylane"]["mean_ms"])
            mr.pl_std_ms.append(entry["pennylane"]["std_ms"])

    return modes_seen


# ------------------------------------------------------------------
# Ratio computation
# ------------------------------------------------------------------

def _compute_ratio_with_error(
    mr: ModeResults,
) -> tuple[List[float], List[float]]:
    """Compute PL/Yaqsi ratio and propagated uncertainty."""
    ratios: List[float] = []
    errors: List[float] = []
    for ys, pl, sy, sp in zip(
        mr.ys_mean_ms, mr.pl_mean_ms, mr.ys_std_ms, mr.pl_std_ms
    ):
        r = pl / ys
        # σ_r = r * sqrt((σ_pl/pl)² + (σ_ys/ys)²)
        err = r * ((sp / pl) ** 2 + (sy / ys) ** 2) ** 0.5
        ratios.append(r)
        errors.append(err)
    return ratios, errors


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

MODE_COLORS: Dict[str, str] = {
    "probs": "#E69F00",
    "expval": "#ED665A",
    "state": "#009371",
    "density": "#002D4C",
}


def plot_ratio(
    results: Dict[str, ModeResults],
    *,
    title_suffix: str = "",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Create a PennyLane / Yaqsi time-ratio plot.

    Parameters
    ----------
    results:
        Dict returned by :func:`load_results`.
    title_suffix:
        Extra text appended to the plot title (e.g. iter/batch info).
    output_path:
        If given, save the figure to this path.
    show:
        If ``True``, call ``plt.show()``.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for mode, mr in results.items():
        color = MODE_COLORS.get(mode, "#333333")
        ratios, errors = _compute_ratio_with_error(mr)
        ax.errorbar(
            mr.qubit_sizes,
            ratios,
            yerr=errors,
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2,
            capsize=4,
            capthick=1.5,
            elinewidth=1.2,
        )

    # Reference line at ratio = 1
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=2)

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Time ratio  PennyLane / Yaqsi")
    title = "Yaqsi vs PennyLane — Parametric Benchmark"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    ax.set_yscale("log")

    # Determine tick range from data
    all_qubits = sorted(
        {q for mr in results.values() for q in mr.qubit_sizes}
    )
    if all_qubits:
        ax.set_xticks(all_qubits)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, linestyle=":", alpha=0.6)

    # Legend
    mode_handles = [
        mlines.Line2D(
            [],
            [],
            color=MODE_COLORS.get(m, "#333333"),
            linestyle="-",
            marker="o",
            linewidth=2,
            label=m,
        )
        for m in results
    ]
    ax.legend(
        handles=mode_handles,
        title="Simulation Mode",
        loc="lower left",
        fontsize=9,
        title_fontsize=10,
    )

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(str(output_path), dpi=150)
        logger.info(f"Figure saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_absolute(
    results: Dict[str, ModeResults],
    *,
    title_suffix: str = "",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Create a side-by-side absolute-time plot for each mode.

    Parameters
    ----------
    results:
        Dict returned by :func:`load_results`.
    output_path:
        If given, save the figure to this path.
    show:
        If ``True``, call ``plt.show()``.
    """
    n_modes = len(results)
    if n_modes == 0:
        logger.warning("No results to plot.")
        return

    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5), squeeze=False)
    axes = axes.flatten()

    for ax, (mode, mr) in zip(axes, results.items()):
        color = MODE_COLORS.get(mode, "#333333")

        ax.errorbar(
            mr.qubit_sizes,
            mr.ys_mean_ms,
            yerr=mr.ys_std_ms,
            label="Yaqsi",
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2,
            capsize=3,
        )
        ax.errorbar(
            mr.qubit_sizes,
            mr.pl_mean_ms,
            yerr=mr.pl_std_ms,
            label="PennyLane",
            color=color,
            linestyle="--",
            marker="s",
            linewidth=2,
            capsize=3,
            alpha=0.7,
        )

        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Time (ms)")
        ax.set_title(mode)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    suptitle = "Absolute Timings — Yaqsi vs PennyLane"
    if title_suffix:
        suptitle += f" ({title_suffix})"
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150)
        logger.info(f"Figure saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def print_summary(results: Dict[str, ModeResults]) -> None:
    """Print a human-readable summary table to the logger."""
    header = (
        f"{'Mode':<10} {'Qubits':>6} {'Yaqsi (ms)':>14} {'PL (ms)':>14} {'Ratio':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for mode, mr in results.items():
        for q, ys, pl in zip(mr.qubit_sizes, mr.ys_mean_ms, mr.pl_mean_ms):
            ratio = pl / ys if ys > 0 else float("inf")
            logger.info(
                f"{mode:<10} {q:>6} {ys:>14.3f} {pl:>14.3f} {ratio:>8.2f}x"
            )