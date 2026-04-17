"""Visualization and result aggregation for benchmark CSV files."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

logger = logging.getLogger(__name__)

# Reference simulator used as denominator in ratio plots
REFERENCE_SIMULATOR = "yaqsi"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class SimTimings:
    """Timing data for a single simulator within a mode."""

    qubit_sizes: List[int] = field(default_factory=list)
    mean_ms: List[float] = field(default_factory=list)
    std_ms: List[float] = field(default_factory=list)


@dataclass
class ModeResults:
    """Aggregated timing data for a single measurement mode.

    ``simulators`` maps simulator name → :class:`SimTimings`.
    Only qubit counts where *all* simulators have data are included.
    """

    qubit_sizes: List[int] = field(default_factory=list)
    simulators: Dict[str, SimTimings] = field(default_factory=dict)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_results(csv_path: str | Path) -> Dict[str, ModeResults]:
    """Parse a benchmark CSV into per-mode result containers.

    Returns a dict mapping mode name → :class:`ModeResults`.
    Only qubit counts where *all* simulators present in the file have
    data are included (so partial runs are handled gracefully).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    # First pass: collect rows keyed by (n_qubits, mode, simulator)
    raw: Dict[Tuple[int, str], Dict[str, dict]] = {}
    all_simulators: set[str] = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["n_qubits"]), row["mode"])
            if key not in raw:
                raw[key] = {}
            sim_name = row["simulator"]
            all_simulators.add(sim_name)
            raw[key][sim_name] = {
                "mean_ms": float(row["mean_ms"]),
                "std_ms": float(row["std_ms"]),
            }

    # Second pass: organise by mode, keeping only complete qubit rows
    modes_seen: Dict[str, ModeResults] = {}
    for (n_qubits, mode) in sorted(raw.keys()):
        entry = raw[(n_qubits, mode)]
        # Only include rows where every simulator seen in the file has data
        if not all_simulators.issubset(entry.keys()):
            continue

        if mode not in modes_seen:
            modes_seen[mode] = ModeResults()
        mr = modes_seen[mode]
        mr.qubit_sizes.append(n_qubits)

        for sim_name in sorted(all_simulators):
            if sim_name not in mr.simulators:
                mr.simulators[sim_name] = SimTimings()
            st = mr.simulators[sim_name]
            st.qubit_sizes.append(n_qubits)
            st.mean_ms.append(entry[sim_name]["mean_ms"])
            st.std_ms.append(entry[sim_name]["std_ms"])

    return modes_seen


# ------------------------------------------------------------------
# Ratio computation
# ------------------------------------------------------------------

def _compute_ratio_with_error(
    ref_mean: List[float],
    ref_std: List[float],
    other_mean: List[float],
    other_std: List[float],
) -> Tuple[List[float], List[float]]:
    """Compute other/ref ratio and propagated uncertainty."""
    ratios: List[float] = []
    errors: List[float] = []
    for rm, rs, om, os_ in zip(ref_mean, ref_std, other_mean, other_std):
        r = om / rm
        # σ_r = r * sqrt((σ_other/other)² + (σ_ref/ref)²)
        err = r * ((os_ / om) ** 2 + (rs / rm) ** 2) ** 0.5
        ratios.append(r)
        errors.append(err)
    return ratios, errors


# ------------------------------------------------------------------
# Colors & styling
# ------------------------------------------------------------------

MODE_COLORS: Dict[str, str] = {
    "probs": "#E69F00",
    "expval": "#ED665A",
    "state": "#009371",
    "density": "#002D4C",
}

# Per-simulator line styles and markers so they are visually distinct
SIMULATOR_STYLES: Dict[str, dict] = {
    "yaqsi": {"linestyle": "-", "marker": "o"},
    "pennylane": {"linestyle": "--", "marker": "s"},
    "qiskit": {"linestyle": "-.", "marker": "D"},
}

_DEFAULT_STYLE = {"linestyle": ":", "marker": "^"}


def _sim_style(sim_name: str) -> dict:
    return SIMULATOR_STYLES.get(sim_name, _DEFAULT_STYLE)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_ratio(
    results: Dict[str, ModeResults],
    *,
    reference: str = REFERENCE_SIMULATOR,
    title_suffix: str = "",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Create a time-ratio plot (other / *reference*) for every non-reference simulator.

    One sub-plot per non-reference simulator; within each plot lines are
    coloured by measurement mode.
    """
    # Determine which other simulators exist
    other_sims: List[str] = []
    for mr in results.values():
        for s in mr.simulators:
            if s != reference and s not in other_sims:
                other_sims.append(s)
    if not other_sims:
        logger.warning("No non-reference simulators to plot ratios for.")
        return

    n_plots = len(other_sims)
    fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 5), squeeze=False)
    axes = axes.flatten()

    for ax, other_sim in zip(axes, other_sims):
        for mode, mr in results.items():
            if reference not in mr.simulators or other_sim not in mr.simulators:
                continue
            color = MODE_COLORS.get(mode, "#333333")
            ref_st = mr.simulators[reference]
            oth_st = mr.simulators[other_sim]
            ratios, errors = _compute_ratio_with_error(
                ref_st.mean_ms, ref_st.std_ms,
                oth_st.mean_ms, oth_st.std_ms,
            )
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

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=2)
        ax.set_xlabel("Number of qubits")
        ax.set_ylabel(f"Time ratio  {other_sim} / {reference}")
        title = f"{other_sim} / {reference}"
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_title(title)
        ax.set_yscale("log")

        all_qubits = sorted(
            {q for mr in results.values() for q in mr.qubit_sizes}
        )
        if all_qubits:
            ax.set_xticks(all_qubits)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, linestyle=":", alpha=0.6)

        # Mode legend
        mode_handles = [
            mlines.Line2D(
                [], [], color=MODE_COLORS.get(m, "#333333"),
                linestyle="-", marker="o", linewidth=2, label=m,
            )
            for m in results
        ]
        ax.legend(
            handles=mode_handles, title="Mode",
            loc="lower left", fontsize=9, title_fontsize=10,
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

    Within each subplot every simulator is drawn with a distinct
    line style / marker; the colour encodes the mode.
    """
    n_modes = len(results)
    if n_modes == 0:
        logger.warning("No results to plot.")
        return

    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5), squeeze=False)
    axes = axes.flatten()

    for ax, (mode, mr) in zip(axes, results.items()):
        color = MODE_COLORS.get(mode, "#333333")

        for sim_name, st in mr.simulators.items():
            style = _sim_style(sim_name)
            ax.errorbar(
                st.qubit_sizes,
                st.mean_ms,
                yerr=st.std_ms,
                label=sim_name,
                color=color,
                linewidth=2,
                capsize=3,
                alpha=0.85,
                **style,
            )

        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Time (ms)")
        ax.set_title(mode)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    suptitle = "Absolute Timings"
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
    # Determine all simulators across all modes
    all_sims: List[str] = []
    for mr in results.values():
        for s in mr.simulators:
            if s not in all_sims:
                all_sims.append(s)

    sim_col_width = 14
    header_parts = [f"{'Mode':<10}", f"{'Qubits':>6}"]
    for s in all_sims:
        header_parts.append(f"{s + ' (ms)':>{sim_col_width}}")
    header = " ".join(header_parts)
    logger.info(header)
    logger.info("-" * len(header))

    for mode, mr in results.items():
        for i, q in enumerate(mr.qubit_sizes):
            parts = [f"{mode:<10}", f"{q:>6}"]
            for s in all_sims:
                if s in mr.simulators:
                    val = mr.simulators[s].mean_ms[i]
                    parts.append(f"{val:>{sim_col_width}.3f}")
                else:
                    parts.append(f"{'n/a':>{sim_col_width}}")
            logger.info(" ".join(parts))