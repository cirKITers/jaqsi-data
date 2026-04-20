"""Visualization and result aggregation for benchmark CSV files."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker

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

# Per-simulator colours — visually distinct and colourblind-friendly
SIMULATOR_COLORS: Dict[str, str] = {
    "yaqsi": "#1f77b4",      # blue
    "pennylane": "#ff7f0e",  # orange
    "qiskit": "#2ca02c",     # green
    "qibo": "#d62728",       # red
}

# Fallback palette for simulators not listed above
_EXTRA_COLORS = [
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

_extra_idx = 0


def _sim_color(sim_name: str) -> str:
    """Return a consistent colour for *sim_name*."""
    global _extra_idx  # noqa: PLW0603
    if sim_name not in SIMULATOR_COLORS:
        SIMULATOR_COLORS[sim_name] = _EXTRA_COLORS[_extra_idx % len(_EXTRA_COLORS)]
        _extra_idx += 1
    return SIMULATOR_COLORS[sim_name]


def _set_integer_xticks(ax: plt.Axes, qubit_sizes: List[int]) -> None:
    """Force x-axis to show only integer tick values."""
    ax.set_xticks(qubit_sizes)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(qubit_sizes))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(
        [str(q) for q in qubit_sizes]
    ))


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
    """Create a time-ratio plot (other / *reference*).

    One subplot per measurement mode; within each subplot every
    non-reference simulator is drawn with a distinct colour.
    """
    n_modes = len(results)
    if n_modes == 0:
        logger.warning("No results to plot.")
        return

    # Determine which other simulators exist
    other_sims: List[str] = []
    for mr in results.values():
        for s in mr.simulators:
            if s != reference and s not in other_sims:
                other_sims.append(s)
    if not other_sims:
        logger.warning("No non-reference simulators to plot ratios for.")
        return

    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5), squeeze=False)
    axes = axes.flatten()

    for ax, (mode, mr) in zip(axes, results.items()):
        if reference not in mr.simulators:
            continue
        ref_st = mr.simulators[reference]

        for other_sim in other_sims:
            if other_sim not in mr.simulators:
                continue
            oth_st = mr.simulators[other_sim]
            ratios, errors = _compute_ratio_with_error(
                ref_st.mean_ms, ref_st.std_ms,
                oth_st.mean_ms, oth_st.std_ms,
            )
            ax.errorbar(
                mr.qubit_sizes,
                ratios,
                yerr=errors,
                label=other_sim,
                color=_sim_color(other_sim),
                linestyle="-",
                marker="o",
                linewidth=2,
                capsize=4,
                capthick=1.5,
                elinewidth=1.2,
                alpha=0.85,
            )

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=2)
        ax.set_xlabel("Number of qubits")
        ax.set_ylabel(f"Time ratio  vs {reference}")
        ax.set_title(mode)
        ax.set_yscale("log")
        _set_integer_xticks(ax, mr.qubit_sizes)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    suptitle = f"Time Ratio vs {reference}"
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


def plot_absolute(
    results: Dict[str, ModeResults],
    *,
    title_suffix: str = "",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Create a side-by-side absolute-time plot for each mode.

    Within each subplot every simulator is drawn with a distinct
    colour.
    """
    n_modes = len(results)
    if n_modes == 0:
        logger.warning("No results to plot.")
        return

    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5), squeeze=False)
    axes = axes.flatten()

    for ax, (mode, mr) in zip(axes, results.items()):
        for sim_name, st in mr.simulators.items():
            ax.errorbar(
                st.qubit_sizes,
                st.mean_ms,
                yerr=st.std_ms,
                label=sim_name,
                color=_sim_color(sim_name),
                linestyle="-",
                marker="o",
                linewidth=2,
                capsize=3,
                alpha=0.85,
            )

        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Time (ms)")
        ax.set_title(mode)
        ax.set_yscale("log")
        _set_integer_xticks(ax, mr.qubit_sizes)
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