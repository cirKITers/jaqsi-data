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
REFERENCE_SIMULATOR = "jaqsi"


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

# Two-column text width (inches) of the IEEE conference paper. Figures are
# sized to this width so they drop into ``figure*`` at their natural scale.
TEXTWIDTH_IN = 7.16

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

# Per-simulator colours — Paul Tol "bright" qualitative scheme (colourblind safe)
SIMULATOR_COLORS: Dict[str, str] = {
    "jaqsi": "#4477AA",      # blue
    "pennylane": "#EE6677",  # red
    "qiskit": "#228833",     # green
    "qibo": "#AA3377",       # purple
}

# Fallback palette for simulators not listed above (remaining Tol bright colours)
_EXTRA_COLORS = [
    "#66CCEE",  # cyan
    "#CCBB44",  # yellow
    "#BBBBBB",  # grey
]

_extra_idx = 0


def _sim_color(sim_name: str) -> str:
    """Return a consistent colour for *sim_name*."""
    global _extra_idx  # noqa: PLW0603
    if sim_name not in SIMULATOR_COLORS:
        SIMULATOR_COLORS[sim_name] = _EXTRA_COLORS[_extra_idx % len(_EXTRA_COLORS)]
        _extra_idx += 1
    return SIMULATOR_COLORS[sim_name]


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    """Write *fig* as PGF (for the paper) and PNG (for quick inspection).

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


def _add_shared_legend(fig: plt.Figure, axes) -> None:
    """Add one de-duplicated legend below the subplots.

    All subplots share the same simulators, so a single horizontal legend at
    the bottom is cleaner than per-axis legends overlapping the data.
    """
    handles: list = []
    labels: list = []
    seen: set = set()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 6),
            bbox_to_anchor=(0.5, 0.0),
        )


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

    height = max(2.6, TEXTWIDTH_IN / n_modes * 0.8)
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            1, n_modes, figsize=(TEXTWIDTH_IN, height),
            sharey=True, squeeze=False,
        )
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
                    label=other_sim.capitalize(),
                    color=_sim_color(other_sim),
                    linestyle="-",
                    marker="o",
                    capsize=3,
                    capthick=1.0,
                    elinewidth=1.0,
                    alpha=0.9,
                )

            ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0)
            ax.set_xlabel("n Qubits")
            ax.set_title(mode.capitalize())
            ax.set_yscale("log")
            _set_integer_xticks(ax, mr.qubit_sizes)
            ax.grid(True, linestyle=":", alpha=0.5)

        axes[0].set_ylabel(f"Time ratio vs {reference.capitalize()}")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        _add_shared_legend(fig, axes)

        if output_path is not None:
            _save_figure(fig, output_path)
        if show:
            plt.show()
        plt.close(fig)


def plot_absolute(
    results: Dict[str, ModeResults],
    *,
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

    height = max(2.6, TEXTWIDTH_IN / n_modes * 0.8)
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            1, n_modes, figsize=(TEXTWIDTH_IN, height),
            sharey=True, squeeze=False,
        )
        axes = axes.flatten()

        for ax, (mode, mr) in zip(axes, results.items()):
            for sim_name, st in mr.simulators.items():
                ax.errorbar(
                    st.qubit_sizes,
                    st.mean_ms,
                    yerr=st.std_ms,
                    label=sim_name.capitalize(),
                    color=_sim_color(sim_name),
                    linestyle="-",
                    marker="o",
                    capsize=3,
                    alpha=0.9,
                )

            ax.set_xlabel("n Qubits")
            ax.set_title(mode.capitalize())
            ax.set_yscale("log")
            _set_integer_xticks(ax, mr.qubit_sizes)
            ax.grid(True, linestyle=":", alpha=0.5)

        axes[0].set_ylabel("Time (ms)")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        _add_shared_legend(fig, axes)

        if output_path is not None:
            _save_figure(fig, output_path)
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