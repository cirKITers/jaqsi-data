"""Visualization and result aggregation for benchmark CSV files."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker

from benchmark.style import PLOT_RC, SIMULATOR_COLORS, style_axes

logger = logging.getLogger(__name__)

# Reference simulators used as denominator in ratio plots, in the order they
# are picked when none is given.  A pulse-level result file contains only the
# pulse reference, a gate-level one only the gate reference.
REFERENCE_SIMULATOR = "jaqsi"
REFERENCE_PREFERENCE = (REFERENCE_SIMULATOR, "jaqsi_pulse")

# Measurement modes to include in the generated figures, per simulation level.
# A single mode keeps the paper figures one column wide.  Set to None to plot
# every mode present in the results file.
PLOT_MODES: Optional[Tuple[str, ...]] = ("expval", "density")


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class SimTimings:
    """Timing data for a single simulator within a mode."""

    qubit_sizes: List[int] = field(default_factory=list)
    mean_ms: List[float] = field(default_factory=list)
    std_ms: List[float] = field(default_factory=list)
    # None where the CSV cell is empty (gate-level rows, expval mode) or where
    # the column is absent entirely (result files predating it).
    infidelity: List[Optional[float]] = field(default_factory=list)


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
            raw_inf = row.get("infidelity", "")
            raw[key][sim_name] = {
                "mean_ms": float(row["mean_ms"]),
                "std_ms": float(row["std_ms"]),
                "infidelity": float(raw_inf) if raw_inf else None,
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
            st.infidelity.append(entry[sim_name]["infidelity"])

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

# Single-column width (inches) of the same layout.  A one-panel figure is sized
# to it so it drops into ``figure`` inside the running text.
COLUMNWIDTH_IN = 3.487

# Double-precision floor. An infidelity at or below $\varepsilon$ means the pulse
# result is indistinguishable from the gate-level reference, so such values are
# clipped to it rather than dropped by the logarithmic axis.
MACHINE_EPS = 2.220446049250313e-16


def _figsize(n_modes: int) -> Tuple[float, float]:
    """Figure size for *n_modes* side-by-side panels."""
    width = COLUMNWIDTH_IN if n_modes <= 2 else TEXTWIDTH_IN
    return width, max(2.2, width / n_modes * 0.8)


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
    """Lay out the subplots and add one de-duplicated legend above them.

    All subplots share the same simulators, so a single horizontal legend on
    top is cleaner than per-axis legends overlapping the data.  A single-column
    figure is too narrow for one legend row, so the entries wrap and the
    reserved headroom grows with the number of rows.
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
    if not handles:
        fig.tight_layout()
        return

    ncol = min(len(labels), 3 if fig.get_figwidth() < 5 else 6)
    rows = -(-len(labels) // ncol)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.065 * rows))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 1.0),
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

def _select_modes(results: Dict[str, ModeResults]) -> Dict[str, ModeResults]:
    """Drop modes not listed in :data:`PLOT_MODES`."""
    if PLOT_MODES is None:
        return results
    # Panels follow the PLOT_MODES order, not the order found in the CSV.
    return {m: results[m] for m in PLOT_MODES if m in results}


def _infidelity_series(
    results: Dict[str, ModeResults],
) -> Dict[str, Dict[str, Tuple[List[int], List[float]]]]:
    """Extract the measured infidelities, dropping empty modes and simulators.

    Returns mode → simulator → ``(qubit_sizes, infidelities)``, keeping only the
    qubit counts where that simulator reported a value.
    """
    series: Dict[str, Dict[str, Tuple[List[int], List[float]]]] = {}
    for mode, mr in results.items():
        per_sim: Dict[str, Tuple[List[int], List[float]]] = {}
        for sim_name, st in mr.simulators.items():
            points = [
                (q, v) for q, v in zip(st.qubit_sizes, st.infidelity) if v is not None
            ]
            if points:
                per_sim[sim_name] = ([q for q, _ in points], [v for _, v in points])
        if per_sim:
            series[mode] = per_sim
    return series


def _label(sim: str) -> str:
    """Display name for a simulator; pulse adapters drop their suffix."""
    return sim.removesuffix("_pulse").capitalize()


def _pick_reference(results: Dict[str, ModeResults]) -> str:
    """Return the first reference simulator present in *results*."""
    present = {s for mr in results.values() for s in mr.simulators}
    for candidate in REFERENCE_PREFERENCE:
        if candidate in present:
            return candidate
    return REFERENCE_SIMULATOR


def plot_ratio(
    results: Dict[str, ModeResults],
    *,
    reference: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Create a time-ratio plot (other / *reference*).

    One subplot per measurement mode; within each subplot every
    non-reference simulator is drawn with a distinct colour.  When *reference*
    is omitted it is taken from the simulators present in *results*.
    """
    results = _select_modes(results)
    n_modes = len(results)
    if n_modes == 0:
        logger.warning("No results to plot.")
        return

    if reference is None:
        reference = _pick_reference(results)

    # Determine which other simulators exist
    other_sims: List[str] = []
    for mr in results.values():
        for s in mr.simulators:
            if s != reference and s not in other_sims:
                other_sims.append(s)
    if not other_sims:
        logger.warning("No non-reference simulators to plot ratios for.")
        return

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            1, n_modes, figsize=_figsize(n_modes),
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
                    label=_label(other_sim),
                    color=SIMULATOR_COLORS[other_sim],
                    linestyle="-",
                    marker="o",
                    capsize=3,
                    capthick=1.0,
                    elinewidth=1.0,
                    alpha=0.9,
                )

            ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0)
            ax.set_xlabel("n Qubits")
            if n_modes > 1:
                ax.set_title(mode.capitalize())
            ax.set_yscale("log")
            _set_integer_xticks(ax, mr.qubit_sizes)
            style_axes(ax)

        axes[0].set_ylabel(f"Time ratio vs {_label(reference)}")
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
    results = _select_modes(results)
    n_modes = len(results)
    if n_modes == 0:
        logger.warning("No results to plot.")
        return

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            1, n_modes, figsize=_figsize(n_modes),
            sharey=True, squeeze=False,
        )
        axes = axes.flatten()

        for ax, (mode, mr) in zip(axes, results.items()):
            for sim_name, st in mr.simulators.items():
                ax.errorbar(
                    st.qubit_sizes,
                    st.mean_ms,
                    yerr=st.std_ms,
                    label=_label(sim_name),
                    color=SIMULATOR_COLORS[sim_name],
                    linestyle="-",
                    marker="o",
                    capsize=3,
                    alpha=0.9,
                )

            ax.set_xlabel("n Qubits")
            if n_modes > 1:
                ax.set_title(mode.capitalize())
            ax.set_yscale("log")
            _set_integer_xticks(ax, mr.qubit_sizes)
            style_axes(ax)

        axes[0].set_ylabel("Time (ms)")
        _add_shared_legend(fig, axes)

        if output_path is not None:
            _save_figure(fig, output_path)
        if show:
            plt.show()
        plt.close(fig)


def plot_infidelity(
    results: Dict[str, ModeResults],
    *,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Create a side-by-side infidelity plot for each mode.

    Only pulse-level rows carry an infidelity, and only for the modes whose
    output defines a state or a distribution, so modes and simulators without
    data are dropped instead of drawn as empty panels.  Values at or below
    :data:`MACHINE_EPS` are clipped to it and the floor is marked with a dashed
    line.
    """
    data = _infidelity_series(_select_modes(results))
    n_modes = len(data)
    if n_modes == 0:
        logger.info("No infidelity data to plot.")
        return

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            1, n_modes, figsize=_figsize(n_modes),
            sharey=True, squeeze=False,
        )
        axes = axes.flatten()

        for ax, (mode, series) in zip(axes, data.items()):
            for sim_name, (qubit_sizes, values) in series.items():
                ax.plot(
                    qubit_sizes,
                    [max(v, MACHINE_EPS) for v in values],
                    label=_label(sim_name),
                    color=SIMULATOR_COLORS[sim_name],
                    linestyle="-",
                    marker="o",
                    alpha=0.9,
                )

            all_qubit_sizes = sorted({q for qs, _ in series.values() for q in qs})
            ax.axhline(MACHINE_EPS, color="0.4", linestyle="--", linewidth=1.0)
            ax.set_xlabel("n Qubits")
            if n_modes > 1:
                ax.set_title(mode.capitalize())
            ax.set_yscale("log")
            _set_integer_xticks(ax, all_qubit_sizes)
            style_axes(ax)

        axes[0].set_ylabel("Infidelity $1 - F$")
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