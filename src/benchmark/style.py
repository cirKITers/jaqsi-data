"""Shared figure style for benchmark plots.

Mirrors the ggplot2 ``theme_bw`` look used by the paper's R figures: a white
panel with a full grey border, solid light-grey major/minor gridlines, a serif
(Computer Modern) font, a horizontal top legend, and a fixed colour palette.
"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt

COLOURS = [
    "#000000",  # black
    "#E69F00",  # orange
    "#999999",  # grey
    "#009371",  # teal
    "#beaed4",  # purple
    "#ed665a",  # salmon
    "#1f78b4",  # blue
    "#002D4C",  # navy
]

# Per-simulator colours drawn from the paper palette (jaqsi = brand navy).
SIMULATOR_COLORS: Dict[str, str] = {
    "jaqsi": "#002D4C",      # navy
    "pennylane": "#E69F00",  # orange
    # Gradient variants of the same framework: lighter and darker orange, so a
    # gradient figure keeps PennyLane visually grouped.
    "pennylane_lightning": "#F0C471",  # light orange
    "pennylane_psr": "#B37800",      # dark orange
    "qiskit": "#009371",     # teal
    "qibo": "#ed665a",       # salmon
    "qulacs": "#1f78b4",     # blue
    # Pulse-level adapters reuse each framework's colour; they are plotted
    # from separate result files, so the two levels never share a figure.
    "jaqsi_pulse": "#002D4C",      # navy
    "pennylane_pulse": "#E69F00",  # orange
    "qutip_pulse": "#beaed4",      # purple
    "dynamiqs_pulse": "#999999",   # grey
}

# theme_bw rcParams: serif fonts, white panel with a full grey border, solid
# light-grey grid, top legend, and a PGF backend for the pdflatex paper build.
PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#333333",  # grey20 panel border
    "axes.linewidth": 0.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "#EBEBEB",  # grey92
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
    "xtick.color": "#4D4D4D",  # grey30
    "ytick.color": "#4D4D4D",
    "legend.frameon": False,
    "lines.linewidth": 0.8,
    "lines.markersize": 3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
}


def style_axes(ax: plt.Axes) -> None:
    """Apply the per-axis theme_bw touches rcParams cannot express."""
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.5)
    ax.grid(True, which="major", color="#EBEBEB", linewidth=0.4)
    ax.grid(True, which="minor", axis="y", color="#EBEBEB", linewidth=0.2)
