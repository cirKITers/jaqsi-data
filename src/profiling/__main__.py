"""CLI entry-point for the JAQSI profiling suite.

Usage examples::

    # Run with default settings
    python -m profiling

    # Profile specific modes and qubit counts
    python -m profiling --modes probs expval --qubits 4 8 16

    # Customise warm-up, profiled runs and output directory
    python -m profiling --warmup 3 --runs 5 --output ./traces

    # Re-generate plots from an existing CSV (skip profiling)
    python -m profiling --visualize-only profiling_results/profiling_results.csv

    # Enable verbose (debug) logging
    python -m profiling -v

    # Full list of options
    python -m profiling --help
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("profiling")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Profile the JAQSI quantum simulator using JAX profiler / Perfetto."
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=None,
        help="Qubit counts to profile (default: 2 4 8 12 16).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=None,
        choices=["probs", "expval", "state", "density"],
        help="Measurement modes to profile (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for each execution (default: 1).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warm-up runs before profiling (default: 2).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of profiled simulation runs (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Random seed for parameter generation (default: 1000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profiling_results",
        help="Output directory for trace files (default: profiling_results).",
    )
    parser.add_argument(
        "--visualize-only",
        type=str,
        default=None,
        metavar="CSV",
        help="Skip profiling; only generate plots from an existing CSV file.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation after profiling.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() to display plots interactively.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Visualize-only mode
    # ------------------------------------------------------------------
    if args.visualize_only:
        from profiling.visualize import (
            load_profiling_csv,
            plot_memory,
            plot_mode_comparison_bar,
            plot_per_mode,
            plot_scaling,
            print_profiling_summary,
        )

        csv_path = Path(args.visualize_only)
        logger.info("Visualize-only mode: loading %s", csv_path)
        by_mode = load_profiling_csv(csv_path)
        print_profiling_summary(by_mode)

        out_dir = csv_path.parent
        plot_scaling(
            by_mode,
            output_path=out_dir / "profiling-scaling.pdf",
            show=args.show,
        )
        plot_per_mode(
            by_mode,
            output_path=out_dir / "profiling-per-mode.pdf",
            show=args.show,
        )
        plot_mode_comparison_bar(
            by_mode,
            output_path=out_dir / "profiling-mode-comparison.pdf",
            show=args.show,
        )
        plot_memory(
            by_mode,
            output_path=out_dir / "profiling-memory.pdf",
            show=args.show,
        )
        return

    # ------------------------------------------------------------------
    # Full profiling run
    # ------------------------------------------------------------------
    from profiling.config import ProfilingConfig
    from profiling.profiler import YaqsiProfiler

    defaults = ProfilingConfig()

    config = ProfilingConfig(
        qubit_counts=args.qubits or defaults.qubit_counts,
        modes=args.modes or defaults.modes,
        batch_size=args.batch_size,
        warmup_runs=args.warmup,
        profile_runs=args.runs,
        seed=args.seed,
        output_dir=args.output,
    )

    logger.info(
        "Profiling config: qubits=%s, modes=%s, batch=%d, "
        "warmup=%d, runs=%d, seed=%d",
        config.qubit_counts,
        config.modes,
        config.batch_size,
        config.warmup_runs,
        config.profile_runs,
        config.seed,
    )

    profiler = YaqsiProfiler(config, no_plot=args.no_plot, show=args.show)
    profiler.run_all()


if __name__ == "__main__":
    main()