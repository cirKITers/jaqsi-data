"""CLI entry-point for the benchmark suite.

Usage examples::

    # Run with default config
    python -m benchmark

    # Run with a custom config file
    python -m benchmark --config path/to/config.yaml

    # Override specific values
    python -m benchmark qubits.max=10 execution.n_iters=20

    # Only visualise existing results (skip computation)
    python -m benchmark --visualize-only results/benchmarks-20250101120000.csv

    # Resume a previously interrupted run by specifying its identifier
    python -m benchmark output.identifier=20250101120000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quantum simulator benchmarking suite"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file (default: built-in default.yaml)",
    )
    parser.add_argument(
        "--visualize-only",
        type=str,
        default=None,
        metavar="CSV",
        help="Skip benchmarking; only generate plots from an existing CSV file.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation after benchmarking.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() to display plots interactively.",
    )

    # Everything that is not a recognised flag is treated as a dotlist override
    args, overrides = parser.parse_known_args(argv)

    from benchmark.config import load_config
    from benchmark.visualize import (
        load_results,
        plot_ratio,
        plot_absolute,
        plot_infidelity,
        print_summary,
    )

    if args.visualize_only:
        csv_path = Path(args.visualize_only)
        logger.info(f"Visualise-only mode: loading {csv_path}")
        results = load_results(csv_path)
        print_summary(results)
        stem = csv_path.stem  # e.g. "benchmarks-20250101"
        plot_ratio(
            results,
            output_path=csv_path.parent / f"{stem}-ratio.pgf",
            show=args.show,
        )
        plot_absolute(
            results,
            output_path=csv_path.parent / f"{stem}-absolute.pgf",
            show=args.show,
        )
        plot_infidelity(
            results,
            output_path=csv_path.parent / f"{stem}-infidelity.pgf",
            show=args.show,
        )
        return

    cfg = load_config(config_path=args.config, overrides=overrides)
    logger.info(f"Benchmark identifier: {cfg.output.identifier}")
    logger.info(
        f"Config: qubits={cfg.qubits.min}–{cfg.qubits.max}, "
        f"modes={cfg.modes}, simulators={cfg.simulators}, "
        f"iters={cfg.execution.n_iters}, "
        f"batch={cfg.execution.batch_size}, warmup={cfg.warmup}"
    )

    from benchmark.runner import run_benchmarks

    csv_path = run_benchmarks(cfg)

    if not args.no_plot:
        results = load_results(csv_path)
        print_summary(results)
        plot_ratio(
            results,
            output_path=csv_path.parent / f"{csv_path.stem}-ratio.pgf",
            show=args.show,
        )
        plot_absolute(
            results,
            output_path=csv_path.parent / f"{csv_path.stem}-absolute.pgf",
            show=args.show,
        )
        plot_infidelity(
            results,
            output_path=csv_path.parent / f"{csv_path.stem}-infidelity.pgf",
            show=args.show,
        )


if __name__ == "__main__":
    main()