"""Profile JAQSI with JAX traces viewable in Perfetto.

``JaqsiProfiler`` records traces across configured qubit counts and modes.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import psutil

from benchmark.circuits import build_spec
from benchmark.simulators.jaqsi_sim import JaqsiBenchmark
from profiling.config import ProfilingConfig, Mode
from profiling.visualize import (
    load_profiling_results,
    plot_memory,
    plot_mode_comparison_bar,
    plot_per_mode,
    plot_scaling,
    print_profiling_summary,
    save_profiling_csv,
)

logger = logging.getLogger(__name__)


class JaqsiProfiler:
    """Profile JAQSI with JAX traces.

    ``config`` defaults to ``ProfilingConfig()``.
    """

    def __init__(
        self,
        config: ProfilingConfig | None = None,
        *,
        no_plot: bool = False,
        show: bool = False,
    ) -> None:
        self.config = config or ProfilingConfig()
        self._no_plot = no_plot
        self._show = show
        self._results: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> list[dict]:
        """Profile each configured mode and qubit count.

        Return one result dictionary per combination, including timing data and
        the Perfetto trace directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._results.clear()

        for mode in self.config.modes:
            for n_qubits in self.config.qubit_counts:
                result = self._profile_combination(mode, n_qubits, output_dir)
                self._results.append(result)

        self._write_summary(output_dir)
        self._save_csv(output_dir)
        if not self._no_plot:
            self._generate_plots(output_dir)
        return self._results

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _jax_live_bytes() -> int:
        """Return total bytes held by all live JAX arrays."""
        return sum(a.nbytes for a in jax.live_arrays())

    @staticmethod
    def _process_rss_bytes() -> int:
        """Return the current process RSS in bytes."""
        return psutil.Process(os.getpid()).memory_info().rss

    # ------------------------------------------------------------------
    # Core profiling logic
    # ------------------------------------------------------------------

    def _profile_combination(
        self,
        mode: Mode,
        n_qubits: int,
        output_dir: Path,
    ) -> dict:
        """Warm up and profile one mode and qubit count."""
        logger.info(
            "Profiling JAQSI – mode=%s, n_qubits=%d …", mode, n_qubits
        )

        # Create a fresh simulator instance and set up the circuit
        spec = build_spec(
            self.config.circuit_family, n_qubits, self.config.n_layers
        )
        simulator = JaqsiBenchmark()
        simulator.setup(spec, mode)

        # Generate random parameters (same approach as the benchmark runner)
        rng = jax.random.PRNGKey(self.config.seed)
        input_key, weight_key = jax.random.split(rng)
        total_runs = self.config.warmup_runs + self.config.profile_runs
        all_inputs = jax.random.uniform(
            input_key,
            shape=(total_runs, self.config.batch_size, spec.n_inputs),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )
        all_weights = jax.random.uniform(
            weight_key,
            shape=(total_runs, spec.n_weights),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )

        # -- Warm-up: let JAX trace & JIT-compile ----------------------------
        for i in range(self.config.warmup_runs):
            logger.debug("  warm-up run %d/%d", i + 1, self.config.warmup_runs)
            simulator.warmup(all_inputs[i], all_weights[i])

        # Block until all previous JAX computations have completed so that
        # the profiled region only contains the runs of interest.
        jax.block_until_ready(jnp.zeros(1))

        # -- Memory baseline --------------------------------------------------
        gc.collect()
        jax_mem_before = self._jax_live_bytes()
        rss_before = self._process_rss_bytes()
        peak_jax_mem = jax_mem_before  # track peak across profiled runs

        # -- Profiled region --------------------------------------------------
        trace_name = f"jaqsi_{mode}_{n_qubits}q"
        trace_dir = output_dir / trace_name

        wall_start = time.perf_counter()

        jax.profiler.start_trace(str(trace_dir))
        try:
            for i in range(self.config.profile_runs):
                run_idx = self.config.warmup_runs + i
                logger.debug(
                    "  profiled run %d/%d", i + 1, self.config.profile_runs
                )
                simulator.run(all_inputs[run_idx], all_weights[run_idx])

                # Sample memory after each run to capture the peak
                jax.block_until_ready(jnp.zeros(1))
                current_jax_mem = self._jax_live_bytes()
                if current_jax_mem > peak_jax_mem:
                    peak_jax_mem = current_jax_mem

            # Ensure all asynchronous work finishes inside the trace region.
            jax.block_until_ready(jnp.zeros(1))
        finally:
            jax.profiler.stop_trace()

        wall_elapsed = time.perf_counter() - wall_start

        # -- Memory after profiling -------------------------------------------
        jax_mem_after = self._jax_live_bytes()
        rss_after = self._process_rss_bytes()

        result = {
            "mode": mode,
            "n_qubits": n_qubits,
            "circuit": spec.family,
            "n_layers": spec.n_layers,
            "batch_size": self.config.batch_size,
            "warmup_runs": self.config.warmup_runs,
            "profile_runs": self.config.profile_runs,
            "wall_time_s": wall_elapsed,
            "avg_time_per_run_s": wall_elapsed / self.config.profile_runs,
            "trace_dir": str(trace_dir),
            # Memory metrics (bytes)
            "jax_mem_before_bytes": jax_mem_before,
            "jax_mem_after_bytes": jax_mem_after,
            "jax_peak_mem_bytes": peak_jax_mem,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
        }

        peak_mb = peak_jax_mem / (1024 * 1024)
        rss_delta_mb = (rss_after - rss_before) / (1024 * 1024)
        logger.info(
            "  ✓ %s – total %.4fs (%.4fs / run), "
            "JAX peak %.3f MB, RSS Δ%+.1f MB, trace → %s",
            trace_name,
            wall_elapsed,
            result["avg_time_per_run_s"],
            peak_mb,
            rss_delta_mb,
            trace_dir,
        )
        return result

    def _write_summary(self, output_dir: Path) -> None:
        """Write a readable summary of all profiling results."""
        summary_path = output_dir / "profiling_summary.txt"

        lines: list[str] = []
        lines.append("=" * 96)
        lines.append("JAQSI Profiling Summary")
        lines.append("=" * 96)
        lines.append(
            f"{'Mode':<10} {'Qubits':>6} {'Batch':>6} {'Runs':>5} "
            f"{'Total (s)':>10} {'Avg (s)':>10} "
            f"{'JAX Peak':>10} {'RSS Δ':>10} {'Trace Dir'}"
        )
        lines.append("-" * 96)

        for r in self._results:
            peak_mb = r["jax_peak_mem_bytes"] / (1024 * 1024)
            rss_delta = (r["rss_after_bytes"] - r["rss_before_bytes"]) / (
                1024 * 1024
            )
            lines.append(
                f"{r['mode']:<10} {r['n_qubits']:>6} "
                f"{r['batch_size']:>6} {r['profile_runs']:>5} "
                f"{r['wall_time_s']:>10.4f} {r['avg_time_per_run_s']:>10.4f} "
                f"{peak_mb:>9.3f}M {rss_delta:>+9.1f}M "
                f"{r['trace_dir']}"
            )

        lines.append("=" * 96)
        lines.append("")
        lines.append(
            "Open the trace directories in https://ui.perfetto.dev/ to "
            "inspect detailed JAX/XLA execution traces."
        )
        lines.append("")

        summary_text = "\n".join(lines)
        summary_path.write_text(summary_text)
        logger.info("Summary written to %s", summary_path)
        print(summary_text)

    def _save_csv(self, output_dir: Path) -> Path:
        """Save results as CSV for later plotting."""
        csv_path = output_dir / "profiling_results.csv"
        return save_profiling_csv(self._results, csv_path)

    def _generate_plots(self, output_dir: Path) -> None:
        """Plot the profiling results."""
        by_mode = load_profiling_results(self._results)

        print_profiling_summary(by_mode)

        plot_scaling(
            by_mode,
            title="JAQSI Execution Time vs Qubit Count",
            output_path=output_dir / "profiling-scaling.pgf",
            show=self._show,
        )
        logger.info("Generated scaling plot.")

        plot_per_mode(
            by_mode,
            output_path=output_dir / "profiling-per-mode.pgf",
            show=self._show,
        )
        logger.info("Generated per-mode plot.")

        plot_mode_comparison_bar(
            by_mode,
            output_path=output_dir / "profiling-mode-comparison.pgf",
            show=self._show,
        )
        logger.info("Generated mode comparison bar chart.")

        plot_memory(
            by_mode,
            title="JAQSI Peak Memory vs Qubit Count",
            output_path=output_dir / "profiling-memory.pgf",
            show=self._show,
        )
        logger.info("Generated memory scaling plot.")