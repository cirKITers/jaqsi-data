"""Abstract base class for quantum-simulator benchmarks."""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.circuits import CircuitSpec
from benchmark.threads import num_threads

# ``grad`` is not a measurement but a differentiation workload: the gradient of
# $\sum_i \langle Z_i \rangle$, summed over the batch, with respect to the
# circuit's trainable parameter vector.  It shares the runner's mode loop
# because it is timed and cross-validated exactly like a measurement.
Mode = Literal["probs", "expval", "state", "density", "grad"]


def _endian_reverse_indices(n_qubits: int) -> np.ndarray:
    """Return the permutation mapping little-endian to big-endian basis order.

    Little-endian simulators (Qiskit, Qulacs) label qubit 0 as the
    least-significant bit, so basis index $b_{n-1}\\dots b_1 b_0$ corresponds
    to $b_0 b_1 \\dots b_{n-1}$ in the big-endian convention used by JAQSI,
    PennyLane and Qibo. The returned array re-sorts a length-$2^n$ vector from
    little-endian to big-endian order.
    """
    N = 1 << n_qubits
    indices = np.zeros(N, dtype=int)
    for i in range(N):
        rev = int(f"{i:0{n_qubits}b}"[::-1], 2)
        indices[rev] = i
    return indices


@dataclass
class BenchmarkResult:
    """Container for a single benchmark measurement."""

    simulator: str
    mode: Mode
    n_qubits: int
    circuit: str
    n_layers: int
    batch_size: int
    n_iters: int
    # Threads every simulator was pinned to, or ``None`` when nothing was.
    threads: Optional[int]
    mean_ms: float
    std_ms: float
    raw_output: jnp.ndarray  # last execution result – used for correctness checks
    # Deviation from the gate-level simulation, filled in by the runner for
    # pulse-level simulators only.
    infidelity: Optional[float] = None


class SimulatorBenchmark(ABC):
    """Interface every simulator adapter must implement."""

    name: str  # e.g. "jaqsi", "pennylane"

    @abstractmethod
    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        """Prepare the circuit / device for a given circuit and mode.

        Called once before warmup and timing loops so that device
        instantiation time is not included in the measurement.  When
        optimal_config is set, the adapter selects its performance-optimized
        configuration instead of the default fallback.
        """

    @abstractmethod
    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """Run a single *un-timed* execution to trigger JIT compilation."""

    @abstractmethod
    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """Execute one batched pass and return the result.

        *inputs* has shape ``(batch, n_inputs)`` and *weights* the unbatched
        shape ``(n_weights,)``.
        """

    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        """Whether this adapter can run *mode* on *spec*.

        The runner skips and logs unsupported combinations instead of failing,
        so a simulator without a gradient interface still contributes its
        forward measurements.
        """
        return True

    # ------------------------------------------------------------------
    # Shared timing harness
    # ------------------------------------------------------------------
    def benchmark(
        self,
        spec: CircuitSpec,
        mode: Mode,
        all_inputs: jnp.ndarray,
        all_weights: jnp.ndarray,
        *,
        do_warmup: bool = True,
        optimal_config: bool = False,
    ) -> BenchmarkResult:
        """Time ``self.run`` over *n_iters* iterations.

        Parameters
        ----------
        spec:
            Circuit to execute.
        mode:
            Measurement mode, or ``grad`` for the differentiation workload.
        all_inputs:
            Array of shape ``(n_iters + 1, batch_size, n_inputs)`` where the
            *last* entry is used for warmup and entries ``0 .. n_iters-1`` are
            used for the timed loop.
        all_weights:
            Array of shape ``(n_iters + 1, n_weights)`` following the same
            convention.  Weights are shared across the batch and redrawn every
            iteration, so no simulator can cache across the timed loop.
        do_warmup:
            Whether to run a warmup pass before timing.
        optimal_config:
            Whether to use the performance-optimized simulator configuration.
        """
        n_iters = all_inputs.shape[0] - 1
        batch_size = all_inputs.shape[1]

        self.setup(spec, mode, optimal_config=optimal_config)

        if do_warmup:
            jax.block_until_ready(self.warmup(all_inputs[-1], all_weights[-1]))

        times: list[float] = []
        result = None
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for i in range(n_iters):
                t0 = time.perf_counter()
                # JAX dispatches asynchronously, so an unforced result would
                # time the enqueue rather than the simulation.  Blocking is a
                # no-op for the adapters that already return materialized data.
                result = jax.block_until_ready(
                    self.run(all_inputs[i], all_weights[i])
                )
                times.append(time.perf_counter() - t0)
        finally:
            if gc_was_enabled:
                gc.enable()

        mean_s = float(np.mean(times))
        std_s = float(np.std(times))

        return BenchmarkResult(
            simulator=self.name,
            mode=mode,
            n_qubits=spec.n_qubits,
            circuit=spec.family,
            n_layers=spec.n_layers,
            batch_size=batch_size,
            n_iters=n_iters,
            # Report what the libraries actually saw, not what was requested.
            threads=num_threads(),
            mean_ms=mean_s * 1000.0,
            std_ms=std_s * 1000.0,
            raw_output=result,
        )