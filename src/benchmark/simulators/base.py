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
#
# ``noise`` returns the density matrix like ``density``, but of the circuit
# with a depolarizing channel after every gate.  The state it evolves is mixed,
# so no simulator can form the result from a state vector.
Mode = Literal["probs", "expval", "state", "density", "grad", "noise"]


def _endian_reverse_indices(n_qubits: int) -> np.ndarray:
    """Return indices that reorder a state from little- to big-endian basis.

    Qiskit and Qulacs use little-endian basis indices; JAQSI, PennyLane, and
    Qibo use big-endian indices. The permutation reverses the qubit bits of
    each basis index.
    """
    N = 1 << n_qubits
    indices = np.zeros(N, dtype=int)
    for i in range(N):
        rev = int(f"{i:0{n_qubits}b}"[::-1], 2)
        indices[rev] = i
    return indices


@dataclass
class BenchmarkResult:
    """Store one timed benchmark result and its output."""

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
    """Define the interface and timing harness for simulator adapters."""

    name: str  # e.g. "jaqsi", "pennylane"

    envelope: str = "gaussian"

    @abstractmethod
    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        """Prepare the circuit and device outside the timed loop.

        ``optimal_config`` selects the adapter's performance configuration.
        """

    @abstractmethod
    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """Run once before timing to trigger compilation."""

    @abstractmethod
    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """Execute one batch and return its output.

        ``inputs`` has shape ``(batch, n_inputs)``; shared ``weights`` has shape
        ``(n_weights,)``.
        """

    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        """Return whether the adapter supports ``mode`` on ``spec``.

        The runner skips unsupported combinations.
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
        """Time ``run`` for one circuit and mode.

        ``all_inputs`` has shape ``(n_iters + 1, batch_size, n_inputs)`` and
        ``all_weights`` has shape ``(n_iters + 1, n_weights)``. The final entry
        warms up the adapter; the others are timed. Weights are shared within each
        batch and redrawn per iteration. ``do_warmup`` controls the warmup pass,
        and ``optimal_config`` selects the adapter's performance configuration.
        """
        n_iters = all_inputs.shape[0] - 1
        batch_size = all_inputs.shape[1]

        self.setup(spec, mode, optimal_config=optimal_config)

        # Collect before the warmup
        gc.collect()
        if do_warmup:
            jax.block_until_ready(self.warmup(all_inputs[-1], all_weights[-1]))

        times: list[float] = []
        result = None
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