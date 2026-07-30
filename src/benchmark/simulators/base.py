"""Abstract base class for quantum-simulator benchmarks."""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

Mode = Literal["probs", "expval", "state", "density"]


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
    batch_size: int
    n_iters: int
    mean_ms: float
    std_ms: float
    raw_output: jnp.ndarray  # last execution result – used for correctness checks


class SimulatorBenchmark(ABC):
    """Interface every simulator adapter must implement."""

    name: str  # e.g. "jaqsi", "pennylane"

    @abstractmethod
    def setup(self, n_qubits: int, mode: Mode, *, optimal_config: bool = False) -> None:
        """Prepare the circuit / device for a given qubit count and mode.

        Called once before warmup and timing loops so that device
        instantiation time is not included in the measurement.  When
        optimal_config is set, the adapter selects its performance-optimized
        configuration instead of the default fallback.
        """

    @abstractmethod
    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Run a single *un-timed* execution to trigger JIT compilation."""

    @abstractmethod
    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Execute one (possibly batched) forward pass and return the result."""

    # ------------------------------------------------------------------
    # Shared timing harness
    # ------------------------------------------------------------------
    def benchmark(
        self,
        n_qubits: int,
        mode: Mode,
        all_phis: jnp.ndarray,
        *,
        do_warmup: bool = True,
        optimal_config: bool = False,
    ) -> BenchmarkResult:
        """Time ``self.run`` over *n_iters* iterations.

        Parameters
        ----------
        n_qubits:
            Number of qubits in the circuit.
        mode:
            Measurement mode.
        all_phis:
            Array of shape ``(n_iters + 1, batch_size)`` where the *last*
            entry is used for warmup and entries ``0 .. n_iters-1`` are used
            for the timed loop.
        do_warmup:
            Whether to run a warmup pass before timing.
        optimal_config:
            Whether to use the performance-optimized simulator configuration.
        """
        n_iters = all_phis.shape[0] - 1
        batch_size = all_phis.shape[1]

        self.setup(n_qubits, mode, optimal_config=optimal_config)

        if do_warmup:
            self.warmup(all_phis[-1])

        times: list[float] = []
        result = None
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for i in range(n_iters):
                t0 = time.perf_counter()
                result = self.run(all_phis[i])
                times.append(time.perf_counter() - t0)
        finally:
            if gc_was_enabled:
                gc.enable()

        mean_s = float(np.mean(times))
        std_s = float(np.std(times))

        return BenchmarkResult(
            simulator=self.name,
            mode=mode,
            n_qubits=n_qubits,
            batch_size=batch_size,
            n_iters=n_iters,
            mean_ms=mean_s * 1000.0,
            std_ms=std_s * 1000.0,
            raw_output=result,
        )