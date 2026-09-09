from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

Mode = Literal["probs", "expval", "state", "density"]


@dataclass
class ProfilingConfig:
    """Configuration for JAQSI JAX/Perfetto profiling.

    Attributes
    ----------
    circuit_family:
        Circuit family from :mod:`benchmark.circuits` to profile.
    n_layers:
        Depth of that circuit.
    qubit_counts:
        List of qubit counts to profile.
    modes:
        Measurement modes to profile.  Supported values mirror the
        benchmark suite: ``"probs"``, ``"expval"``, ``"state"``,
        ``"density"``.
    batch_size:
        Batch size for each profiled execution.
    warmup_runs:
        Number of warm-up executions before the profiled region.
        Warm-up ensures JAX has completed tracing / JIT compilation so
        the captured trace only contains execution time.
    profile_runs:
        Number of simulation runs captured inside the profiled region.
    output_dir:
        Directory where Perfetto trace files are written.
    seed:
        Random seed for reproducible parameter generation.
    """

    circuit_family: str = "hea"
    n_layers: int = 1
    qubit_counts: List[int] = field(default_factory=lambda: [2, 4, 8, 12, 16])
    modes: List[Mode] = field(
        default_factory=lambda: ["probs", "expval", "state", "density"]
    )
    batch_size: int = 1
    warmup_runs: int = 2
    profile_runs: int = 3
    output_dir: str = "profiling_results"
    seed: int = 1000