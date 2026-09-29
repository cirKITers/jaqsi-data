from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

Mode = Literal["probs", "expval", "state", "density"]


@dataclass
class ProfilingConfig:
    """Settings for JAQSI JAX/Perfetto profiling."""

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
