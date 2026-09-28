"""Configuration loader for benchmark runs using OmegaConf."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf, DictConfig

from benchmark.circuits import FAMILIES

logger = logging.getLogger(__name__)


@dataclass
class QubitsConfig:
    min: int = 2
    max: int = 10


@dataclass
class CircuitConfig:
    """Which circuit to benchmark and at which depths.

    ``family`` selects a circuit from :mod:`benchmark.circuits`; ``layers``
    lists the depths to sweep, so a run can vary depth at fixed width and
    width at fixed depth from the same config.
    """

    family: str = "hea"
    layers: List[int] = field(default_factory=lambda: [1])


@dataclass
class ExecutionConfig:
    n_iters: int = 50
    batch_size: int = 1


@dataclass
class OutputConfig:
    dir: str = "benchmarking_results"
    identifier: Optional[str] = None


# All available simulator names (used for validation).  Names ending in
# ``_pulse`` run the circuit at pulse level and are validated against
# ``jaqsi_pulse`` rather than the gate-level reference.
ALL_SIMULATORS: List[str] = [
    "jaqsi",
    "pennylane",
    "pennylane_adjoint",
    "pennylane_psr",
    "qiskit",
    "qibo",
    "qulacs",
    "jaqsi_pulse",
    "pennylane_pulse",
    "qutip_pulse",
    "dynamiqs_pulse",
]


@dataclass
class BenchmarkConfig:
    seed: int = 1000
    warmup: bool = True
    # Threads every simulator is pinned to.  One is the single-thread regime
    # the Yao, Qulacs and JuliVQC benchmarks report; raise it for the
    # multi-threaded regime and report the two separately.
    threads: int = 1
    # Probability of the single-qubit depolarizing channel the ``noise`` mode
    # applies after every gate, on each wire the gate acts on.
    depolarizing: float = 0.01
    precision: float = 1.0e-8
    optimal_config: bool = False
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    qubits: QubitsConfig = field(default_factory=QubitsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    modes: List[str] = field(
        default_factory=lambda: ["expval", "state", "density", "grad"]
    )
    simulators: List[str] = field(default_factory=lambda: list(ALL_SIMULATORS))
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> BenchmarkConfig:
    """Load and merge benchmark configuration.

    Resolution order (later wins):
        1. Structured defaults from ``BenchmarkConfig``
        2. YAML file (if *config_path* given)
        3. CLI-style dot-list overrides (e.g. ``["qubits.max=10", "seed=42"]``)

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file.  When *None* the built-in
        ``configs/default.yaml`` shipped with the package is used.
    overrides:
        Optional list of dot-list overrides applied on top of the loaded
        config.  Useful for quick command-line tweaks.

    Returns
    -------
    BenchmarkConfig
        Fully resolved, typed configuration object.
    """
    # 1. Structured defaults
    schema = OmegaConf.structured(BenchmarkConfig)

    # 2. YAML file
    if config_path is None:
        config_path = str(
            Path(__file__).parent / "configs" / "default.yaml"
        )
    if os.path.isfile(config_path):
        file_cfg = OmegaConf.load(config_path)
    else:
        file_cfg = OmegaConf.create()

    # 3. CLI overrides
    cli_cfg = OmegaConf.from_dotlist(overrides or [])

    # Merge
    merged: DictConfig = OmegaConf.merge(schema, file_cfg, cli_cfg)

    # Resolve interpolations and convert to the dataclass
    cfg: BenchmarkConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]

    # Generate identifier if not provided
    if cfg.output.identifier is None:
        cfg.output.identifier = datetime.now().strftime("%Y%m%d%H%M%S")

    # Validate simulator names
    unknown = set(cfg.simulators) - set(ALL_SIMULATORS)
    if unknown:
        raise ValueError(
            f"Unknown simulator(s): {sorted(unknown)}. "
            f"Available: {ALL_SIMULATORS}"
        )

    # Validate circuit family and depths
    if cfg.circuit.family not in FAMILIES:
        raise ValueError(
            f"Unknown circuit family: {cfg.circuit.family!r}. "
            f"Available: {sorted(FAMILIES)}"
        )
    if not cfg.circuit.layers or any(n < 1 for n in cfg.circuit.layers):
        raise ValueError(
            f"circuit.layers must be a non-empty list of positive integers, "
            f"got {cfg.circuit.layers}"
        )

    if cfg.threads < 1:
        raise ValueError(f"threads must be at least 1, got {cfg.threads}")
    if cfg.execution.batch_size % cfg.threads:
        logger.warning(
            f"batch_size {cfg.execution.batch_size} is not a multiple of threads "
            f"{cfg.threads}: jaqsi runs the batch on one CPU device instead of "
            f"splitting it over {cfg.threads}."
        )

    # At zero the noise mode has no channels, and jaqsi would take the
    # state-vector route that mode exists to rule out.
    if not 0.0 < cfg.depolarizing <= 1.0:
        raise ValueError(
            f"depolarizing must be in (0, 1], got {cfg.depolarizing}"
        )

    return cfg