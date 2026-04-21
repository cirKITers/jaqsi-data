"""Configuration loader for benchmark runs using OmegaConf."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf, DictConfig, MISSING


@dataclass
class QubitsConfig:
    min: int = 2
    max: int = 16


@dataclass
class ExecutionConfig:
    n_iters: int = 50
    batch_size: int = 1


@dataclass
class OutputConfig:
    dir: str = "benchmarking_results"
    identifier: Optional[str] = None


# All available simulator names (used for validation)
ALL_SIMULATORS: List[str] = ["yaqsi", "pennylane", "qiskit", "qibo", "qulacs"]


@dataclass
class BenchmarkConfig:
    seed: int = 1000
    warmup: bool = True
    precision: float = 1.0e-8
    qubits: QubitsConfig = field(default_factory=QubitsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    modes: List[str] = field(default_factory=lambda: ["probs", "expval", "state", "density"])
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

    return cfg