"""Record the installed environment and reject incompatible CSV resumes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax

from benchmark.config import BenchmarkConfig


def _source_identity(package: str) -> dict:
    """Identify the imported source, including uncommitted local edits."""
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.origin is not None
    root = Path(spec.origin).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"path": str(root), "sha256": digest.hexdigest()}


def _environment() -> dict:
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        entry = {"version": dist.version}
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            entry["direct_url"] = json.loads(direct_url)
        packages[name] = entry
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = platform.processor()
    if cpu_info.exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu_info.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_model,
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": jax.config.read("jax_enable_x64"),
        "packages": packages,
        "sources": {name: _source_identity(name) for name in ("benchmark", "jaqsi")},
    }


def record_provenance(
    csv_path: Path, cfg: BenchmarkConfig, *, has_results: bool
) -> None:
    """Write a JSON sidecar; settings absent from CSV keys must match on resume.

    Sweep axes (circuit, depth, width, batch, threads, modes and simulators) may
    change. Every invocation's resolved configuration and CPU settings are kept.
    Legacy CSVs remain readable but cannot be resumed without provenance.
    """
    settings = {
        name: getattr(cfg, name)
        for name in (
            "seed",
            "warmup",
            "precision",
            "optimal_config",
            "closed_form",
            "depolarizing",
        )
    }
    settings["n_iters"] = cfg.execution.n_iters
    identity = {
        "schema_version": 1,
        "settings": settings,
        "environment": _environment(),
    }
    path = csv_path.with_suffix(".json")
    metadata: dict[str, Any]
    if path.exists():
        metadata = json.loads(path.read_text())
        for name, value in identity.items():
            if metadata.get(name) != value:
                raise ValueError(
                    f"Cannot resume {csv_path}: {name} changed. "
                    "Use a new output.identifier."
                )
    else:
        if has_results:
            raise ValueError(
                f"Cannot resume {csv_path}: provenance is missing. "
                "Use a new output.identifier; the existing CSV can still be plotted."
            )
        metadata = dict[str, Any](identity, runs=[])

    run = {
        "config": asdict(cfg),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "QULACS_NUM_THREADS",
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "JAX_NUM_CPU_DEVICES",
            )
        },
    }
    for previous in metadata["runs"]:
        if previous["config"]["threads"] == cfg.threads and any(
            previous[name] != run[name]
            for name in ("cpu_affinity", "thread_environment")
        ):
            raise ValueError(
                f"Cannot resume {csv_path}: CPU affinity or thread environment changed "
                "for this thread count. Use a new output.identifier."
            )
    if run not in metadata["runs"]:
        metadata["runs"].append(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
