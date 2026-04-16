"""Simulator adapters.

Concrete simulators are imported lazily to avoid hard failures when
optional backends (e.g. ``qml_essentials``) are not installed.
"""

__all__ = ["YaqsiBenchmark", "PennylaneBenchmark"]


def __getattr__(name: str):
    if name == "YaqsiBenchmark":
        from benchmark.simulators.yaqsi_sim import YaqsiBenchmark
        return YaqsiBenchmark
    if name == "PennylaneBenchmark":
        from benchmark.simulators.pennylane_sim import PennylaneBenchmark
        return PennylaneBenchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")