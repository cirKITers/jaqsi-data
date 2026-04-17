"""Simulator adapters.


Concrete simulators are imported lazily to avoid hard failures when
optional backends (e.g. ``qml_essentials``, ``qiskit``) are not installed.
"""

__all__ = ["YaqsiBenchmark", "PennylaneBenchmark", "QiskitBenchmark", "QiboBenchmark"]


def __getattr__(name: str):
    if name == "YaqsiBenchmark":
        from benchmark.simulators.yaqsi_sim import YaqsiBenchmark
        return YaqsiBenchmark
    if name == "PennylaneBenchmark":
        from benchmark.simulators.pennylane_sim import PennylaneBenchmark
        return PennylaneBenchmark
    if name == "QiskitBenchmark":
        from benchmark.simulators.qiskit_sim import QiskitBenchmark
        return QiskitBenchmark
    if name == "QiboBenchmark":
        from benchmark.simulators.qibo_sim import QiboBenchmark
        return QiboBenchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")