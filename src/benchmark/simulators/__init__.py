"""Simulator adapters.


Concrete simulators are imported lazily to avoid hard failures when
optional backends (e.g. ``jaqsi``, ``qiskit``) are not installed.
"""

import importlib

# Maps the public adapter name to the submodule that defines it.
_MODULES = {
    "JaqsiBenchmark": "jaqsi_sim",
    "PennylaneBenchmark": "pennylane_sim",
    "QiskitBenchmark": "qiskit_sim",
    "QiboBenchmark": "qibo_sim",
    "QulacsBenchmark": "qulacs_sim",
    "JaqsiPulseBenchmark": "jaqsi_pulse_sim",
    "PennylanePulseBenchmark": "pennylane_pulse_sim",
    "QutipPulseBenchmark": "qutip_pulse_sim",
    "DynamiqsPulseBenchmark": "dynamiqs_pulse_sim",
}

__all__ = list(_MODULES)


def __getattr__(name: str):
    if name in _MODULES:
        module = importlib.import_module(f"benchmark.simulators.{_MODULES[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")