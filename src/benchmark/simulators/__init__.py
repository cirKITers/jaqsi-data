"""Load simulator adapters lazily to allow optional backends."""

import importlib

import jax

# JAX 0.11 removed ``jax.core.is_concrete``, which PennyLane 0.45.1 still calls
# on every traced tensor, so ``jax.jit`` and ``jax.grad`` through a QNode fail
# without it.  Restored here, where both PennyLane adapters pass on import;
# drop it once PennyLane no longer calls it.
if not hasattr(jax.core, "is_concrete"):
    jax.core.is_concrete = lambda tracer: tracer.to_concrete_value() is not None

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