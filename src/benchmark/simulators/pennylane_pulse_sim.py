"""PennyLane pulse-level simulator benchmark adapter.

Rebuilds the pulse schedule of :mod:`benchmark.simulators.pulse_model` on
``default.qubit`` in one of two configurations.  The default path expresses
each segment as a ``qml.pulse.ParametrizedEvolution``, the idiomatic pulse
API.  The optimized path integrates the same segments directly with
``jax.experimental.ode``, the solver ``ParametrizedEvolution`` itself builds
on, and applies the resulting unitaries as gates.  Both integrate identical
ODEs; the optimized path drops PennyLane's per-operation overhead, which
dominates the default path, at the cost of no longer exercising the pulse API.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import pennylane as qml

from benchmark.circuits import PULSE_FAMILIES, CircuitSpec
from benchmark.simulators.base import SimulatorBenchmark, Mode
from benchmark.simulators.pulse_model import (
    Segment,
    build_schedule,
    drag_env,
    make_coeff_fn,
)

# Tolerances for the jax.experimental.ode integrator, matching jaqsi's solver.
_ATOL = 1.0e-10
_RTOL = 1.0e-10


def _parametrized_hamiltonian(segment: Segment) -> qml.pulse.ParametrizedHamiltonian:
    """Return ``c(t) * op`` for *segment* with the scale factor as parameter.

    The coefficient takes the segment's scale factor as its single trainable
    parameter, so the same object serves every value of $\\phi$.
    """
    op = qml.Hermitian(segment.op, wires=segment.wires)
    if segment.drag is None:
        return (lambda p, t: p) * op

    drag = segment.drag
    return (lambda p, t: 0.5 * drag_env(t, drag, jnp) * p) * op


def _segment_unitary(segment: Segment, params) -> jnp.ndarray:
    """Solve $\\mathrm{d}U/\\mathrm{d}t = -i c(t) H U$ over one segment.

    The Hamiltonian acts on one or two wires, so only the local unitary is
    integrated rather than one over the full register.
    """
    op = jnp.asarray(segment.op)
    coeff = make_coeff_fn(segment, params, jnp)

    def rhs(u, t):
        return -1j * coeff(t) * (op @ u)

    return odeint(
        rhs,
        jnp.eye(op.shape[0], dtype=complex),
        jnp.array([0.0, segment.duration]),
        atol=_ATOL,
        rtol=_RTOL,
    )[-1]


class PennylanePulseBenchmark(SimulatorBenchmark):
    name = "pennylane_pulse"

    def __init__(self) -> None:
        self._circuit_fn: Callable | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0

    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        # ``pulse_model`` only transcribes the Hadamard and $CRX$ pulse
        # decompositions, and the pulse level measures noise-free forward
        # simulation only.
        return spec.family in PULSE_FAMILIES and mode not in ("grad", "noise")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        n_qubits = spec.n_qubits
        self._n_qubits = n_qubits
        self._mode = mode

        segments = build_schedule(spec)
        dev = qml.device("default.qubit", wires=n_qubits)

        return_map: dict[str, Callable] = {
            "density": lambda: qml.density_matrix(wires=range(n_qubits)),
            "state": lambda: qml.state(),
            "probs": lambda: qml.probs(wires=range(n_qubits)),
            "expval": lambda: [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)],
        }

        if optimal_config:
            # Integrate each segment with jax directly and apply the resulting
            # local unitary, bypassing ParametrizedEvolution's per-call cost.
            #
            # ``diff_method`` must not be None: it takes default.qubit off the
            # JAX-traced path, so every call round-trips through host numpy and
            # jax.jit has nothing to compile.  Measured here at 8x to 10x the
            # cost for bit-identical results.  Naming the backprop pipeline
            # keeps the circuit traceable; no gradient is ever taken.
            @qml.qnode(dev, interface="jax", diff_method="backprop")
            def circuit(params):
                for segment in segments:
                    qml.QubitUnitary(
                        _segment_unitary(segment, params), wires=segment.wires
                    )
                return return_map[mode]()

            self._circuit_fn = jax.jit(circuit)
        else:
            # ParametrizedHamiltonians have to be built outside the QNode:
            # creating one inline queues its bare operator as an additional gate.
            evolutions: List[
                Tuple[qml.pulse.ParametrizedHamiltonian, Callable, float]
            ] = [
                (_parametrized_hamiltonian(seg), seg.angle_fn, seg.duration)
                for seg in segments
            ]

            # ``backprop`` for the same reason as above: it is what keeps the
            # QNode on the JAX-traced path.  No gradient is taken here either.
            @qml.qnode(dev, interface="jax", diff_method="backprop")
            def circuit(params):
                for hamiltonian, angle_fn, duration in evolutions:
                    qml.evolve(hamiltonian)(
                        [angle_fn(params)], t=duration, atol=_ATOL, rtol=_RTOL
                    )
                return return_map[mode]()

            self._circuit_fn = circuit

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _execute(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Batching is a Python loop in both configurations: ParametrizedEvolution
        # rebuilds its Hamiltonian per call, which neither vmap nor jit can
        # share, and keeping the loop makes the two paths directly comparable.
        assert self._circuit_fn is not None
        results = [
            jnp.asarray(self._circuit_fn(jnp.asarray(sample)))
            for sample in np.asarray(inputs)
        ]
        return jnp.stack(results)

    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        return self._execute(inputs)

    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        return self._execute(inputs)
