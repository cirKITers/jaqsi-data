"""Dynamiqs pulse-level simulator benchmark adapter.

Integrates the pulse schedule of :mod:`benchmark.simulators.pulse_model`
segment by segment, using the same Dormand-Prince 8(7) method as jaqsi.  The
default path evolves the statevector of the whole register with
``dynamiqs.sesolve``, the idiomatic formulation.  Under ``optimal_config`` each
segment propagator is integrated on its own two- or four-dimensional space with
``dynamiqs.sepropagator`` and contracted into the statevector instead, matching
how jaqsi and the optimized PennyLane path compose their pulse gates.  Both
integrate identical ODEs; only their dimension differs.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

import dynamiqs as dq

from benchmark.simulators.base import SimulatorBenchmark, Mode
from benchmark.simulators.pulse_model import (
    apply_local,
    build_schedule,
    embed,
    make_coeff_fn,
    project_state,
)

# Solver tolerances matching jaqsi's, so that no adapter integrates to a
# stricter accuracy target than the others.
_METHOD = dq.method.Dopri8(atol=1.0e-10, rtol=1.0e-10)


class DynamiqsPulseBenchmark(SimulatorBenchmark):
    name = "dynamiqs_pulse"

    def __init__(self) -> None:
        self._run_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, n_qubits: int, mode: Mode, *, optimal_config: bool = False) -> None:
        self._n_qubits = n_qubits
        self._mode = mode

        # Dynamiqs defaults to single precision, which is far above the
        # tolerance the pulse results are compared at.
        dq.set_precision("double")

        # Build every segment operator once, outside the timing loop.  The
        # tensor order matches the big-endian convention of the pulse model, so
        # no basis permutation is needed.
        segments = build_schedule(n_qubits)
        if optimal_config:
            ops = [dq.asqarray(jnp.asarray(seg.op)) for seg in segments]
        else:
            # Every embedded operator has at most two non-zero diagonals, hence
            # the dia layout.
            ops = [
                dq.asqarray(
                    jnp.asarray(embed(seg.op, seg.wires, n_qubits, xp=jnp)),
                    dims=(2,) * n_qubits,
                    layout=dq.dia,
                )
                for seg in segments
            ]
        psi0 = dq.basis([2] * n_qubits, [0] * n_qubits)

        def hamiltonian(op, seg, phi: jnp.ndarray):
            """Return the segment Hamiltonian ``c(t) * op`` at *phi*."""
            if seg.drag is None:
                return seg.angle_fn(phi) * op
            return dq.modulated(make_coeff_fn(seg, phi, xp=jnp), op)

        def solve(phi: jnp.ndarray) -> jnp.ndarray:
            """Evolve $\\lvert 0 \\dots 0 \\rangle$ through the pulse schedule."""
            state = psi0
            for op, seg in zip(ops, segments):
                state = dq.sesolve(
                    hamiltonian(op, seg, phi),
                    state,
                    jnp.array([0.0, seg.duration]),
                    method=_METHOD,
                    progress_meter=False,
                ).states[-1]
            return project_state(state.to_jax().ravel(), mode, n_qubits, jnp)

        def solve_local(phi: jnp.ndarray) -> jnp.ndarray:
            """Evolve $\\lvert 0 \\dots 0 \\rangle$ one local propagator at a time."""
            state = psi0.to_jax().ravel()
            for op, seg in zip(ops, segments):
                propagator = dq.sepropagator(
                    hamiltonian(op, seg, phi),
                    jnp.array([0.0, seg.duration]),
                    method=_METHOD,
                    progress_meter=False,
                ).propagators[-1]
                state = apply_local(
                    propagator.to_jax(), state, seg.wires, n_qubits, jnp
                )
            return project_state(state, mode, n_qubits, jnp)

        # Compile the whole schedule once and vectorize over the batch, mirroring
        # how jaqsi executes its pulse circuits.
        self._run_fn = jax.jit(jax.vmap(solve_local if optimal_config else solve))

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(phi)

    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(phi)
