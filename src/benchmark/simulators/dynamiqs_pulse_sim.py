"""Dynamiqs pulse-level simulator benchmark adapter.

Integrates the pulse schedule of :mod:`benchmark.simulators.pulse_model`
segment by segment with ``dynamiqs.sesolve``, using the same Dormand-Prince
8(7) method as jaqsi.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

import dynamiqs as dq

from benchmark.simulators.base import SimulatorBenchmark, Mode
from benchmark.simulators.pulse_model import (
    build_schedule,
    embed,
    make_coeff_fn,
    project_state,
)

# Solver tolerances an order of magnitude below jaqsi's, so the comparison is
# limited by the reference rather than by this adapter.
_METHOD = dq.method.Dopri8(atol=1.0e-12, rtol=1.0e-12)


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

        # Embed every segment operator into the full register once, outside the
        # timing loop.  The tensor order matches the big-endian convention of
        # the pulse model, so no basis permutation is needed.  Every embedded
        # operator has at most two non-zero diagonals, hence the dia layout.
        segments = build_schedule(n_qubits)
        ops = [
            dq.asqarray(
                jnp.asarray(embed(seg.op, seg.wires, n_qubits, xp=jnp)),
                dims=(2,) * n_qubits,
                layout=dq.dia,
            )
            for seg in segments
        ]
        psi0 = dq.basis([2] * n_qubits, [0] * n_qubits)

        def solve(phi: jnp.ndarray) -> jnp.ndarray:
            """Evolve $\\lvert 0 \\dots 0 \\rangle$ through the pulse schedule."""
            state = psi0
            for op, seg in zip(ops, segments):
                if seg.drag is None:
                    hamiltonian = seg.angle_fn(phi) * op
                else:
                    hamiltonian = dq.modulated(make_coeff_fn(seg, phi, xp=jnp), op)
                state = dq.sesolve(
                    hamiltonian,
                    state,
                    jnp.array([0.0, seg.duration]),
                    method=_METHOD,
                    progress_meter=False,
                ).states[-1]
            return project_state(state.to_jax().ravel(), mode, n_qubits, jnp)

        # Compile the whole schedule once and vectorize over the batch, mirroring
        # how jaqsi executes its pulse circuits.
        self._run_fn = jax.jit(jax.vmap(solve))

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(phi)

    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(phi)
