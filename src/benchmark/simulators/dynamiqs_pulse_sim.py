"""Benchmark the shared pulse schedule with Dynamiqs.

The default path evolves the full state with ``sesolve``. With
``optimal_config``, ``sepropagator`` evolves each local segment before
contracting it into the state. Both use jaqsi's Dormand-Prince 8(7)
method.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

import dynamiqs as dq

from benchmark.circuits import PULSE_FAMILIES, CircuitSpec
from benchmark.simulators.base import SimulatorBenchmark, Mode
from benchmark.simulators.pulse_model import (
    Channel,
    apply_local,
    apply_local_density,
    build_schedule,
    depolarize,
    embed,
    make_coeff_fns,
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
    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        # ``pulse_model`` only transcribes the Hadamard and $CRX$ pulse
        # decompositions, and the pulse level measures forward simulation only.
        return spec.family in PULSE_FAMILIES and mode != "grad"

    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        n_qubits = spec.n_qubits
        self._n_qubits = n_qubits
        self._mode = mode

        # Dynamiqs defaults to single precision, which is far above the
        # tolerance the pulse results are compared at.
        dq.set_precision("double")

        # Build every segment operator once, outside the timing loop.  The
        # tensor order matches the big-endian convention of the pulse model, so
        # no basis permutation is needed.  Channels carry no operator; the
        # density-matrix solvers apply them.
        segments = build_schedule(spec, self.envelope)
        if optimal_config:
            ops = [
                None
                if isinstance(seg, Channel)
                else tuple(dq.asqarray(jnp.asarray(op)) for op in seg.ops)
                for seg in segments
            ]
        else:
            # Every embedded operator has at most two non-zero diagonals, hence
            # the dia layout.
            ops = [
                None
                if isinstance(seg, Channel)
                else tuple(
                    dq.asqarray(
                        jnp.asarray(embed(op, seg.wires, n_qubits, xp=jnp)),
                        dims=(2,) * n_qubits,
                        layout=dq.dia,
                    )
                    for op in seg.ops
                )
                for seg in segments
            ]
        psi0 = dq.basis([2] * n_qubits, [0] * n_qubits)

        def hamiltonian(seg_ops, seg, params: jnp.ndarray):
            """Return the segment Hamiltonian $\\sum_k c_k(t) H_k$ at *params*."""
            if seg.envelope is None:
                return seg.angle_fn(params) * seg_ops[0]
            terms = [
                dq.modulated(coeff, op)
                for op, coeff in zip(seg_ops, make_coeff_fns(seg, params, xp=jnp))
            ]
            return sum(terms[1:], terms[0])

        def solve(params: jnp.ndarray) -> jnp.ndarray:
            """Evolve $\\lvert 0 \\dots 0 \\rangle$ through the pulse schedule."""
            state = psi0
            for op, seg in zip(ops, segments):
                state = dq.sesolve(
                    hamiltonian(op, seg, params),
                    state,
                    jnp.array([0.0, seg.duration]),
                    method=_METHOD,
                    progress_meter=False,
                ).states[-1]
            return project_state(state.to_jax().ravel(), mode, n_qubits, jnp)

        def solve_local(params: jnp.ndarray) -> jnp.ndarray:
            """Evolve $\\lvert 0 \\dots 0 \\rangle$ one local propagator at a time."""
            state = psi0.to_jax().ravel()
            for op, seg in zip(ops, segments):
                propagator = dq.sepropagator(
                    hamiltonian(op, seg, params),
                    jnp.array([0.0, seg.duration]),
                    method=_METHOD,
                    progress_meter=False,
                ).propagators[-1]
                state = apply_local(
                    propagator.to_jax(), state, seg.wires, n_qubits, jnp
                )
            return project_state(state, mode, n_qubits, jnp)

        def solve_density(params: jnp.ndarray) -> jnp.ndarray:
            """Evolve the noisy density matrix with ``mesolve``.

            Noise comes from discrete schedule channels, so no jump operators are
            passed. Use the vectorized form because Dynamiqs fails on an empty jump
            sum; restore qubit dimensions before each subsequent solve.
            """
            rho = dq.todm(psi0).to_jax()
            for op, seg in zip(ops, segments):
                if isinstance(seg, Channel):
                    rho = depolarize(rho, seg.p, seg.wire, n_qubits, jnp)
                    continue
                rho = dq.mesolve(
                    hamiltonian(op, seg, params),
                    [],
                    dq.asqarray(rho, dims=(2,) * n_qubits),
                    jnp.array([0.0, seg.duration]),
                    method=_METHOD,
                    progress_meter=False,
                    vectorized=True,
                ).states[-1].to_jax()
            return rho

        def solve_density_local(params: jnp.ndarray) -> jnp.ndarray:
            """Evolve the density matrix one local propagator at a time."""
            rho = dq.todm(psi0).to_jax()
            for op, seg in zip(ops, segments):
                if isinstance(seg, Channel):
                    rho = depolarize(rho, seg.p, seg.wire, n_qubits, jnp)
                    continue
                propagator = dq.sepropagator(
                    hamiltonian(op, seg, params),
                    jnp.array([0.0, seg.duration]),
                    method=_METHOD,
                    progress_meter=False,
                ).propagators[-1]
                rho = apply_local_density(
                    propagator.to_jax(), rho, seg.wires, n_qubits, jnp
                )
            return rho

        if mode == "noise":
            solver = solve_density_local if optimal_config else solve_density
        else:
            solver = solve_local if optimal_config else solve

        # Compile the whole schedule once and vectorize over the batch, mirroring
        # how jaqsi executes its pulse circuits.
        self._run_fn = jax.jit(jax.vmap(solver))

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs)

    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs)
