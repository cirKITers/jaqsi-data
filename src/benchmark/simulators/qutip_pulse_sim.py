"""QuTiP pulse-level simulator benchmark adapter.

Integrates the pulse schedule of :mod:`benchmark.simulators.pulse_model`
segment by segment with ``qutip.sesolve``.  The default path evolves the
statevector of the whole register under the embedded segment operator, the
idiomatic formulation.  Under ``optimal_config`` each segment propagator is
integrated on its own two- or four-dimensional space with ``qutip.propagator``
and contracted into the statevector instead, matching how jaqsi and the
optimized PennyLane path compose their pulse gates.  Both integrate identical
ODEs; only their dimension differs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp

import qutip

from benchmark.circuits import PULSE_FAMILIES, CircuitSpec
from benchmark.simulators.base import SimulatorBenchmark, Mode
from benchmark.simulators.pulse_model import (
    Channel,
    Segment,
    apply_local,
    apply_local_density,
    build_schedule,
    depolarize,
    embed,
    make_coeff_fn,
    project_state,
)

# Solver tolerances matching jaqsi's, so that no adapter integrates to a
# stricter accuracy target than the others.
_OPTIONS = {"atol": 1.0e-10, "rtol": 1.0e-10, "normalize_output": False}


class QutipPulseBenchmark(SimulatorBenchmark):
    name = "qutip_pulse"

    def __init__(self) -> None:
        self._segments: List[Tuple[Optional[qutip.Qobj], Union[Segment, Channel]]] = []
        self._psi0: qutip.Qobj | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0
        self._optimal: bool = False

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
        self._optimal = optimal_config

        # Build every segment operator once, outside the timing loop.  QuTiP's
        # tensor order matches the big-endian convention of the pulse model, so
        # no basis permutation is needed.  Channels carry no operator; the
        # density-matrix solvers apply them.
        segments = build_schedule(spec)
        if optimal_config:
            self._segments = [
                (None, seg)
                if isinstance(seg, Channel)
                else (qutip.Qobj(seg.op, dims=[[2] * len(seg.wires)] * 2), seg)
                for seg in segments
            ]
        else:
            # Each segment acts on at most two wires, so the embedded operator
            # is stored sparsely.
            dims = [[2] * n_qubits, [2] * n_qubits]
            self._segments = [
                (None, seg)
                if isinstance(seg, Channel)
                else (
                    qutip.Qobj(embed(seg.op, seg.wires, n_qubits), dims=dims).to("CSR"),
                    seg,
                )
                for seg in segments
            ]
        self._psi0 = qutip.basis([2] * n_qubits, [0] * n_qubits)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _hamiltonian(self, op: qutip.Qobj, seg: Segment, params: np.ndarray):
        """Return the segment Hamiltonian ``c(t) * op`` at *params*."""
        if seg.drag is None:
            return float(seg.angle_fn(params)) * op
        return qutip.QobjEvo([[op, make_coeff_fn(seg, params)]])

    def _solve(self, params: np.ndarray) -> np.ndarray:
        """Evolve $\\lvert 0 \\dots 0 \\rangle$ through the full pulse schedule."""
        psi = self._psi0
        for op, seg in self._segments:
            psi = qutip.sesolve(
                self._hamiltonian(op, seg, params),
                psi,
                [0.0, seg.duration],
                options=_OPTIONS,
            ).states[-1]
        return psi.full().ravel()

    def _solve_local(self, params: np.ndarray) -> np.ndarray:
        """Evolve $\\lvert 0 \\dots 0 \\rangle$ one local propagator at a time."""
        psi = self._psi0.full().ravel()
        for op, seg in self._segments:
            propagator = qutip.propagator(
                self._hamiltonian(op, seg, params), seg.duration, options=_OPTIONS
            )
            psi = apply_local(propagator.full(), psi, seg.wires, self._n_qubits)
        return psi

    def _solve_density(self, params: np.ndarray) -> np.ndarray:
        """Evolve $\\lvert 0 \\dots 0 \\rangle\\langle 0 \\dots 0 \\rvert$ with ``mesolve``.

        No collapse operators are passed: the noise is the schedule's discrete
        channels, applied between the solves.
        """
        rho = qutip.ket2dm(self._psi0)
        for op, seg in self._segments:
            if isinstance(seg, Channel):
                rho = qutip.Qobj(
                    depolarize(rho.full(), seg.p, seg.wire, self._n_qubits),
                    dims=rho.dims,
                )
                continue
            rho = qutip.mesolve(
                self._hamiltonian(op, seg, params),
                rho,
                [0.0, seg.duration],
                options=_OPTIONS,
            ).states[-1]
        return rho.full()

    def _solve_density_local(self, params: np.ndarray) -> np.ndarray:
        """Evolve the density matrix one local propagator at a time."""
        rho = qutip.ket2dm(self._psi0).full()
        for op, seg in self._segments:
            if isinstance(seg, Channel):
                rho = depolarize(rho, seg.p, seg.wire, self._n_qubits)
                continue
            propagator = qutip.propagator(
                self._hamiltonian(op, seg, params), seg.duration, options=_OPTIONS
            )
            rho = apply_local_density(propagator.full(), rho, seg.wires, self._n_qubits)
        return rho

    def _execute(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self._mode == "noise":
            # The density matrix is the result, so there is nothing to project.
            solve = self._solve_density_local if self._optimal else self._solve_density
            return jnp.array(np.stack([solve(sample) for sample in np.asarray(inputs)]))
        solve = self._solve_local if self._optimal else self._solve
        results = [
            project_state(solve(sample), self._mode, self._n_qubits)
            for sample in np.asarray(inputs)
        ]
        return jnp.array(np.stack(results))

    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        return self._execute(inputs)

    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        return self._execute(inputs)
