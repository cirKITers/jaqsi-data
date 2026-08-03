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

from typing import List, Tuple

import numpy as np
import jax.numpy as jnp

import qutip

from benchmark.simulators.base import SimulatorBenchmark, Mode
from benchmark.simulators.pulse_model import (
    Segment,
    apply_local,
    build_schedule,
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
        self._segments: List[Tuple[qutip.Qobj, Segment]] = []
        self._psi0: qutip.Qobj | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0
        self._optimal: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, n_qubits: int, mode: Mode, *, optimal_config: bool = False) -> None:
        self._n_qubits = n_qubits
        self._mode = mode
        self._optimal = optimal_config

        # Build every segment operator once, outside the timing loop.  QuTiP's
        # tensor order matches the big-endian convention of the pulse model, so
        # no basis permutation is needed.
        segments = build_schedule(n_qubits)
        if optimal_config:
            local_dims = [[2] * len(seg.wires) for seg in segments]
            self._segments = [
                (qutip.Qobj(seg.op, dims=[dim, dim]), seg)
                for seg, dim in zip(segments, local_dims)
            ]
        else:
            # Each segment acts on at most two wires, so the embedded operator
            # is stored sparsely.
            dims = [[2] * n_qubits, [2] * n_qubits]
            self._segments = [
                (qutip.Qobj(embed(seg.op, seg.wires, n_qubits), dims=dims).to("CSR"), seg)
                for seg in segments
            ]
        self._psi0 = qutip.basis([2] * n_qubits, [0] * n_qubits)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _hamiltonian(self, op: qutip.Qobj, seg: Segment, phi: float):
        """Return the segment Hamiltonian ``c(t) * op`` at *phi*."""
        if seg.drag is None:
            return float(seg.angle_fn(phi)) * op
        return qutip.QobjEvo([[op, make_coeff_fn(seg, phi)]])

    def _solve(self, phi: float) -> np.ndarray:
        """Evolve $\\lvert 0 \\dots 0 \\rangle$ through the full pulse schedule."""
        psi = self._psi0
        for op, seg in self._segments:
            psi = qutip.sesolve(
                self._hamiltonian(op, seg, phi),
                psi,
                [0.0, seg.duration],
                options=_OPTIONS,
            ).states[-1]
        return psi.full().ravel()

    def _solve_local(self, phi: float) -> np.ndarray:
        """Evolve $\\lvert 0 \\dots 0 \\rangle$ one local propagator at a time."""
        psi = self._psi0.full().ravel()
        for op, seg in self._segments:
            propagator = qutip.propagator(
                self._hamiltonian(op, seg, phi), seg.duration, options=_OPTIONS
            )
            psi = apply_local(propagator.full(), psi, seg.wires, self._n_qubits)
        return psi

    def _execute(self, phi_batch: jnp.ndarray) -> jnp.ndarray:
        solve = self._solve_local if self._optimal else self._solve
        results = [
            project_state(solve(float(phi_val)), self._mode, self._n_qubits)
            for phi_val in np.asarray(phi_batch)
        ]
        return jnp.array(np.stack(results))

    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        return self._execute(phi)

    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        return self._execute(phi)
