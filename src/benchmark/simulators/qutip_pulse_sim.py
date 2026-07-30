"""QuTiP pulse-level simulator benchmark adapter.

Integrates the pulse schedule of :mod:`benchmark.simulators.pulse_model`
segment by segment with ``qutip.sesolve``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import jax.numpy as jnp

import qutip

from benchmark.simulators.base import SimulatorBenchmark, Mode
from benchmark.simulators.pulse_model import (
    Segment,
    build_schedule,
    embed,
    make_coeff_fn,
    project_state,
)

# Solver tolerances an order of magnitude below jaqsi's, so the comparison is
# limited by the reference rather than by this adapter.
_OPTIONS = {"atol": 1.0e-12, "rtol": 1.0e-12, "normalize_output": False}


class QutipPulseBenchmark(SimulatorBenchmark):
    name = "qutip_pulse"

    def __init__(self) -> None:
        self._segments: List[Tuple[qutip.Qobj, Segment]] = []
        self._psi0: qutip.Qobj | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, n_qubits: int, mode: Mode, *, optimal_config: bool = False) -> None:
        self._n_qubits = n_qubits
        self._mode = mode

        # Embed every segment operator into the full register once, outside the
        # timing loop.  QuTiP's tensor order matches the big-endian convention
        # of the pulse model, so no basis permutation is needed.  Each segment
        # acts on at most two wires, so the embedded operator is stored sparsely.
        dims = [[2] * n_qubits, [2] * n_qubits]
        self._segments = [
            (qutip.Qobj(embed(seg.op, seg.wires, n_qubits), dims=dims).to("CSR"), seg)
            for seg in build_schedule(n_qubits)
        ]
        self._psi0 = qutip.basis([2] * n_qubits, [0] * n_qubits)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _solve(self, phi: float) -> np.ndarray:
        """Evolve $\\lvert 0 \\dots 0 \\rangle$ through the full pulse schedule."""
        psi = self._psi0
        for op, seg in self._segments:
            if seg.drag is None:
                hamiltonian = float(seg.angle_fn(phi)) * op
            else:
                hamiltonian = qutip.QobjEvo([[op, make_coeff_fn(seg, phi)]])
            psi = qutip.sesolve(
                hamiltonian, psi, [0.0, seg.duration], options=_OPTIONS
            ).states[-1]
        return psi.full().ravel()

    def _execute(self, phi_batch: jnp.ndarray) -> jnp.ndarray:
        results = [
            project_state(self._solve(float(phi_val)), self._mode, self._n_qubits)
            for phi_val in np.asarray(phi_batch)
        ]
        return jnp.array(np.stack(results))

    def warmup(self, phi: jnp.ndarray) -> jnp.ndarray:
        return self._execute(phi)

    def run(self, phi: jnp.ndarray) -> jnp.ndarray:
        return self._execute(phi)
