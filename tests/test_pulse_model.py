"""Tests for the transcribed pulse schedule.

The schedule in :mod:`benchmark.simulators.pulse_model` restates jaqsi's
pulse-level gate decomposition for the other simulators, so it is verified
two ways: against an independent scipy integration of the same segments, and
against the ideal gate-level circuit the pulses are calibrated to implement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from benchmark.simulators.jaqsi_sim import JaqsiBenchmark
from benchmark.simulators.jaqsi_pulse_sim import JaqsiPulseBenchmark
from benchmark.simulators.pulse_model import (
    build_schedule,
    embed,
    make_coeff_fn,
    project_state,
)

# Enable 64-bit precision for JAX (matches the benchmark runner)
jax.config.update("jax_enable_x64", True)

# Both integrators run near their own accuracy floor; the residual is the
# difference between two independent ODE solvers, not a modelling error.
SOLVER_PRECISION = 1e-5

PHI = 0.7123


def _scipy_state(n_qubits: int, phi: float) -> np.ndarray:
    """Integrate the pulse schedule with scipy, independently of jax."""
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0

    for segment in build_schedule(n_qubits):
        op = embed(segment.op, segment.wires, n_qubits)
        coeff = make_coeff_fn(segment, phi)
        solution = solve_ivp(
            lambda t, y: -1j * coeff(t) * (op @ y),
            (0.0, segment.duration),
            psi,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
        )
        psi = solution.y[:, -1]
    return psi


class TestSchedule:
    """Structural properties of the generated schedule."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_segment_count(self, n_qubits):
        # 3 segments per Hadamard and 18 per CRX, one of each per qubit.
        assert len(build_schedule(n_qubits)) == 21 * n_qubits

    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_embedding_is_hermitian(self, n_qubits):
        for segment in build_schedule(n_qubits):
            op = embed(segment.op, segment.wires, n_qubits)
            assert op.shape == (2**n_qubits, 2**n_qubits)
            np.testing.assert_allclose(op, op.conj().T, atol=1e-12)


class TestTranscription:
    """The schedule reproduces what jaqsi's pulse gates actually execute."""

    def test_scipy_matches_jaqsi_pulse(self):
        n_qubits = 2
        sim = JaqsiPulseBenchmark()
        sim.setup(n_qubits, "state")
        jaqsi_psi = np.asarray(sim.run(jnp.array([PHI])))[0]

        np.testing.assert_allclose(
            jaqsi_psi,
            _scipy_state(n_qubits, PHI),
            atol=SOLVER_PRECISION,
            err_msg="transcribed schedule diverges from jaqsi's pulse gates",
        )

    def test_project_state_matches_jaqsi_pulse(self):
        n_qubits = 2
        psi = _scipy_state(n_qubits, PHI)
        sim = JaqsiPulseBenchmark()

        for mode in ("probs", "expval", "density"):
            sim.setup(n_qubits, mode)
            expected = np.asarray(sim.run(jnp.array([PHI])))[0]
            np.testing.assert_allclose(
                project_state(psi, mode, n_qubits),
                expected,
                atol=SOLVER_PRECISION,
                err_msg=f"projection mismatch for mode {mode}",
            )


class TestGateEquivalence:
    """The pulses implement the gate-level circuit they are calibrated for."""

    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_expval_approximates_gate_level(self, n_qubits):
        pulse = JaqsiPulseBenchmark()
        pulse.setup(n_qubits, "expval")
        gate = JaqsiBenchmark()
        gate.setup(n_qubits, "expval")

        batch = jnp.array([PHI])
        np.testing.assert_allclose(
            np.asarray(pulse.run(batch)),
            np.asarray(gate.run(batch)),
            atol=SOLVER_PRECISION,
            err_msg="pulse circuit does not reproduce the gate-level circuit",
        )
