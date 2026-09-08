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

from benchmark.circuits import build_spec
from benchmark.simulators.jaqsi_sim import JaqsiBenchmark
from benchmark.simulators.jaqsi_pulse_sim import JaqsiPulseBenchmark
from benchmark.simulators.pulse_model import (
    apply_local,
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


def _spec(n_qubits: int, n_layers: int = 1):
    """Return the pulse-capable circuit spec at the given size."""
    return build_spec("crx_ring", n_qubits, n_layers)


def _inputs(spec) -> np.ndarray:
    """Return one parameter vector, every CRX angle set to ``PHI``."""
    return np.full(spec.n_inputs, PHI)


def _scipy_state(n_qubits: int, phi: float) -> np.ndarray:
    """Integrate the pulse schedule with scipy, independently of jax."""
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0

    spec = _spec(n_qubits)
    params = np.full(spec.n_inputs, phi)
    for segment in build_schedule(spec):
        op = embed(segment.op, segment.wires, n_qubits)
        coeff = make_coeff_fn(segment, params)
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
        assert len(build_schedule(_spec(n_qubits))) == 21 * n_qubits

    @pytest.mark.parametrize("n_layers", [1, 2])
    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_segment_count_scales_with_depth(self, n_qubits, n_layers):
        """Each extra CRX ring adds another 18 segments per qubit."""
        schedule = build_schedule(_spec(n_qubits, n_layers))
        assert len(schedule) == 3 * n_qubits + 18 * n_qubits * n_layers

    def test_unsupported_gate_raises(self):
        """Only the Hadamard and CRX decompositions are transcribed."""
        with pytest.raises(ValueError, match="no pulse transcription"):
            build_schedule(build_spec("hea", 2, 1))

    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_embedding_is_hermitian(self, n_qubits):
        for segment in build_schedule(_spec(n_qubits)):
            op = embed(segment.op, segment.wires, n_qubits)
            assert op.shape == (2**n_qubits, 2**n_qubits)
            np.testing.assert_allclose(op, op.conj().T, atol=1e-12)

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_apply_local_matches_embedding(self, n_qubits):
        """Contracting a local operator equals applying its embedded form.

        The optimized adapters integrate local propagators and contract them
        into the statevector, so the two routes have to agree exactly.
        """
        rng = np.random.default_rng(n_qubits)
        psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)

        for segment in build_schedule(_spec(n_qubits)):
            local = rng.normal(size=segment.op.shape) + 1j * rng.normal(
                size=segment.op.shape
            )
            np.testing.assert_allclose(
                apply_local(local, psi, segment.wires, n_qubits),
                embed(local, segment.wires, n_qubits) @ psi,
                atol=1e-12,
                err_msg=f"local application mismatch on wires {segment.wires}",
            )


class TestTranscription:
    """The schedule reproduces what jaqsi's pulse gates actually execute."""

    def test_scipy_matches_jaqsi_pulse(self):
        n_qubits = 2
        sim = JaqsiPulseBenchmark()
        spec = _spec(n_qubits)
        sim.setup(spec, "state")
        jaqsi_psi = np.asarray(sim.run(jnp.array([_inputs(spec)]), jnp.zeros(0)))[0]

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
            spec = _spec(n_qubits)
            sim.setup(spec, mode)
            expected = np.asarray(
                sim.run(jnp.array([_inputs(spec)]), jnp.zeros(0))
            )[0]
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
        spec = _spec(n_qubits)
        pulse.setup(spec, "expval")
        gate = JaqsiBenchmark()
        gate.setup(spec, "expval")

        inputs = jnp.array([_inputs(spec)])
        weights = jnp.zeros(0)
        np.testing.assert_allclose(
            np.asarray(pulse.run(inputs, weights)),
            np.asarray(gate.run(inputs, weights)),
            atol=SOLVER_PRECISION,
            err_msg="pulse circuit does not reproduce the gate-level circuit",
        )
