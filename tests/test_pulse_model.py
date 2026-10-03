"""Test the shared pulse schedule against independent integrations.

Compare the transcription with SciPy integration and with the calibrated
gate-level circuit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

# Enable 64-bit precision for JAX (matches the benchmark runner).  It has to
# precede the adapter imports: jaqsi fixes the dtype of its constant gate
# matrices when it is imported.
jax.config.update("jax_enable_x64", True)

from benchmark.circuits import build_spec
from benchmark.simulators.jaqsi_sim import JaqsiBenchmark
from benchmark.simulators.jaqsi_pulse_sim import JaqsiPulseBenchmark
from benchmark.simulators.pulse_model import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    Channel,
    apply_local,
    apply_local_density,
    build_schedule,
    depolarize,
    embed,
    in_phase_env,
    make_coeff_fns,
    project_state,
    quadrature_env,
    ry_envelope,
)
from benchmark.simulators import pulse_model

# Both integrators run near their own accuracy floor; the residual is the
# difference between two independent ODE solvers, not a modelling error.
SOLVER_PRECISION = 1e-5

PHI = 0.7123


def _spec(n_qubits: int, n_layers: int = 1):
    """Return the pulse-capable circuit spec at the given size."""
    return build_spec("crx_ring", n_qubits, n_layers)


def _inputs(spec) -> np.ndarray:
    """Set every CRX angle in one input vector to ``PHI``."""
    return np.full(spec.n_inputs, PHI)


def _scipy_state(n_qubits: int, phi: float, envelope: str = "gaussian") -> np.ndarray:
    """Integrate the pulse schedule independently with SciPy."""
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0

    spec = _spec(n_qubits)
    params = np.full(spec.n_inputs, phi)
    for segment in build_schedule(spec, envelope):
        ops = [embed(op, segment.wires, n_qubits) for op in segment.ops]
        coeffs = make_coeff_fns(segment, params)
        solution = solve_ivp(
            lambda t, y: -1j * sum(c(t) * (op @ y) for c, op in zip(coeffs, ops)),
            (0.0, segment.duration),
            psi,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
        )
        psi = solution.y[:, -1]
    return psi


class TestSchedule:
    """Check the pulse schedule structure."""

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

    def test_unsupported_envelope_raises(self):
        with pytest.raises(ValueError, match="no pulse transcription"):
            build_schedule(_spec(2), "square")

    @pytest.mark.parametrize("envelope", ["gaussian", "drag"])
    def test_only_drag_drives_a_quadrature(self, envelope):
        """DRAG adds a $-X$ term to every RY, without changing the count."""
        schedule = build_schedule(_spec(3), envelope)
        assert len(schedule) == 21 * 3
        # One RY per Hadamard, six per CRX.
        driven = [s for s in schedule if s.envelope is not None]
        assert len(driven) == 7 * 3
        for segment in driven:
            np.testing.assert_array_equal(segment.op, PAULI_Y)
            if envelope == "drag":
                np.testing.assert_array_equal(segment.quad_op, -PAULI_X)
                assert len(make_coeff_fns(segment, _inputs(_spec(3)))) == 2
            else:
                assert segment.quad_op is None

    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_embedding_is_hermitian(self, n_qubits):
        for segment in build_schedule(_spec(n_qubits)):
            op = embed(segment.op, segment.wires, n_qubits)
            assert op.shape == (2**n_qubits, 2**n_qubits)
            np.testing.assert_allclose(op, op.conj().T, atol=1e-12)

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_apply_local_matches_embedding(self, n_qubits):
        """Match local tensor contraction to full operator embedding.

        Optimized adapters use local propagators, so both routes must agree.
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


class TestNoise:
    """Check noise channel placement and density matrix evolution."""

    def test_channels_follow_their_gate(self):
        """Place each channel after its gate's final pulse segment.

        Hadamard uses 3 segments and CRX uses 18.
        """
        clean = build_schedule(_spec(2))
        noisy = build_schedule(build_spec("crx_ring", 2, 1, depolarizing=0.1))

        positions = [i for i, item in enumerate(noisy) if isinstance(item, Channel)]
        assert positions == [3, 7, 26, 27, 46, 47]
        assert [(noisy[i].wire, noisy[i].p) for i in positions] == [
            (0, 0.1), (1, 0.1), (0, 0.1), (1, 0.1), (1, 0.1), (0, 0.1)
        ]
        segments = [item for item in noisy if not isinstance(item, Channel)]
        assert [(s.wires, s.duration) for s in segments] == [
            (s.wires, s.duration) for s in clean
        ]

    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_apply_local_density_matches_embedding(self, n_qubits):
        rng = np.random.default_rng(n_qubits)
        dim = 2**n_qubits
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        for segment in build_schedule(_spec(n_qubits)):
            local = rng.normal(size=segment.op.shape) + 1j * rng.normal(
                size=segment.op.shape
            )
            full = embed(local, segment.wires, n_qubits)
            np.testing.assert_allclose(
                apply_local_density(local, rho, segment.wires, n_qubits),
                full @ rho @ full.conj().T,
                atol=1e-12,
                err_msg=f"local density application mismatch on wires {segment.wires}",
            )

    @pytest.mark.parametrize("wire", [0, 1, 2])
    def test_depolarize_matches_the_kraus_sum(self, wire):
        n_qubits, p = 3, 0.2
        rng = np.random.default_rng(wire)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        rho = np.outer(psi, psi.conj()) / np.vdot(psi, psi)

        paulis = [embed(P, (wire,), n_qubits) for P in (PAULI_X, PAULI_Y, PAULI_Z)]
        expected = (1 - p) * rho + p / 3 * sum(P @ rho @ P for P in paulis)
        result = depolarize(rho, p, wire, n_qubits)

        np.testing.assert_allclose(result, expected, atol=1e-12)
        assert np.trace(result) == pytest.approx(1.0)


class TestTranscription:
    """Compare the shared schedule with JAQSI pulse gates."""

    @pytest.mark.parametrize("xp", [np, jnp])
    def test_envelopes_match_jaqsi(self, xp):
        """Match JAQSI's Gaussian and DRAG envelopes.

        The in-phase term vanishes at pulse edges; both terms remain finite where
        adaptive solvers may probe outside the pulse.
        """
        from jaqsi.pulses import PulseEnvelope

        (amplitude, _, sigma), duration = ry_envelope("drag")
        envelope = (amplitude, 0.3, sigma)
        ts = np.array([-1e11, 0.0, 0.3 * duration, duration / 2, duration, 1e11])

        for ours, fn in (
            (in_phase_env, PulseEnvelope.drag),
            (quadrature_env, PulseEnvelope.drag_quadrature),
        ):
            values = np.asarray(ours(xp.asarray(ts), envelope, duration, xp))
            expected = [fn(jnp.array(envelope), t, duration / 2) for t in ts]
            np.testing.assert_allclose(values, expected, atol=1e-12)
            assert np.all(np.isfinite(values))
        assert in_phase_env(0.0, envelope, duration) == 0.0
        assert in_phase_env(duration, envelope, duration) == 0.0

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

    def test_drag_quadrature_matches_jaqsi_pulse(self, drag_with_beta, monkeypatch):
        """Match the sign and axis of JAQSI's driven-RY DRAG quadrature."""
        n_qubits = 2
        sim = JaqsiPulseBenchmark()
        spec = _spec(n_qubits)
        sim.setup(spec, "state")
        jaqsi_psi = np.asarray(sim.run(jnp.array([_inputs(spec)]), jnp.zeros(0)))[0]

        np.testing.assert_allclose(
            jaqsi_psi,
            _scipy_state(n_qubits, PHI, "drag"),
            atol=SOLVER_PRECISION,
            err_msg="transcribed DRAG schedule diverges from jaqsi's pulse gates",
        )
        monkeypatch.setattr(pulse_model, "PAULI_X", -PAULI_X)
        flipped = _scipy_state(n_qubits, PHI, "drag")
        assert np.max(np.abs(jaqsi_psi - flipped)) > 100 * SOLVER_PRECISION

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
    """Compare calibrated pulses with their gate-level circuit."""

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
