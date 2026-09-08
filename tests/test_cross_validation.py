"""Cross-validation tests: verify all simulators produce matching results.

Every simulator adapter is compared against PennyLane (the reference
implementation) across all measurement modes, both circuit families and
several qubit counts.  PennyLane returns expval as ``(n_obs, batch)`` while the
other adapters use ``(batch, n_obs)``, so the comparison accounts for that
transpose.

The gradient mode is cross-validated separately and against jaqsi, since the
adapters reach it through four different differentiation methods:
backpropagation, the adjoint method, the parameter-shift rule and Qulacs'
backprop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.circuits import build_spec
from benchmark.simulators.jaqsi_sim import JaqsiBenchmark
from benchmark.simulators.pennylane_sim import (
    PennylaneAdjointBenchmark,
    PennylaneBenchmark,
    PennylanePsrBenchmark,
)
from benchmark.simulators.qiskit_sim import QiskitBenchmark
from benchmark.simulators.qibo_sim import QiboBenchmark
from benchmark.simulators.qulacs_sim import QulacsBenchmark

# Enable 64-bit precision for JAX (matches the benchmark runner)
jax.config.update("jax_enable_x64", True)

PRECISION = 1e-8

# Reverse-mode AD, the adjoint method and Qulacs' backprop accumulate their
# sums in different orders, so gradients agree to a looser bound.
GRAD_PRECISION = 1e-6

# Simulators under test (excluding PennyLane, which is the reference).
_OTHER_SIMULATORS = [
    pytest.param(PennylaneAdjointBenchmark, id="pennylane_adjoint"),
    pytest.param(QiskitBenchmark, id="qiskit"),
    pytest.param(QiboBenchmark, id="qibo"),
    pytest.param(QulacsBenchmark, id="qulacs"),
]

_FAMILIES = ["crx_ring", "hea"]


def _params(spec, batch_size: int = 1):
    """Return reproducible ``(inputs, weights)`` for *spec*."""
    key = jax.random.PRNGKey(7)
    input_key, weight_key = jax.random.split(key)
    inputs = jax.random.uniform(
        input_key, (batch_size, spec.n_inputs), minval=-jnp.pi, maxval=jnp.pi
    )
    weights = jax.random.uniform(
        weight_key, (spec.n_weights,), minval=-jnp.pi, maxval=jnp.pi
    )
    return inputs, weights


def _run(sim, spec, mode: str, batch_size: int = 1):
    """Set up a simulator and run it on *spec*."""
    sim.setup(spec, mode)
    return sim.run(*_params(spec, batch_size))


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def pennylane():
    return PennylaneBenchmark()


# ------------------------------------------------------------------
# Single-value cross-validation
# ------------------------------------------------------------------

class TestCrossValidation:
    """Compare every simulator against PennyLane for every mode."""

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_state_matches(self, pennylane, sim_cls, n_qubits, family):
        spec = build_spec(family, n_qubits, 2)
        other = sim_cls()
        pl = _run(pennylane, spec, "state")
        ot = _run(other, spec, "state")
        np.testing.assert_allclose(
            ot, pl, atol=PRECISION,
            err_msg=f"state mismatch: {other.name} vs pennylane, {n_qubits} qubits",
        )

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_probs_matches(self, pennylane, sim_cls, n_qubits, family):
        spec = build_spec(family, n_qubits, 2)
        other = sim_cls()
        pl = _run(pennylane, spec, "probs")
        ot = _run(other, spec, "probs")
        np.testing.assert_allclose(
            ot, pl, atol=PRECISION,
            err_msg=f"probs mismatch: {other.name} vs pennylane, {n_qubits} qubits",
        )

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_expval_matches(self, pennylane, sim_cls, n_qubits, family):
        spec = build_spec(family, n_qubits, 2)
        other = sim_cls()
        pl = _run(pennylane, spec, "expval")
        ot = _run(other, spec, "expval")
        # PennyLane returns expval as (n_obs, batch) while others use
        # (batch, n_obs), so transpose PennyLane's output.
        expected = jnp.asarray(pl)
        if not other.name.startswith("pennylane"):
            expected = expected.T
        np.testing.assert_allclose(
            ot, expected, atol=PRECISION,
            err_msg=f"expval mismatch: {other.name} vs pennylane, {n_qubits} qubits",
        )

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_density_matches(self, pennylane, sim_cls, n_qubits, family):
        spec = build_spec(family, n_qubits, 2)
        other = sim_cls()
        pl = _run(pennylane, spec, "density")
        ot = _run(other, spec, "density")
        np.testing.assert_allclose(
            ot, pl, atol=PRECISION,
            err_msg=f"density mismatch: {other.name} vs pennylane, {n_qubits} qubits",
        )


# ------------------------------------------------------------------
# Batch cross-validation
# ------------------------------------------------------------------

class TestBatchCrossValidation:
    """Verify batch execution produces consistent results across simulators."""

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    def test_state_batch(self, pennylane, sim_cls, family):
        spec = build_spec(family, 2, 2)
        other = sim_cls()
        pl = _run(pennylane, spec, "state", batch_size=3)
        ot = _run(other, spec, "state", batch_size=3)
        np.testing.assert_allclose(
            ot, pl, atol=PRECISION,
            err_msg=f"state batch mismatch: {other.name} vs pennylane",
        )

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_weights_are_shared_across_the_batch(self, family):
        """Every sample of a batch must see the same weights."""
        spec = build_spec(family, 3, 2)
        sim = JaqsiBenchmark()
        sim.setup(spec, "state")
        inputs, weights = _params(spec, batch_size=2)
        repeated = jnp.concatenate([inputs[:1], inputs[:1]], axis=0)
        out = sim.run(repeated, weights)
        np.testing.assert_allclose(out[0], out[1], atol=PRECISION)


# ------------------------------------------------------------------
# Gradient cross-validation
# ------------------------------------------------------------------

# Every adapter that implements the gradient mode, with the differentiation
# method it uses.  Qiskit and Qibo are absent: their gradient interfaces live in
# separate packages that are not dependencies here.
_GRAD_SIMULATORS = [
    pytest.param(PennylaneBenchmark, id="pennylane-backprop"),
    pytest.param(PennylaneAdjointBenchmark, id="pennylane-adjoint"),
    pytest.param(PennylanePsrBenchmark, id="pennylane-parameter-shift"),
    pytest.param(QulacsBenchmark, id="qulacs-backprop"),
]


class TestGradientCrossValidation:
    """All differentiation methods must agree with jaqsi's reverse-mode AD."""

    @pytest.mark.parametrize("sim_cls", _GRAD_SIMULATORS)
    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_grad_matches_jaqsi(self, sim_cls, n_qubits):
        spec = build_spec("hea", n_qubits, 2)
        other = sim_cls()
        assert other.supports(spec, "grad")
        reference = _run(JaqsiBenchmark(), spec, "grad")
        ot = _run(other, spec, "grad")
        np.testing.assert_allclose(
            ot, reference, atol=GRAD_PRECISION,
            err_msg=f"grad mismatch: {other.name} vs jaqsi, {n_qubits} qubits",
        )

    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_crx_ring_grad_matches_jaqsi(self, n_qubits):
        """crx_ring trains its inputs, so the gradient keeps the batch axis."""
        spec = build_spec("crx_ring", n_qubits, 2)
        other = PennylaneBenchmark()
        reference = _run(JaqsiBenchmark(), spec, "grad", batch_size=2)
        ot = _run(other, spec, "grad", batch_size=2)
        assert np.shape(reference) == (2, spec.n_inputs)
        np.testing.assert_allclose(ot, reference, atol=GRAD_PRECISION)

    def test_grad_shape_follows_trainable_vector(self):
        """hea trains shared weights, so its gradient has no batch axis."""
        spec = build_spec("hea", 3, 2)
        out = _run(JaqsiBenchmark(), spec, "grad", batch_size=4)
        assert spec.trainable == "weights"
        assert np.shape(out) == (spec.n_weights,)

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_grad_matches_finite_differences(self, family):
        """An independent check that the differentiated loss is the right one."""
        spec = build_spec(family, 3, 2)
        sim = JaqsiBenchmark()
        sim.setup(spec, "grad")
        inputs, weights = _params(spec, batch_size=2)
        analytic = np.asarray(sim.run(inputs, weights)).ravel()

        expval = JaqsiBenchmark()
        expval.setup(spec, "expval")

        def loss(ins, wts) -> float:
            return float(jnp.sum(expval.run(ins, wts)))

        # Perturb the trainable vector one entry at a time.
        trains_inputs = spec.trainable == "inputs"
        base = inputs if trains_inputs else weights
        eps = 1e-6
        numeric = []
        for index in np.ndindex(base.shape):
            shifted_up = base.at[index].add(eps)
            shifted_down = base.at[index].add(-eps)
            if trains_inputs:
                numeric.append(
                    (loss(shifted_up, weights) - loss(shifted_down, weights))
                    / (2 * eps)
                )
            else:
                numeric.append(
                    (loss(inputs, shifted_up) - loss(inputs, shifted_down)) / (2 * eps)
                )

        np.testing.assert_allclose(analytic, np.array(numeric), atol=1e-6)
