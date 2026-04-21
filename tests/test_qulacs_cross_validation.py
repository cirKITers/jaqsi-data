"""Cross-validation tests: verify Qulacs matches PennyLane for all modes.

These tests replicate the benchmark runner's cross-validation logic,
ensuring the Qulacs adapter produces results consistent with PennyLane
(which uses the same convention as the YAQSI reference).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.simulators.pennylane_sim import PennylaneBenchmark
from benchmark.simulators.qulacs_sim import QulacsBenchmark


# Enable 64-bit precision for JAX (matches the benchmark runner)
jax.config.update("jax_enable_x64", True)

PRECISION = 1e-8


def _run_single(sim, n_qubits: int, mode: str, phi: float):
    """Set up a simulator, run a single phi value, return the result."""
    sim.setup(n_qubits, mode)
    batch = jnp.array([phi])
    return sim.run(batch)


class TestQulacsCrossValidation:
    """Compare Qulacs output against PennyLane for every mode."""

    @pytest.fixture()
    def pennylane(self):
        return PennylaneBenchmark()

    @pytest.fixture()
    def qulacs(self):
        return QulacsBenchmark()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_state_matches(self, pennylane, qulacs, n_qubits):
        phi = 1.0
        pl = _run_single(pennylane, n_qubits, "state", phi)
        qu = _run_single(qulacs, n_qubits, "state", phi)
        np.testing.assert_allclose(
            qu, pl, atol=PRECISION,
            err_msg=f"state mismatch for {n_qubits} qubits",
        )

    # ------------------------------------------------------------------
    # probs
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_probs_matches(self, pennylane, qulacs, n_qubits):
        phi = 1.0
        pl = _run_single(pennylane, n_qubits, "probs", phi)
        qu = _run_single(qulacs, n_qubits, "probs", phi)
        np.testing.assert_allclose(
            qu, pl, atol=PRECISION,
            err_msg=f"probs mismatch for {n_qubits} qubits",
        )

    # ------------------------------------------------------------------
    # expval
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_expval_matches(self, pennylane, qulacs, n_qubits):
        phi = 1.0
        pl = _run_single(pennylane, n_qubits, "expval", phi)
        qu = _run_single(qulacs, n_qubits, "expval", phi)
        # PennyLane returns expval as (n_obs, batch) while others use
        # (batch, n_obs), so transpose PennyLane's output.
        pl_t = jnp.asarray(pl).T
        np.testing.assert_allclose(
            qu, pl_t, atol=PRECISION,
            err_msg=f"expval mismatch for {n_qubits} qubits",
        )

    # ------------------------------------------------------------------
    # density
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_density_matches(self, pennylane, qulacs, n_qubits):
        phi = 1.0
        pl = _run_single(pennylane, n_qubits, "density", phi)
        qu = _run_single(qulacs, n_qubits, "density", phi)
        np.testing.assert_allclose(
            qu, pl, atol=PRECISION,
            err_msg=f"density mismatch for {n_qubits} qubits",
        )

    # ------------------------------------------------------------------
    # Multiple phi values (batch)
    # ------------------------------------------------------------------
    def test_state_batch(self, pennylane, qulacs):
        n_qubits = 2
        pennylane.setup(n_qubits, "state")
        qulacs.setup(n_qubits, "state")
        batch = jnp.array([0.5, 1.0, -0.3])
        pl = pennylane.run(batch)
        qu = qulacs.run(batch)
        np.testing.assert_allclose(qu, pl, atol=PRECISION)