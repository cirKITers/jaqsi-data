"""Cross-validation tests: verify all simulators produce matching results.

Every simulator adapter is compared against PennyLane (the reference
implementation) across all measurement modes and several qubit counts.
PennyLane returns expval as ``(n_obs, batch)`` while the other adapters
use ``(batch, n_obs)``, so the comparison accounts for that transpose.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.simulators.pennylane_sim import PennylaneBenchmark
from benchmark.simulators.qiskit_sim import QiskitBenchmark
from benchmark.simulators.qibo_sim import QiboBenchmark
from benchmark.simulators.qulacs_sim import QulacsBenchmark

# Enable 64-bit precision for JAX (matches the benchmark runner)
jax.config.update("jax_enable_x64", True)

PRECISION = 1e-8

# Simulators under test (excluding PennyLane, which is the reference).
_OTHER_SIMULATORS = [
    pytest.param(QiskitBenchmark, id="qiskit"),
    pytest.param(QiboBenchmark, id="qibo"),
    pytest.param(QulacsBenchmark, id="qulacs"),
]


def _run_single(sim, n_qubits: int, mode: str, phi: float):
    """Set up a simulator, run a single phi value, return the result."""
    sim.setup(n_qubits, mode)
    batch = jnp.array([phi])
    return sim.run(batch)


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

    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_state_matches(self, pennylane, sim_cls, n_qubits):
        phi = 1.0
        other = sim_cls()
        pl = _run_single(pennylane, n_qubits, "state", phi)
        ot = _run_single(other, n_qubits, "state", phi)
        np.testing.assert_allclose(
            ot, pl, atol=PRECISION,
            err_msg=f"state mismatch: {other.name} vs pennylane, {n_qubits} qubits",
        )

    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_probs_matches(self, pennylane, sim_cls, n_qubits):
        phi = 1.0
        other = sim_cls()
        pl = _run_single(pennylane, n_qubits, "probs", phi)
        ot = _run_single(other, n_qubits, "probs", phi)
        np.testing.assert_allclose(
            ot, pl, atol=PRECISION,
            err_msg=f"probs mismatch: {other.name} vs pennylane, {n_qubits} qubits",
        )

    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_expval_matches(self, pennylane, sim_cls, n_qubits):
        phi = 1.0
        other = sim_cls()
        pl = _run_single(pennylane, n_qubits, "expval", phi)
        ot = _run_single(other, n_qubits, "expval", phi)
        # PennyLane returns expval as (n_obs, batch) while others use
        # (batch, n_obs), so transpose PennyLane's output.
        pl_t = jnp.asarray(pl).T
        np.testing.assert_allclose(
            ot, pl_t, atol=PRECISION,
            err_msg=f"expval mismatch: {other.name} vs pennylane, {n_qubits} qubits",
        )

    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_density_matches(self, pennylane, sim_cls, n_qubits):
        phi = 1.0
        other = sim_cls()
        pl = _run_single(pennylane, n_qubits, "density", phi)
        ot = _run_single(other, n_qubits, "density", phi)
        np.testing.assert_allclose(
            ot, pl, atol=PRECISION,
            err_msg=f"density mismatch: {other.name} vs pennylane, {n_qubits} qubits",
        )


# ------------------------------------------------------------------
# Batch cross-validation
# ------------------------------------------------------------------

class TestBatchCrossValidation:
    """Verify batch execution produces consistent results across simulators."""

    @pytest.mark.parametrize("sim_cls", _OTHER_SIMULATORS)
    def test_state_batch(self, pennylane, sim_cls):
        n_qubits = 2
        other = sim_cls()
        pennylane.setup(n_qubits, "state")
        other.setup(n_qubits, "state")
        batch = jnp.array([0.5, 1.0, -0.3])
        pl = pennylane.run(batch)
        ot = other.run(batch)
        np.testing.assert_allclose(
            ot, pl, atol=PRECISION,
            err_msg=f"state batch mismatch: {other.name} vs pennylane",
        )