"""Tests for the backend-agnostic circuit specifications.

Every adapter builds its circuit from a :class:`CircuitSpec`, so the parameter
bookkeeping here is what keeps all simulators executing the same circuit.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.circuits import FAMILIES, GATES, PULSE_FAMILIES, Op, angle, build_spec

_FAMILIES = sorted(FAMILIES)


class TestBuildSpec:
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_records_its_size(self, family):
        spec = build_spec(family, 4, 3)
        assert spec.family == family
        assert spec.n_qubits == 4
        assert spec.n_layers == 3

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown circuit family"):
            build_spec("nope", 2, 1)

    @pytest.mark.parametrize("n_layers", [0, -1])
    def test_non_positive_depth_raises(self, n_layers):
        with pytest.raises(ValueError, match="at least 1"):
            build_spec("hea", 2, n_layers)


class TestOperations:
    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("n_qubits", [2, 3, 5])
    @pytest.mark.parametrize("n_layers", [1, 2, 4])
    def test_gates_are_implementable(self, family, n_qubits, n_layers):
        for op in build_spec(family, n_qubits, n_layers).ops:
            assert op.gate in GATES

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("n_qubits", [2, 3, 5])
    def test_wires_are_in_range(self, family, n_qubits):
        for op in build_spec(family, n_qubits, 2).ops:
            assert all(0 <= w < n_qubits for w in op.wires)
            # Two-qubit gates must act on distinct wires.
            assert len(set(op.wires)) == len(op.wires)

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("n_qubits", [2, 3, 5])
    @pytest.mark.parametrize("n_layers", [1, 2, 4])
    def test_every_parameter_is_used_exactly_once(self, family, n_qubits, n_layers):
        """No angle is shared between gates and none is left unused."""
        spec = build_spec(family, n_qubits, n_layers)
        used = {"inputs": [], "weights": []}
        for op in spec.ops:
            if op.source is not None:
                used[op.source].append(op.index)

        assert sorted(used["inputs"]) == list(range(spec.n_inputs))
        assert sorted(used["weights"]) == list(range(spec.n_weights))

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_non_parametric_gates_have_no_source(self, family):
        for op in build_spec(family, 3, 2).ops:
            if op.gate in ("H", "CNOT"):
                assert op.source is None


class TestParameterCounts:
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_hea_counts(self, n_layers):
        n_qubits = 4
        spec = build_spec("hea", n_qubits, n_layers)
        # One RX encoding gate per wire; three weights per wire per layer,
        # except the last layer, which drops its trailing RZ.
        assert spec.n_inputs == n_qubits
        assert spec.n_weights == 3 * n_layers * n_qubits - n_qubits

    def test_crx_ring_counts(self):
        spec = build_spec("crx_ring", 4, 3)
        # One CRX angle per wire per layer, and no shared weights.
        assert spec.n_inputs == 3 * 4
        assert spec.n_weights == 0

    def test_crx_ring_has_one_hadamard_layer(self):
        spec = build_spec("crx_ring", 4, 3)
        assert sum(op.gate == "H" for op in spec.ops) == 4

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_gate_count_grows_with_depth(self, family):
        shallow = build_spec(family, 3, 1)
        deep = build_spec(family, 3, 4)
        assert len(deep.ops) > len(shallow.ops)


class TestKnownAnsatzProperties:
    """Properties of `hea` that the timings and the gradient depend on."""

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_last_block_ends_on_rx(self, n_layers):
        """No weight may sit in a position that cannot move the observable.

        A trailing RZ could not change a Pauli-Z expectation value: it only
        adds phases, and the CNOT ring after it permutes the computational
        basis.  Following Yao and Qulacs, the last block is RZ-RX, so every
        weight has a non-zero gradient.
        """
        n_qubits = 4
        spec = build_spec("hea", n_qubits, n_layers)
        weight_ops = [op for op in spec.ops if op.source == "weights"]
        last_block = weight_ops[-2 * n_qubits:]
        assert [op.gate for op in last_block] == ["RZ", "RX"] * n_qubits

    def test_first_block_keeps_its_leading_rz(self):
        """The encoding layer rotates off the Z axis, so a leading RZ acts."""
        spec = build_spec("hea", 4, 3)
        weight_ops = [op for op in spec.ops if op.source == "weights"]
        assert weight_ops[0].gate == "RZ"


class TestTrainableVector:
    def test_hea_trains_shared_weights(self):
        spec = build_spec("hea", 3, 2)
        assert spec.trainable == "weights"
        assert spec.n_trainable == spec.n_weights

    def test_crx_ring_trains_its_inputs(self):
        spec = build_spec("crx_ring", 3, 2)
        assert spec.trainable == "inputs"
        assert spec.n_trainable == spec.n_inputs


class TestAngleLookup:
    def test_reads_from_the_named_vector(self):
        spec = build_spec("hea", 2, 1)
        inputs = np.arange(spec.n_inputs)
        weights = 10 * np.arange(spec.n_weights)
        for op in spec.ops:
            if op.source == "inputs":
                assert angle(op, inputs, weights) == op.index
            elif op.source == "weights":
                assert angle(op, inputs, weights) == 10 * op.index

    def test_indexes_the_last_axis_of_a_batch(self):
        """Adapters relying on their own broadcasting pass a whole batch."""
        spec = build_spec("hea", 2, 1)
        batch = np.tile(np.arange(spec.n_inputs), (3, 1))
        weights = np.zeros(spec.n_weights)
        encoding = next(op for op in spec.ops if op.source == "inputs")
        np.testing.assert_array_equal(
            angle(encoding, batch, weights),
            np.full(3, encoding.index),
        )


class TestDepolarizing:
    """The noise mode's channels are ops in the spec, like the gates."""

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_noise_free_by_default(self, family):
        spec = build_spec(family, 3, 2)
        assert spec.depolarizing == 0.0
        assert all(op.gate != "DEPOL" for op in spec.ops)

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_channel_follows_every_gate_on_each_wire(self, family):
        clean = build_spec(family, 3, 2)
        noisy = build_spec(family, 3, 2, depolarizing=0.1)
        assert noisy.depolarizing == 0.1

        ops = iter(noisy.ops)
        for gate in clean.ops:
            assert next(ops) == gate
            for w in gate.wires:
                assert next(ops) == Op("DEPOL", (w,))
        assert next(ops, None) is None


class TestPulseFamilies:
    def test_only_transcribed_families_are_listed(self):
        assert PULSE_FAMILIES <= set(FAMILIES)

    def test_crx_ring_is_pulse_capable(self):
        assert "crx_ring" in PULSE_FAMILIES

    def test_pulse_families_use_only_transcribed_gates(self):
        """pulse_model transcribes the Hadamard and CRX decompositions only."""
        for family in PULSE_FAMILIES:
            for op in build_spec(family, 3, 2).ops:
                assert op.gate in ("H", "CRX")
