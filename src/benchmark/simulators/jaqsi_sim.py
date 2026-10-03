"""Jaqsi simulator benchmark adapter."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from jaqsi import DepolarizingChannel, Gates, PauliZ, Script

from benchmark.circuits import CircuitSpec, angle
from benchmark.simulators.base import SimulatorBenchmark, Mode


def build_circuit(spec: CircuitSpec, *, pulse: bool = False) -> Callable:
    """Build the jaqsi circuit from ``spec``.

    The gate and pulse adapters share this builder and select gates with the
    ``pulse`` flag.
    """

    def circuit(inputs: jnp.ndarray, weights: jnp.ndarray) -> None:
        for op in spec.ops:
            if op.gate == "H":
                Gates.H(wires=op.wires[0], pulse=pulse)
            elif op.gate == "RX":
                Gates.RX(w=angle(op, inputs, weights), wires=op.wires[0], pulse=pulse)
            elif op.gate == "RZ":
                Gates.RZ(w=angle(op, inputs, weights), wires=op.wires[0], pulse=pulse)
            elif op.gate == "CRX":
                Gates.CRX(
                    w=angle(op, inputs, weights), wires=list(op.wires), pulse=pulse
                )
            elif op.gate == "CNOT":
                Gates.CX(wires=list(op.wires), pulse=pulse)
            elif op.gate == "DEPOL":
                # A channel on the tape switches jaqsi to density-matrix
                # simulation.
                DepolarizingChannel(spec.depolarizing, wires=op.wires[0])
            else:
                raise ValueError(f"Unsupported gate: {op.gate!r}")

    return circuit


class JaqsiBenchmark(SimulatorBenchmark):
    name = "jaqsi"

    pulse = False

    def __init__(self) -> None:
        self._script: Script | None = None
        self._spec: CircuitSpec | None = None
        self._mode: Mode = "probs"
        self._run_fn: Callable | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        self._spec = spec
        self._mode = mode
        self._script = Script(f=build_circuit(spec, pulse=self.pulse))
        self._run_fn = self._make_run_fn(spec, mode)

    def _make_run_fn(self, spec: CircuitSpec, mode: Mode) -> Callable:
        """Build the batched execution function for ``mode``.

        Vectorize over ``inputs`` while sharing ``weights`` across the batch.
        """
        script = self._script
        obs = [PauliZ(wires=i, record=False) for i in range(spec.n_qubits)]

        if mode != "grad":
            # The noise mode measures the density matrix; its channels are
            # already part of the circuit.
            measurement = "density" if mode == "noise" else mode

            def _forward(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                return script.execute(
                    type=measurement,
                    obs=obs,
                    args=(inputs, weights),
                    in_axes=(0, None),
                )

            return _forward

        # Gradient of the summed Pauli-Z expectation over the whole batch.
        # jaqsi is pure JAX, so reverse-mode AD differentiates straight through
        # the vmapped kernel and no separate gradient interface is needed.
        argnums = 1 if spec.trainable == "weights" else 0

        def _loss(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
            expvals = script.execute(
                type="expval",
                obs=obs,
                args=(inputs, weights),
                in_axes=(0, None),
            )
            return jnp.sum(expvals)

        return jax.jit(jax.grad(_loss, argnums=argnums))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)

    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)
