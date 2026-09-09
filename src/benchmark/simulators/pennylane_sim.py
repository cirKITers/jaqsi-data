"""PennyLane simulator benchmark adapter."""

from __future__ import annotations

import logging
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from benchmark.circuits import CircuitSpec, angle
from benchmark.simulators.base import SimulatorBenchmark, Mode

logger = logging.getLogger(__name__)


def _apply(spec: CircuitSpec, inputs, weights) -> None:
    """Queue the operations of *spec* onto the active PennyLane tape."""
    for op in spec.ops:
        if op.gate == "H":
            qml.Hadamard(wires=op.wires[0])
        elif op.gate == "RX":
            qml.RX(angle(op, inputs, weights), wires=op.wires[0])
        elif op.gate == "RZ":
            qml.RZ(angle(op, inputs, weights), wires=op.wires[0])
        elif op.gate == "CRX":
            qml.CRX(angle(op, inputs, weights), wires=list(op.wires))
        elif op.gate == "CNOT":
            qml.CNOT(wires=list(op.wires))
        else:
            raise ValueError(f"Unsupported gate: {op.gate!r}")


class PennylaneBenchmark(SimulatorBenchmark):
    """PennyLane on ``default.qubit``, differentiated by backpropagation.

    Forward modes rely on PennyLane's own parameter broadcasting for the batch
    axis rather than on a Python loop, which is the fastest batching route the
    framework offers.
    """

    name = "pennylane"

    device_name = "default.qubit"
    diff_method = "backprop"
    # Whether the forward pass and the gradient run through JAX, or through
    # PennyLane's own interface.  Compiled devices such as lightning cannot be
    # traced, so JAX routes every call through a host callback instead.
    jax_forward = True
    jax_gradient = True

    def __init__(self) -> None:
        self._run_fn: Callable | None = None
        self._spec: CircuitSpec | None = None
        self._mode: Mode = "probs"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        self._spec = spec
        self._mode = mode

        dev = qml.device(self.device_name, wires=spec.n_qubits)

        if mode == "grad":
            self._run_fn = self._make_grad_fn(spec, dev)
            return

        return_map: dict[str, Callable] = {
            "density": lambda: qml.density_matrix(wires=range(spec.n_qubits)),
            "state": lambda: qml.state(),
            "probs": lambda: qml.probs(wires=range(spec.n_qubits)),
            "expval": lambda: [
                qml.expval(qml.PauliZ(i)) for i in range(spec.n_qubits)
            ],
        }

        if not self.jax_forward:
            # A compiled device gains nothing from the JAX interface: it cannot
            # be traced, so every call goes through a host callback, which costs
            # roughly twice the native call and measures the bridge rather than
            # the simulator.
            @qml.qnode(dev, diff_method=None)
            def circuit(inputs, weights):
                _apply(spec, inputs, weights)
                return return_map[mode]()

            def run_native(inputs, weights):
                return jnp.asarray(
                    np.asarray(circuit(np.asarray(inputs), np.asarray(weights)))
                )

            self._run_fn = run_native
        elif optimal_config:
            # Compile the QNode with jax.jit (XLA) for fast repeated calls.
            #
            # ``diff_method`` must not be None here.  It reads as the natural
            # choice for a forward-only benchmark, but it takes default.qubit
            # off the JAX-traced path, so every call round-trips through host
            # numpy and jax.jit has nothing to compile.  Measured on the
            # hardware-efficient ansatz at batch 10, that costs 9x at ten
            # qubits and 147x at four, for bit-identical results.  Naming the
            # backprop pipeline keeps the circuit traceable; the gradient
            # infrastructure costs nothing when no gradient is taken.
            @qml.qnode(dev, interface="jax", diff_method="backprop")
            def circuit(inputs, weights):
                _apply(spec, inputs, weights)
                return return_map[mode]()

            self._run_fn = jax.jit(circuit)
        else:
            @qml.qnode(dev, interface="jax")
            def circuit(inputs, weights):
                _apply(spec, inputs, weights)
                return return_map[mode]()

            self._run_fn = circuit

    def _make_grad_fn(self, spec: CircuitSpec, dev) -> Callable:
        """Return the gradient of the summed Pauli-Z expectation.

        The loss matches the other adapters: the per-qubit $\\langle Z \\rangle$
        summed over qubits and over the batch, differentiated with respect to
        the circuit's trainable vector.
        """
        observable = qml.sum(*[qml.PauliZ(i) for i in range(spec.n_qubits)])
        argnum = 1 if spec.trainable == "weights" else 0

        if self.jax_gradient:
            @qml.qnode(dev, interface="jax", diff_method=self.diff_method)
            def circuit(inputs, weights):
                _apply(spec, inputs, weights)
                return qml.expval(observable)

            def loss(inputs, weights):
                # Parameter broadcasting returns one expectation per sample.
                return jnp.sum(circuit(inputs, weights))

            return jax.jit(jax.grad(loss, argnums=argnum))

        # adjoint and parameter-shift run through PennyLane's own interface.
        # Both accept a broadcast batch, which costs about half of looping in
        # Python, so that is the route taken wherever it is supported.
        @qml.qnode(dev, diff_method=self.diff_method)
        def circuit(inputs, weights):
            _apply(spec, inputs, weights)
            return qml.expval(observable)

        grad_batched = qml.grad(
            lambda x, w: qml.math.sum(circuit(x, w)), argnums=argnum
        )
        grad_single = qml.grad(circuit, argnums=argnum)

        def _to_pennylane(inputs, weights):
            return (
                qml.numpy.array(np.asarray(inputs), requires_grad=argnum == 0),
                qml.numpy.array(np.asarray(weights), requires_grad=argnum == 1),
            )

        def broadcast_grad(inputs, weights):
            x, w = _to_pennylane(inputs, weights)
            return jnp.asarray(np.asarray(grad_batched(x, w)))

        def looped_grad(inputs, weights):
            x, w = _to_pennylane(inputs, weights)
            grads = [np.asarray(grad_single(sample, w)) for sample in x]
            if argnum == 0:
                return jnp.asarray(np.stack(grads))
            return jnp.asarray(np.sum(grads, axis=0))

        # Probe the broadcast route once, here, where it is not timed.  The
        # parameter-shift transform refuses to differentiate parameters that
        # are themselves broadcast, which is the ``crx_ring`` case; adjoint
        # accepts both.  Probing rather than hard-coding that rule keeps the
        # choice correct across PennyLane versions.
        probe = (np.zeros((2, spec.n_inputs)), np.zeros(spec.n_weights))
        try:
            broadcast_grad(*probe)
        except NotImplementedError:
            logger.info(
                f"{self.name}: {self.diff_method} cannot differentiate a "
                f"broadcast batch here; falling back to a per-sample loop."
            )
            return looped_grad
        return broadcast_grad

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)

    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)


class PennylaneAdjointBenchmark(PennylaneBenchmark):
    """PennyLane on ``lightning.qubit`` with the adjoint gradient.

    Lightning is PennyLane's compiled state-vector backend and adjoint
    differentiation is the method it is tuned for, so this is the reference
    point a gradient comparison against PennyLane has to clear.
    """

    name = "pennylane_adjoint"

    device_name = "lightning.qubit"
    diff_method = "adjoint"
    jax_forward = False
    jax_gradient = False


class PennylanePsrBenchmark(PennylaneBenchmark):
    """PennyLane on ``default.qubit`` with parameter-shift gradients.

    Included because the parameter-shift rule is the only gradient that also
    runs on hardware, which makes it the baseline QML papers report even
    though it costs two circuit evaluations per parameter.
    """

    name = "pennylane_psr"

    device_name = "default.qubit"
    diff_method = "parameter-shift"
    jax_gradient = False

    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        # The forward pass is identical to the ``pennylane`` adapter; only the
        # gradient differs, so running the other modes would duplicate rows.
        return mode == "grad"
