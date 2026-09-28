"""Qibo simulator benchmark adapter.

Local statevector / density-matrix simulation, no external provider or API key
required.  ``optimal_config`` selects the qibojit backend in every mode; see
:meth:`QiboBenchmark.setup` for the measurements behind that choice.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import jax.numpy as jnp

from qibo import Circuit, gates, set_backend, set_threads
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z

from benchmark.circuits import CircuitSpec, Op
from benchmark.simulators.base import SimulatorBenchmark, Mode
from benchmark.threads import num_threads


class QiboBenchmark(SimulatorBenchmark):
    name = "qibo"

    def __init__(self) -> None:
        self._circuit: Circuit | None = None
        self._circuit_dm: Circuit | None = None
        self._spec: CircuitSpec | None = None
        self._mode: Mode = "probs"
        self._n_qubits: int = 0
        # Parametric gates in the order Qibo's ``set_parameters`` expects them.
        self._param_ops: Tuple[Op, ...] = ()
        self._run_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None
        self._z_hams: List[SymbolicHamiltonian] = []

    def supports(self, spec: CircuitSpec, mode: Mode) -> bool:
        # Qibo's differentiation lives in the separate qiboml package, which is
        # not a dependency here; the forward measurements are unaffected.
        return mode != "grad"

    # ------------------------------------------------------------------
    # Circuit builders
    # ------------------------------------------------------------------
    def _build_circuit(
        self, spec: CircuitSpec, *, density_matrix: bool = False
    ) -> Circuit:
        """Build the parametric benchmark circuit of *spec*."""
        c = Circuit(spec.n_qubits, density_matrix=density_matrix)
        for op in spec.ops:
            if op.gate == "H":
                c.add(gates.H(op.wires[0]))
            elif op.gate == "RX":
                c.add(gates.RX(op.wires[0], theta=0.0))
            elif op.gate == "RZ":
                c.add(gates.RZ(op.wires[0], theta=0.0))
            elif op.gate == "CRX":
                c.add(gates.CRX(op.wires[0], op.wires[1], theta=0.0))
            elif op.gate == "CNOT":
                c.add(gates.CNOT(op.wires[0], op.wires[1]))
            elif op.gate == "DEPOL":
                # Qibo's $\lambda$ is the probability $p$ of the other
                # frameworks at $\lambda = 4p/3$.
                c.add(
                    gates.DepolarizingChannel((op.wires[0],), 4 * spec.depolarizing / 3)
                )
            else:
                raise ValueError(f"Unsupported gate: {op.gate!r}")
        return c

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(
        self, spec: CircuitSpec, mode: Mode, *, optimal_config: bool = False
    ) -> None:
        # Backend selection is global to the process but only affects Qibo, so
        # it is set on every call and neither path can leak into the other.
        #
        if optimal_config:
            set_backend("qibojit", platform="numba")
            # qibojit's constructor pins numba to one thread per available
            # core, ignoring the environment, so it is the one backend that has
            # to be pinned after the fact.  Setting it before construction is
            # not an option: NUMBA_NUM_THREADS is a cap, and a cap below the
            # core count makes that same constructor raise.
            threads = num_threads()
            if threads is not None:
                set_threads(threads)
        else:
            # Single-threaded by construction, and it rejects ``set_threads``
            # for any count above one, so it is left alone.
            set_backend("numpy")

        self._spec = spec
        self._n_qubits = spec.n_qubits
        self._mode = mode
        self._param_ops = tuple(op for op in spec.ops if op.source is not None)

        if mode in ("density", "noise"):
            self._circuit_dm = self._build_circuit(spec, density_matrix=True)
            self._circuit = None
        else:
            self._circuit = self._build_circuit(spec)
            self._circuit_dm = None

        if mode == "expval":
            self._z_hams = [
                SymbolicHamiltonian(Z(i), nqubits=spec.n_qubits)
                for i in range(spec.n_qubits)
            ]

        self._run_fn = self._make_run_fn(mode)

    def _parameters(self, sample: np.ndarray, weights: np.ndarray) -> List[float]:
        """Return the parametric gate angles in circuit order."""
        vectors = {"inputs": sample, "weights": weights}
        return [float(vectors[op.source][op.index]) for op in self._param_ops]

    # ------------------------------------------------------------------
    # Run function factory
    # ------------------------------------------------------------------
    def _make_run_fn(self, mode: Mode) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Return a callable that maps a batch of inputs to results."""

        if mode == "state":
            def _run_state(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    self._circuit.set_parameters(self._parameters(sample, w))
                    result = self._circuit()
                    results.append(np.array(result.state()))
                return jnp.array(np.stack(results))
            return _run_state

        elif mode == "probs":
            def _run_probs(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    self._circuit.set_parameters(self._parameters(sample, w))
                    result = self._circuit()
                    results.append(np.array(result.probabilities()))
                return jnp.array(np.stack(results))
            return _run_probs

        elif mode == "expval":
            def _run_expval(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    self._circuit.set_parameters(self._parameters(sample, w))
                    self._circuit()
                    # ``expectation`` reads the state the circuit just computed
                    # and contracts it with the single-qubit Z of each term, the
                    # route Qibo documents for symbolic Hamiltonians.
                    # ``expectation_from_state`` would multiply the state with
                    # the dense $2^n \times 2^n$ matrix of every observable.
                    # https://qibo.science/qibo/stable/code-examples/advancedexamples.html
                    evs = [
                        float(ham.expectation(self._circuit)) for ham in self._z_hams
                    ]
                    results.append(evs)
                return jnp.array(np.array(results))
            return _run_expval

        elif mode in ("density", "noise"):
            def _run_density(inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
                w = np.asarray(weights)
                results = []
                for sample in np.asarray(inputs):
                    self._circuit_dm.set_parameters(self._parameters(sample, w))
                    result = self._circuit_dm()
                    results.append(np.array(result.state()))
                return jnp.array(np.stack(results))
            return _run_density

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def warmup(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)

    def run(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        assert self._run_fn is not None
        return self._run_fn(inputs, weights)
