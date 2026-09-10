# JAQSI Benchmarking & Profiling Data

This repo contains code to produce benchmarking and profiling results for [JAQSI](https://github.com/cirKITers/jaqsi) comparing against the following quantum circuit simulators:

- [PennyLane](https://github.com/PennyLaneAI/pennylane) — Xanadu's differentiable quantum programming framework (`default.qubit` device with JAX interface)
- [Qiskit](https://github.com/Qiskit/qiskit) — IBM's quantum computing SDK (local `Statevector` / `DensityMatrix` simulation)
- [Qibo](https://github.com/qiboteam/qibo) — Open-source framework for quantum simulation (numpy backend)
- [Qulacs](https://github.com/qulacs/qulacs) — Fast C/C++ quantum circuit simulator with Python interface

JAQSI also simulates at pulse level, which is benchmarked separately against the simulators that offer time-dependent Hamiltonian evolution:

- [PennyLane](https://github.com/PennyLaneAI/pennylane) — `qml.pulse` with `ParametrizedEvolution` on `default.qubit`, or `jax.experimental.ode` directly under `optimal_config`
- [QuTiP](https://github.com/qutip/qutip) — `sesolve` on a `QobjEvo` Hamiltonian
- [dynamiqs](https://github.com/dynamiqs/dynamiqs) — `sesolve` with JAX and Diffrax solvers

Qiskit is absent from that list because pulse support was removed in Qiskit 2.0 and the successor package pins `qiskit<=1.3`. 
Qulacs has no pulse-level interface, and Qibo exposes it only through the Qibolab hardware emulator.

## Benchmark Circuits

Circuits are defined once, backend-agnostically, in [`src/benchmark/circuits.py`](src/benchmark/circuits.py) as a list of gates with the parameter each rotation reads.
Every adapter builds its circuit by walking that list, so all simulators provably execute the same gate sequence with the same angles rather than each restating the circuit in its own API.

Two families are available:

| Family | Structure | Parameters |
|---|---|---|
| `hea` | An `RX` data-encoding layer, then `layers` blocks of rotations on every wire followed by a `CNOT` ring. Blocks are `RZ · RX · RZ`, except the last, which is `RZ · RX` | `n` inputs, `3 · layers · n − n` weights |
| `crx_ring` | A Hadamard layer, then `layers` rings of `CRX`: qubit *i* → qubit *(i+1) mod n* | `layers · n` inputs, no weights |

`hea` is the rotation-layer/entangler-ring circuit the Yao and Qulacs benchmarks use, with the encoding layer that turns the batch axis into a data axis.
The last block ends on `RZ · RX`, following that convention: a trailing `RZ` could not move a Pauli-Z expectation value, since it only adds phases and the `CNOT` ring after it permutes the computational basis, so keeping it would leave `n` weights with an exactly zero gradient.
The leading `RZ` of the first block is kept, unlike in Yao and Qulacs, because the encoding layer has already rotated the state off the Z axis.
`crx_ring` is the circuit the pulse-level benchmark runs, and the only one [`pulse_model`](src/benchmark/simulators/pulse_model.py) transcribes.
Every rotation carries its own angle, drawn uniformly from $[-\pi, \pi]$ with a shared seed so all simulators see identical parameters.
Neither family is Clifford, and no angle repeats, so nothing can be precomputed across gates or across the sweep.

`circuit.layers` accepts a list, which sweeps depth at every qubit count.
Sweeping `qubits` at fixed depth and `layers` at fixed width from the same config separates the two scaling directions.

Batch size is one value per run, but the recovery key includes it, so a batch sweep accumulates in one results file by re-running with the same identifier:

```bash
for b in 1 10 100 1000; do
    python -m benchmark output.identifier=batchsweep execution.batch_size=$b \
        qubits.min=10 qubits.max=10 circuit.layers=[4]
done
```

### Parameters, batching and the gradient

Parameters are split into two flat vectors.
`inputs` carries the batch axis, one vector per sample; `weights` are shared across the batch.
This is the parameter layout of a QML training step: a batch of data evaluated against one set of trainable parameters.
Weights are redrawn every timed iteration, so no simulator can cache across the loop.

The measurement modes probs, expval, state and density are joined by `grad`, which is not a measurement but a differentiation workload: the gradient of $\sum_i \langle Z_i \rangle$, summed over the batch, with respect to the circuit's trainable vector.
Each framework reaches it by its own route, and the adapters are named after the method so a gradient figure says which one it plots:

| Adapter | Device | Differentiation |
|---|---|---|
| `jaqsi` | JAX | reverse-mode AD through the vmapped kernel |
| `pennylane` | `default.qubit` | backpropagation |
| `pennylane_adjoint` | `lightning.qubit` | adjoint method |
| `pennylane_psr` | `default.qubit` | parameter-shift rule |
| `qulacs` | Qulacs | `ParametricQuantumCircuit.backprop` |

Qiskit and Qibo contribute forward measurements only: their gradient interfaces live in the separate `qiskit-algorithms` and `qiboml` packages, which are not dependencies here.
Qulacs differentiates `hea` but not `crx_ring`, because it builds controlled rotations as dense matrices with an attached control qubit, which it cannot register as a parametric gate.
The runner skips and logs any unsupported combination rather than failing, so every simulator still contributes what it can.

`pennylane_psr` contributes gradient rows only, since its forward pass is the `pennylane` one.
It is also left out of the default simulator list: the parameter-shift rule costs two circuit evaluations per parameter, so at the top of the default sweep it is roughly 500 times a single forward pass and would dominate the wall clock.
Run it as its own small sweep instead:

```bash
python -m benchmark simulators=[jaqsi,pennylane,pennylane_adjoint,pennylane_psr] \
    modes=[grad] qubits.max=8 circuit.layers=[1,2] execution.n_iters=20
```

All modes are cross-validated against JAQSI as the reference: forward results to `1e-8`, gradients to `1e-6`, since reverse-mode AD, the adjoint method and Qulacs' backprop accumulate their sums in different orders.

Batching is asymmetric by design and is left that way rather than normalized.
JAQSI vectorizes the batch with `jax.vmap`; PennyLane uses its own parameter broadcasting; Qiskit, Qibo and Qulacs loop over the batch in Python, and so do PennyLane's adjoint and parameter-shift paths, which do not accept broadcast inputs.
That difference is a property of the frameworks, and it is what a QML workload actually pays.

`lightning.qubit` is called through PennyLane's native interface rather than the JAX one.
It is a compiled device and cannot be traced, so JAX routes every call through a host callback, which costs roughly twice the native call across the whole sweep and would measure the bridge instead of the simulator.

### Threads

Every simulator is pinned to `threads` before any numerical library is imported, since each of them sizes its pool once at import time.
Pinning narrows the process CPU affinity first, which is the one setting OpenMP, the BLAS pool, numba and XLA all agree to read, and sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `QULACS_NUM_THREADS` on top of it.
Cores are taken from the set already permitted, so a narrower allocation from a batch scheduler is respected rather than overridden.

This is not cosmetic. qibojit sizes numba from `len(psutil.Process().cpu_affinity())` in its backend constructor and ignores every environment variable, so without pinning it takes one thread per core while nothing else does. On a 16-core machine that moved its two-qubit `expval` time from 14 ms to 2.6 ms once pinned to one thread, because thread coordination dominates at small register sizes.

Qibo is the one backend that cannot be pinned through the environment at all: qibojit sizes numba inside its constructor, so the adapter calls `qibo.set_threads` afterwards. The numpy backend is skipped there, because it is single-threaded by construction and raises for any count above one.

Neither Qibo backend wins everywhere, so `optimal_config` picks per mode: qibojit for `density`, numpy for the state-vector modes. Measured on the hardware-efficient ansatz at four layers, batch 10, sixteen threads:

| Mode | Qubits | numpy | qibojit | Winner |
|---|---|---|---|---|
| expval | 8 | 47 ms | 934 ms | numpy, 20x |
| state | 8 | 39 ms | 470 ms | numpy, 12x |
| density | 9 | 14182 ms | 1078 ms | qibojit, 13x |
| density | 10 | 66491 ms | 2452 ms | qibojit, 27x |

qibojit parallelises its kernels with numba, which pays off once the operand is large enough to amortise the launches. A density matrix holds $4^n$ entries against a state vector's $2^n$, so it crosses that point while the state-vector modes never do. Committing to either backend for the whole sweep costs an order of magnitude on half of it. The Qiskit adapter selects its Aer method by mode for the same reason.

The margins shift with thread count. At one thread qibojit is only 2x behind on the state-vector modes and 5.4x ahead on density at eight qubits; at sixteen the state-vector gap widens to 20x. The split above holds in both regimes, but re-measure before trusting a single backend anywhere.

`NUMBA_NUM_THREADS` is deliberately not set. It is a hard cap, and qibojit's constructor requests one thread per core it can see, so a lower cap makes numba raise and the backend fails to build. Narrowing the affinity changes what that constructor sees instead.

The default is one thread, the single-thread regime Yao, Qulacs and JuliVQC report. Set `threads` to the core count for the multi-threaded regime and report the two separately; the value is recorded per row, and it is part of the recovery key, so both regimes can accumulate in one results file.

## Pulse-Level Benchmark Circuit

The pulse-level benchmark runs the `crx_ring` circuit, but every gate is expanded into the sequence of time evolutions $\mathrm{d}U/\mathrm{d}t = -i H(t) U$ that JAQSI's `PulseGates` execute: $3n$ segments for the Hadamard layer plus $18n$ for every $CRX$ ring, i.e. $21n$ at the default depth of one.
That schedule is transcribed once in [`src/benchmark/simulators/pulse_model.py`](src/benchmark/simulators/pulse_model.py) and rebuilt from there by each adapter, so all simulators integrate the same segments with the same coefficients and durations rather than their own pulse model.

The transcription covers the shipped JAQSI defaults, i.e. the drag envelope with the rotating-wave approximation enabled, under which the driven rotations evolve under

$$ H(t) = \tfrac{1}{2}\,\Omega(t)\,w\,P, \qquad \Omega(t) = A e^{-t^2/(8\sigma^2)}\left(1 - \frac{\beta t}{2\sigma^2}\right) $$

while the virtual $RZ$, the $CZ$ coupling and the Hadamard correction phase evolve under a constant $H$.

Pulse results carry each backend's ODE solver error, so they are cross-validated against `jaqsi_pulse` at a solver-limited tolerance.
That error accumulates over the $21n$ solves run in sequence, and `expval` amplifies it because it sums $2^n$ probabilities, which makes it the mode that sets the tolerance: at 8 qubits the full-register path deviates by $1.2 \cdot 10^{-6}$ there against $3 \cdot 10^{-8}$ in `probs`.
Widening the qubit sweep therefore needs a looser `precision`.

As at gate level, `optimal_config` selects each simulator's performance-optimized configuration.
At pulse level the two configurations differ in the dimension of the integrated ODE, which dominates the cost:

- Default: QuTiP and dynamiqs embed each segment operator into the full register and evolve the $2^n$-dimensional statevector with `sesolve`, the idiomatic formulation of a pulse schedule. PennyLane uses `qml.evolve`, whose per-operation overhead costs roughly 190 ms per segment independently of solver tolerance and dominates everything else.
- Optimized: each segment is integrated on its own $2 \times 2$ or $4 \times 4$ space and the resulting local propagator is contracted into the statevector, via `qutip.propagator`, `dynamiqs.sepropagator` and `jax.experimental.ode` respectively. JAQSI composes its pulse gates this way by construction, so only this configuration compares like with like.

Both configurations agree within the solver-limited tolerance and are cross-checked against each other in the test suite.
Comparing against the default configuration therefore measures an architectural difference, local versus full-register evolution, on top of the implementation difference; comparing against the optimized configuration isolates the implementation.
Reporting both separates the two effects.

Two asymmetries are left in place rather than normalized.
JAQSI and dynamiqs vectorize the batch dimension with `jax.vmap` while PennyLane and QuTiP loop over it in Python, so the latter two pay the full batch factor.
PennyLane's optimized path integrates with Dormand-Prince 5(4), the method `ParametrizedEvolution` is built on, while the others use Dormand-Prince 8(7); all four run at $10^{-10}$ absolute and relative tolerance.

Timings block on the returned array before the clock is stopped, so the JAX-based adapters measure the completed computation rather than the asynchronous dispatch that returns immediately.

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
git clone git@github.com:cirKITers/jaqsi-data.git
cd jaqsi-data

# Install dependencies via uv
uv sync
```

## Usage

### Run with default configuration

```bash
uv run python -m benchmark
```

### Run with a custom config file

```bash
uv run python -m benchmark --config path/to/config.yaml
```

### Override specific parameters

```bash
uv run python -m benchmark qubits.max=10 execution.n_iters=20
```

### Run with optimized simulator configurations

By default each competitor simulator runs in its baseline configuration.
Setting `optimal_config=true` switches them to performance-optimized configurations (PennyLane: `jax.jit`-compiled QNode; Qiskit: qiskit-aer C++ simulator; Qibo: qibojit for density and numpy for the state-vector modes; Qulacs: gate-fusion via `QuantumCircuitOptimizer`). 
These configurations are numerically equivalent to the defaults.
JAQSI is unaffected.

```bash
uv run python -m benchmark optimal_config=true
```

### Choose the threading regime

`threads` pins every simulator, so exporting `QULACS_NUM_THREADS` or
`OMP_NUM_THREADS` before launch has no effect: the configuration overwrites
them. Set the config value instead.

```bash
# Single-thread regime (default)
uv run python -m benchmark threads=1

# Multi-threaded regime, reported separately
uv run python -m benchmark threads=8
```

### Visualise existing results (skip computation)

```bash
uv run python -m benchmark --visualize-only results/benchmarks-<identifier>-<commit>.csv
```

### Run only specific simulators

```bash
# Run only JAQSI and PennyLane
uv run python -m benchmark 'simulators=[jaqsi,pennylane]'

# Run only JAQSI and Qulacs
uv run python -m benchmark 'simulators=[jaqsi,qulacs]'
```

### Skip plot generation

```bash
uv run python -m benchmark --no-plot
```

### Run the pulse-level benchmark

```bash
uv run python -m benchmark --config src/benchmark/configs/pulse.yaml
```

The pulse configuration sweeps a smaller qubit range than the gate benchmark, because the schedule expands to $21n$ ODE segments that are solved sequentially.

## Configuration

The default configuration is located at [`src/benchmark/configs/default.yaml`](src/benchmark/configs/default.yaml):

| Parameter | Default | Description |
|---|---|---|
| `seed` | `1000` | RNG seed for reproducible parameter generation |
| `warmup` | `true` | Run an untimed warmup pass (triggers JIT) |
| `threads` | `1` | Threads every simulator is pinned to |
| `precision` | `1.0e-8` | Cross-validation tolerance for the forward modes |
| `optimal_config` | `true` | Use performance-optimized simulator configurations instead of the default fallback |
| `circuit.family` | `hea` | Circuit family: `hea` or `crx_ring` |
| `circuit.layers` | `[1, 2, 4, 8]` | Depths to sweep at every qubit count |
| `qubits.min` | `2` | Minimum number of qubits |
| `qubits.max` | `10` | Maximum number of qubits |
| `execution.n_iters` | `100` | Number of timed iterations per data point |
| `execution.batch_size` | `10` | Number of input samples per iteration |
| `modes` | `[expval, state, density, grad]` | Measurement modes, plus the gradient workload |
| `simulators` | `[jaqsi, pennylane, pennylane_adjoint, pennylane_psr, qiskit, qibo, qulacs]` | Simulators to include in the run |
| `output.dir` | `results` | Directory for output CSV and plots |
| `output.identifier` | `null` | Run identifier (auto-generated timestamp if null) |

Any parameter can be overridden from the command line using dot-notation (e.g. `qubits.max=10` or `circuit.layers=[1,4]`).

The gradient tolerance is not configurable; it is fixed at `1e-6` in `benchmark.runner.GRAD_PRECISION`.

The pulse-level configuration at [`src/benchmark/configs/pulse.yaml`](src/benchmark/configs/pulse.yaml) uses the same parameters with `circuit.family` set to `crx_ring`, `simulators` set to `[jaqsi_pulse, pennylane_pulse, qutip_pulse, dynamiqs_pulse]` and a looser `precision` of `1.0e-5`. Simulator names ending in `_pulse` run at pulse level and are cross-validated against `jaqsi_pulse`; the two levels are never compared against each other.

## Output

Results are written to the `benchmarking_results/` directory:

- `benchmarks-<identifier>-<commit>.csv` — Raw timing data (mean and std in ms per circuit, depth, qubit count, mode and simulator)
- `benchmarks-<identifier>-<commit>-ratio.pdf` — Time-ratio plot (each competitor vs JAQSI)
- `benchmarks-<identifier>-<commit>-absolute.pdf` — Absolute timing plot per mode
- `benchmarks-<identifier>-<commit>-infidelity.pdf` — Pulse infidelity per mode (pulse-level runs only)

`<commit>` is the short git commit the run started from, with a `-dirty` suffix when tracked files were modified.
It tags the file rather than a column so that a results file identifies its own provenance once it is copied off the machine, and so that a run started after a code change lands in its own file instead of resuming into results produced by different code.

Runs support automatic recovery: if a run is interrupted, re-running with the same `output.identifier` will skip already-completed combinations, keyed by circuit, depth, qubit count, mode and simulator.

The figures plot runtime against qubit count, so a results file sweeping several depths has to be narrowed to one slice before plotting.
`load_results` takes optional `circuit`, `n_layers` and `batch_size` filters; without them it keeps the first slice in the file and logs which other slices it ignored.

### Pulse infidelity

Pulse-level rows carry an additional `infidelity` column, reporting $1 - F$ between the pulse result and the gate-level circuit the pulses implement. It is logged during the run and written per simulator, mode and qubit count, which makes the sensitivity of a pulse simulation to its calibrated parameters and its solver accuracy visible alongside the timings.

Each pulse simulator is scored against its own framework's gate-level adapter where one exists (`jaqsi_pulse` against `jaqsi`, `pennylane_pulse` against `pennylane`); QuTiP and dynamiqs have no gate-level adapter here and fall back to `jaqsi`. The gate-level simulators agree to an infidelity of order $10^{-15}$, so that choice does not affect the reported value.

The fidelity is normalised by the norms of both operands. Without that normalisation the ODE solvers' norm drift, which reaches $10^{-7}$ for the looser configurations, dominates the result and can even push $1 - F$ negative. Because infidelity grows with the square of the state error, an amplitude deviation of $10^{-7}$ registers as roughly $10^{-14}$, close to the double-precision floor.

The column is written for the `probs`, `state` and `density` modes and left empty for `expval`, whose output is not a state. In `probs` mode it is the classical fidelity of the measurement distributions; in `state` and `density` mode it is the quantum state fidelity.

The infidelity plot shows $1 - F$ against the qubit count, one subplot per mode, on a logarithmic axis. Modes without a recorded infidelity are omitted rather than drawn empty, so a gate-level run produces no such figure. Values at or below the double-precision floor $\varepsilon = 2.22 \times 10^{-16}$ are clipped to it, marked by a dashed line, since a pulse result that matches the reference exactly cannot be placed on a logarithmic axis.

## JAQSI Performance Profiling

In addition to the comparative benchmarks above, this repo includes a JAX-level profiling module for JAQSI.  It captures execution traces using `jax.profiler` that can be inspected in [Perfetto UI](https://ui.perfetto.dev/) for detailed analysis of kernel timings, memory allocations, XLA compilation, and device utilisation.

### Run profiling with default settings

```bash
uv run python -m profiling
```

### Profile specific modes and qubit counts

```bash
uv run python -m profiling --modes probs expval --qubits 4 8 16
```

### Customise warm-up, profiled runs and output

```bash
uv run python -m profiling --warmup 3 --runs 5 --batch-size 4 --output ./traces
```

### Skip plot generation

```bash
uv run python -m profiling --no-plot
```

### Re-generate plots from existing results (skip profiling)

```bash
uv run python -m profiling --visualize-only profiling_results/profiling_results.csv
```

### Profiling Configuration

| Parameter | Default | Description |
|---|---|---|
| `--qubits` | `2 4 8 12 16` | Qubit counts to profile |
| `--modes` | `probs expval state density` | Measurement modes to profile |
| `--batch-size` | `1` | Batch size for each execution |
| `--warmup` | `2` | Number of warm-up runs (triggers JIT, excluded from trace) |
| `--runs` | `3` | Number of profiled simulation runs |
| `--seed` | `1000` | Random seed for parameter generation |
| `--output` | `profiling_results` | Output directory for traces, CSV, and plots |

### Profiling Output

Results are written to the `profiling_results/` directory:

- `profiling_summary.txt` — Human-readable summary table
- `profiling_results.csv` — Machine-readable timing data for re-plotting
- `profiling-scaling.pdf` — Execution time vs qubit count (all modes, single plot)
- `profiling-per-mode.pdf` — One subplot per mode showing scaling behaviour
- `profiling-mode-comparison.pdf` — Bar chart comparing modes at the largest qubit count
- `jaqsi_<mode>_<n>q/` — Perfetto trace directories (one per mode/qubit combination)

### Viewing Traces in Perfetto

1. Open [https://ui.perfetto.dev/](https://ui.perfetto.dev/) in your browser
2. Click "Open trace file"
3. Navigate to a trace directory (e.g. `profiling_results/jaqsi_probs_16q/`) and select the `.perfetto-trace` file
4. Explore the timeline view to inspect JAX/XLA kernel execution, memory transfers, and compilation events

## Running Tests

```bash
uv run pytest
```

## Docker

[`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml) run
the sweeps against a fixed slice of the local machine.

```bash
cp .env.example .env

# Pick the cpuset for this machine and paste it into .env
lscpu -e=CPU,CORE,SOCKET,NODE | python3 scripts/pick_cpuset.py --cpus 256 --whole-nodes

docker compose build
docker compose run --rm benchmark                     # single-thread regime
docker compose run --rm benchmark threads=240         # multi-threaded regime
docker compose run --rm benchmark-pulse               # pulse-level sweep
docker compose run --rm profiling
docker compose run --rm tests
```

Arguments after the service name are appended to the entrypoint, so every
override and flag under [Usage](#usage) works unchanged.
The repository is bind-mounted at `/app` and the image holds only the locked
dependencies, so results land in the working tree exactly where a native
`uv run python -m benchmark` puts them, owned by the `UID`/`GID` from `.env`,
and an edit takes effect on the next run — only a change to `pyproject.toml` or
`uv.lock` needs `docker compose build` again.

### The slice

`BENCH_CPUSET` is the container's `cpuset`, an explicit list of host CPUs, and
`BENCH_MEM` is its `mem_reservation`, `mem_limit` and `memswap_limit` at once,
so it never swaps.
Docker's size suffixes stop at `g`, so 4 TB is `4096g`, and exceeding it is an
OOM kill rather than a slowdown — `density` sets the requirement, at $2^{2n}$
complex amplitudes per copy and several copies per batch.

The cpuset fixes what the run may use, not what else may use those cores, so on
a shared machine a sweep is comparable against itself rather than against one
taken on a quiet host.
It is also the set [`threads`](#threads) pins within, since `pin_threads` takes
its cores from what the process is already permitted; a multi-threaded run
should set `threads` to the size of the cpuset.

[`scripts/pick_cpuset.py`](scripts/pick_cpuset.py) chooses the value, because
none of the three properties that matter is visible in a range typed from the
core count: it takes whole physical cores, whose SMT siblings are rarely
adjacent; spans as few NUMA nodes as possible, since a statevector spread across
nodes pays remote-memory latency on every gate; and under `--whole-nodes` takes
no node in part, so the allocation does not share memory bandwidth with whatever
lands on the rest of one.
It can therefore fall short of `--cpus`, and says so: on a 384-CPU host of 8
nodes with 24 SMT-2 cores each, 256 gives 240 — nodes 1 to 5 entire — leaving
node 0, which holds the host's CPU 0, and nodes 6 and 7 to the machine.

## SLURM

A sample SLURM job script is provided for HPC clusters:

```bash
sbatch slurm-job.sh
```

## Project Structure

```
├── profile.py               # Standalone profiling entry-point
├── src/benchmark/
│   ├── __main__.py          # CLI entry-point
│   ├── circuits.py          # Backend-agnostic circuit definitions
│   ├── config.py            # OmegaConf-based configuration loader
│   ├── threads.py           # Process-wide thread pinning
│   ├── runner.py            # Benchmark runner with CSV recovery
│   ├── visualize.py         # Plotting and result aggregation
│   ├── configs/
│   │   ├── default.yaml     # Default benchmark parameters
│   │   └── pulse.yaml       # Pulse-level benchmark parameters
│   └── simulators/
│       ├── base.py          # Abstract base class & timing harness
│       ├── jaqsi_sim.py     # JAQSI adapter (reference)
│       ├── pennylane_sim.py # PennyLane adapters (backprop, adjoint, param-shift)
│       ├── qiskit_sim.py   # Qiskit adapter
│       ├── qibo_sim.py     # Qibo adapter
│       ├── qulacs_sim.py   # Qulacs adapter
│       ├── pulse_model.py           # Pulse schedule transcribed from JAQSI
│       ├── jaqsi_pulse_sim.py       # JAQSI pulse adapter (pulse reference)
│       ├── pennylane_pulse_sim.py   # PennyLane pulse adapter
│       ├── qutip_pulse_sim.py       # QuTiP pulse adapter
│       └── dynamiqs_pulse_sim.py    # dynamiqs pulse adapter
├── src/profiling/
│   ├── __main__.py          # CLI entry-point (python -m profiling)
│   ├── config.py            # Profiling configuration dataclass
│   ├── profiler.py          # JAX profiler integration & trace capture
│   └── visualize.py         # Publication-quality plot generation
├── tests/                   # Test suite
├── scripts/
│   └── pick_cpuset.py       # Whole-core, NUMA-aligned cpuset from lscpu output
├── slurm-job.sh             # Sample SLURM submission script
├── Dockerfile               # Locked dependencies; the source comes from the mount
├── docker-compose.yml       # Core and memory slice the container runs on
├── .env.example             # Slice settings to copy to .env
└── pyproject.toml           # Project metadata & dependencies
```



## Roadmap

- [x] first implementation with four different simulators
    - pennylane
    - qiskit
    - qibo
    - qulacs
- [ ] run an initial benchmark and confirm none is faster
- [ ] improve the implementation of the existing simulators to get the best performance
- [ ] run another benchmark
