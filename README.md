# JAQSI Benchmarking & Profiling Data

This repo contains code to produce benchmarking and profiling results for [JAQSI](https://github.com/cirKITers/qml-essentials) comparing against the following quantum circuit simulators:

- [PennyLane](https://github.com/PennyLaneAI/pennylane) — Xanadu's differentiable quantum programming framework (`default.qubit` device with JAX interface)
- [Qiskit](https://github.com/Qiskit/qiskit) — IBM's quantum computing SDK (local `Statevector` / `DensityMatrix` simulation)
- [Qibo](https://github.com/qiboteam/qibo) — Open-source framework for quantum simulation (numpy backend)
- [Qulacs](https://github.com/qulacs/qulacs) — Fast C/C++ quantum circuit simulator with Python interface

JAQSI also simulates at pulse level, which is benchmarked separately against the simulators that offer time-dependent Hamiltonian evolution:

- [PennyLane](https://github.com/PennyLaneAI/pennylane) — `qml.pulse` with `ParametrizedEvolution` on `default.qubit`, or `jax.experimental.ode` directly under `optimal_config`
- [QuTiP](https://github.com/qutip/qutip) — `sesolve` on a `QobjEvo` Hamiltonian
- [dynamiqs](https://github.com/dynamiqs/dynamiqs) — `sesolve` with JAX and Diffrax solvers

Qiskit is absent from that list because pulse support was removed in Qiskit 2.0 and the successor package pins `qiskit<=1.3`. Qulacs has no pulse-level interface, and Qibo exposes it only through the Qibolab hardware emulator.

## Benchmark Circuit

All simulators execute the same parametric circuit:

1. A Hadamard gate on every qubit
2. A controlled-RX rotation (`CRX(φ)`) in a ring topology: qubit *i* → qubit *(i+1) mod n*

The circuit is evaluated across four measurement modes: probs, expval, state, and density, and results are cross-validated against JAQSI as the reference.

## Pulse-Level Benchmark Circuit

The pulse-level benchmark runs the same circuit, but every gate is expanded into the sequence of time evolutions $\mathrm{d}U/\mathrm{d}t = -i H(t) U$ that JAQSI's `PulseGates` execute, giving $21n$ segments for $n$ qubits. That schedule is transcribed once in [`src/benchmark/simulators/pulse_model.py`](src/benchmark/simulators/pulse_model.py) and rebuilt from there by each adapter, so all simulators integrate an identical sequence of ODEs rather than their own pulse model.

The transcription covers the shipped JAQSI defaults, i.e. the drag envelope with the rotating-wave approximation enabled, under which the driven rotations evolve under

$$ H(t) = \tfrac{1}{2}\,\Omega(t)\,w\,P, \qquad \Omega(t) = A e^{-t^2/(8\sigma^2)}\left(1 - \frac{\beta t}{2\sigma^2}\right) $$

while the virtual $RZ$, the $CZ$ coupling and the Hadamard correction phase evolve under a constant $H$.

Pulse results carry each backend's ODE solver error, so they are cross-validated against `jaqsi_pulse` at a solver-limited tolerance rather than against the exact gate-level results.

As at gate level, `optimal_config` selects each simulator's performance-optimized configuration. For PennyLane the two configurations differ more than elsewhere: the default path uses `qml.evolve`, whose per-operation overhead costs roughly 190 ms per segment independently of solver tolerance, while the optimized path integrates the same segments with `jax.experimental.ode` (the solver `ParametrizedEvolution` is built on) and applies the resulting local unitaries as gates. Both produce identical results to within $10^{-10}$, and the optimized path is two to three orders of magnitude faster, but it no longer exercises the `qml.pulse` API itself. Set `optimal_config=false` to measure that API as such.

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

By default each competitor simulator runs in its baseline configuration. Setting
`optimal_config=true` switches them to performance-optimized configurations
(PennyLane: `jax.jit`-compiled QNode; Qiskit: qiskit-aer C++ simulator; Qibo:
qibojit numba backend; Qulacs: gate-fusion via `QuantumCircuitOptimizer`). These
configurations are numerically equivalent to the defaults. JAQSI is unaffected.

```bash
uv run python -m benchmark optimal_config=true
```

Qulacs also honours the `QULACS_NUM_THREADS` environment variable for CPU
parallelism; it must be set before launch:

```bash
QULACS_NUM_THREADS=8 uv run python -m benchmark optimal_config=true
```

### Visualise existing results (skip computation)

```bash
uv run python -m benchmark --visualize-only results/benchmarks-<identifier>.csv
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
| `optimal_config` | `false` | Use performance-optimized simulator configurations instead of the default fallback |
| `qubits.min` | `2` | Minimum number of qubits |
| `qubits.max` | `16` | Maximum number of qubits |
| `execution.n_iters` | `50` | Number of timed iterations per data point |
| `execution.batch_size` | `1` | Batch size for each iteration |
| `modes` | `[probs, expval, state, density]` | Measurement modes to benchmark |
| `simulators` | `[jaqsi, pennylane, qiskit, qibo, qulacs]` | Simulators to include in the run |
| `output.dir` | `results` | Directory for output CSV and plots |
| `output.identifier` | `null` | Run identifier (auto-generated timestamp if null) |

Any parameter can be overridden from the command line using dot-notation (e.g. `qubits.max=10`).

The pulse-level configuration at [`src/benchmark/configs/pulse.yaml`](src/benchmark/configs/pulse.yaml) uses the same parameters with `simulators` set to `[jaqsi_pulse, pennylane_pulse, qutip_pulse, dynamiqs_pulse]` and a looser `precision` of `1.0e-6`. Simulator names ending in `_pulse` run at pulse level and are cross-validated against `jaqsi_pulse`; the two levels are never compared against each other.

## Output

Results are written to the `benchmarking_results/` directory:

- `benchmarks-<identifier>.csv` — Raw timing data (mean and std in ms per simulator/mode/qubit-count combination)
- `benchmarks-<identifier>-ratio.pdf` — Time-ratio plot (each competitor vs JAQSI)
- `benchmarks-<identifier>-absolute.pdf` — Absolute timing plot per mode

Runs support automatic recovery: if a run is interrupted, re-running with the same `output.identifier` will skip already-completed combinations.

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
│   ├── config.py            # OmegaConf-based configuration loader
│   ├── runner.py            # Benchmark runner with CSV recovery
│   ├── visualize.py         # Plotting and result aggregation
│   ├── configs/
│   │   ├── default.yaml     # Default benchmark parameters
│   │   └── pulse.yaml       # Pulse-level benchmark parameters
│   └── simulators/
│       ├── base.py          # Abstract base class & timing harness
│       ├── jaqsi_sim.py     # JAQSI adapter (reference)
│       ├── pennylane_sim.py # PennyLane adapter
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
├── slurm-job.sh             # Sample SLURM submission script
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