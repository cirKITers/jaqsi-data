# YAQSI Benchmarking & Profiling Data

This repo contains code to produce benchmarking and profiling results for [YAQSI](https://github.com/cirKITers/qml-essentials) comparing against the following quantum circuit simulators:

- **[PennyLane](https://pennylane.ai/)** — Xanadu's differentiable quantum programming framework (`default.qubit` device with JAX interface)
- **[Qiskit](https://qiskit.org/)** — IBM's quantum computing SDK (local `Statevector` / `DensityMatrix` simulation)
- **[Qibo](https://qibo.science/)** — Open-source framework for quantum simulation (numpy backend)
- **[Qulacs](https://github.com/qulacs/qulacs)** — Fast C/C++ quantum circuit simulator with Python interface

## Benchmark Circuit

All simulators execute the same parametric circuit:

1. A Hadamard gate on every qubit
2. A controlled-RX rotation (`CRX(φ)`) in a ring topology: qubit *i* → qubit *(i+1) mod n*

The circuit is evaluated across four measurement modes: **probs**, **expval**, **state**, and **density**, and results are cross-validated against YAQSI as the reference.

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
git clone git@github.com:cirKITers/yaqsi-data.git
cd yaqsi-data

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

### Visualise existing results (skip computation)

```bash
uv run python -m benchmark --visualize-only results/benchmarks-<identifier>.csv
```

### Run only specific simulators

```bash
# Run only YAQSI and PennyLane
uv run python -m benchmark 'simulators=[yaqsi,pennylane]'

# Run only YAQSI and Qulacs
uv run python -m benchmark 'simulators=[yaqsi,qulacs]'
```

### Skip plot generation

```bash
uv run python -m benchmark --no-plot
```

## Configuration

The default configuration is located at [`src/benchmark/configs/default.yaml`](src/benchmark/configs/default.yaml):

| Parameter | Default | Description |
|---|---|---|
| `seed` | `1000` | RNG seed for reproducible parameter generation |
| `warmup` | `true` | Run an untimed warmup pass (triggers JIT) |
| `qubits.min` | `2` | Minimum number of qubits |
| `qubits.max` | `16` | Maximum number of qubits |
| `execution.n_iters` | `50` | Number of timed iterations per data point |
| `execution.batch_size` | `1` | Batch size for each iteration |
| `modes` | `[probs, expval, state, density]` | Measurement modes to benchmark |
| `simulators` | `[yaqsi, pennylane, qiskit, qibo, qulacs]` | Simulators to include in the run |
| `output.dir` | `results` | Directory for output CSV and plots |
| `output.identifier` | `null` | Run identifier (auto-generated timestamp if null) |

Any parameter can be overridden from the command line using dot-notation (e.g. `qubits.max=10`).

## Output

Results are written to the `benchmarking_results/` directory:

- **`benchmarks-<identifier>.csv`** — Raw timing data (mean and std in ms per simulator/mode/qubit-count combination)
- **`benchmarks-<identifier>-ratio.pdf`** — Time-ratio plot (each competitor vs YAQSI)
- **`benchmarks-<identifier>-absolute.pdf`** — Absolute timing plot per mode

Runs support **automatic recovery**: if a run is interrupted, re-running with the same `output.identifier` will skip already-completed combinations.

## YAQSI Performance Profiling

In addition to the comparative benchmarks above, this repo includes a **JAX-level profiling** module for YAQSI.  It captures execution traces using `jax.profiler` that can be inspected in [Perfetto UI](https://ui.perfetto.dev/) for detailed analysis of kernel timings, memory allocations, XLA compilation, and device utilisation.

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

- **`profiling_summary.txt`** — Human-readable summary table
- **`profiling_results.csv`** — Machine-readable timing data for re-plotting
- **`profiling-scaling.pdf`** — Execution time vs qubit count (all modes, single plot)
- **`profiling-per-mode.pdf`** — One subplot per mode showing scaling behaviour
- **`profiling-mode-comparison.pdf`** — Bar chart comparing modes at the largest qubit count
- **`yaqsi_<mode>_<n>q/`** — Perfetto trace directories (one per mode/qubit combination)

### Viewing Traces in Perfetto

1. Open [https://ui.perfetto.dev/](https://ui.perfetto.dev/) in your browser
2. Click **"Open trace file"**
3. Navigate to a trace directory (e.g. `profiling_results/yaqsi_probs_16q/`) and select the `.perfetto-trace` file
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
│   │   └── default.yaml     # Default benchmark parameters
│   └── simulators/
│       ├── base.py          # Abstract base class & timing harness
│       ├── yaqsi_sim.py     # YAQSI adapter (reference)
│       ├── pennylane_sim.py # PennyLane adapter
│       ├── qiskit_sim.py   # Qiskit adapter
│       ├── qibo_sim.py     # Qibo adapter
│       └── qulacs_sim.py   # Qulacs adapter
├── src/profiling/
│   ├── __main__.py          # CLI entry-point (python -m profiling)
│   ├── config.py            # Profiling configuration dataclass
│   ├── profiler.py          # JAX profiler integration & trace capture
│   └── visualize.py         # Publication-quality plot generation
├── tests/                   # Test suite
├── slurm-job.sh             # Sample SLURM submission script
└── pyproject.toml           # Project metadata & dependencies
```

