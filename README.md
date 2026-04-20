# YAQSI Benchmarking Data

This repo contains code to produce benchmarking results for [YAQSI](https://github.com/cirKITers/qml-essentials) comparing against the following quantum circuit simulators:

- **[PennyLane](https://pennylane.ai/)** — Xanadu's differentiable quantum programming framework (`default.qubit` device with JAX interface)
- **[Qiskit](https://qiskit.org/)** — IBM's quantum computing SDK (local `Statevector` / `DensityMatrix` simulation)
- **[Qibo](https://qibo.science/)** — Open-source framework for quantum simulation (numpy backend)

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
| `simulators` | `[yaqsi, pennylane, qiskit, qibo]` | Simulators to include in the run |
| `output.dir` | `results` | Directory for output CSV and plots |
| `output.identifier` | `null` | Run identifier (auto-generated timestamp if null) |

Any parameter can be overridden from the command line using dot-notation (e.g. `qubits.max=10`).

## Output

Results are written to the `results/` directory:

- **`benchmarks-<identifier>.csv`** — Raw timing data (mean and std in ms per simulator/mode/qubit-count combination)
- **`benchmarks-<identifier>-ratio.pdf`** — Time-ratio plot (each competitor vs YAQSI)
- **`benchmarks-<identifier>-absolute.pdf`** — Absolute timing plot per mode

Runs support **automatic recovery**: if a run is interrupted, re-running with the same `output.identifier` will skip already-completed combinations.

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
│       └── qibo_sim.py     # Qibo adapter
├── tests/                   # Test suite
├── slurm-job.sh             # Sample SLURM submission script
└── pyproject.toml           # Project metadata & dependencies
```

