# JAQSI Benchmarking and Profiling

This repository benchmarks [JAQSI](https://github.com/cirKITers/jaqsi) against other quantum circuit simulators and profiles JAQSI's JAX execution.
The benchmark adapters run shared gate-level or pulse-level circuits, compare their outputs for agreement, and record execution times.

## Simulators

| Level | Simulator adapters |
| --- | --- |
| Gate | `jaqsi`, `pennylane`, `pennylane_lightning`, `pennylane_psr`, `qiskit`, `qibo`, `qulacs` |
| Pulse | `jaqsi_pulse`, `pennylane_pulse`, `qutip_pulse`, `dynamiqs_pulse` |

The gate adapters support different subsets of the available workloads: probabilities, expectation values, state vectors, density matrices, noise, and gradients.
The default gate run omits PennyLane's parameter-shift adapter (`pennylane_psr`); select it explicitly if needed.
Pulse adapters use names ending in `_pulse` and are compared against `jaqsi_pulse`.

## Getting started

The project requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
From the repository root, install dependencies and run the default gate benchmark:

```sh
uv sync
uv run python -m benchmark
```

Run the pulse benchmark or profile JAQSI with:

```sh
uv run python -m benchmark --config src/benchmark/configs/pulse.yaml
uv run python -m profiling
```

The profiler writes timing data, plots, and [Perfetto](https://ui.perfetto.dev/) traces to `profiling_results/` by default.
Run the test suite with `uv run pytest`.

## Configuring benchmarks

The default run uses [`src/benchmark/configs/default.yaml`](src/benchmark/configs/default.yaml).
Pass another YAML file with `--config`, or override individual settings on the command line using dot notation:

```sh
uv run python -m benchmark qubits.max=8 execution.n_iters=20
uv run python -m benchmark 'simulators=[jaqsi,pennylane]' 'modes=[expval,grad]'
```

The YAML files in [`src/benchmark/configs/`](src/benchmark/configs/) show the available settings, including circuit family and depth, qubit range, batch size, workloads, simulators, thread count, and output directory.
Command-line overrides take precedence over the YAML file.
Set `output.identifier` to reuse an identifier when resuming an interrupted run.
Use `--no-plot` to collect timings without generating plots.

## Results and visualization

The supplied benchmark configurations write to `results/` by default; set `output.dir` to change the location.
Each run produces a `benchmarks-<identifier>-<commit>.csv` timing file and a matching `.json` file with configuration and environment details.
Unless `--no-plot` is set, the benchmark also writes PGF plots of absolute times and time ratios relative to JAQSI, plus an infidelity plot when pulse data is available.

To regenerate plots from an existing CSV without rerunning the benchmark:

```sh
uv run python -m benchmark --visualize-only path/to/benchmarks.csv
```

Plots are written beside the CSV.
Use `--show` to display them interactively.
TODO: Allow the visualization command to select a circuit, depth, batch size, or thread count when a CSV contains several; it currently plots the first slice.
