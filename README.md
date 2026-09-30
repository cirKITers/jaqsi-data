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
