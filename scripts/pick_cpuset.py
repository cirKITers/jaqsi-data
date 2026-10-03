#!/usr/bin/env python3
"""Select a benchmark CPU set from host topology.

Run on the benchmark host with
``lscpu -e=CPU,CORE,SOCKET,NODE | python3 scripts/pick_cpuset.py --cpus 256``.
Selection keeps SMT siblings together and uses as few NUMA nodes as
possible; ``--whole-nodes`` excludes partial nodes. The script uses only
the standard library.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

# (node, socket, core) -> the logical CPUs of that physical core.
Cores = Dict[Tuple[int, int, int], List[int]]


def parse_topology(lines: Iterable[str]) -> Cores:
    """Group CPUs by physical core from ``lscpu -e`` or ``-p`` output.

    Skip headers, comments, and rows with nonnumeric topology fields.
    """
    cores: Cores = defaultdict(list)
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",") if "," in line else line.split()
        fields = [f.strip() for f in fields[:4]]
        # The node column is absent or empty on a machine without NUMA info.
        if len(fields) == 3:
            fields.append("0")
        if len(fields) < 4 or not all(f.lstrip("-").isdigit() for f in fields):
            continue
        cpu, core, socket, node = (int(f) for f in fields)
        cores[(node, socket, core)].append(cpu)
    if not cores:
        raise SystemExit(
            "no CPUs found; expected lscpu -e=CPU,CORE,SOCKET,NODE on stdin"
        )
    for cpus in cores.values():
        cpus.sort()
    return dict(cores)


def compress(cpus: Iterable[int]) -> str:
    """Format CPU indices as Docker ``--cpuset-cpus`` ranges."""
    ordered = sorted(cpus)
    if not ordered:
        return ""
    ranges: List[Tuple[int, int]] = [(ordered[0], ordered[0])]
    for cpu in ordered[1:]:
        start, end = ranges[-1]
        if cpu == end + 1:
            ranges[-1] = (start, cpu)
        else:
            ranges.append((cpu, cpu))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def by_node(
    cores: Cores, exclude: Iterable[int], whole_nodes: bool = False
) -> Dict[int, List[List[int]]]:
    """Group eligible physical cores by NUMA node.

    Excluding one sibling excludes its core; ``whole_nodes`` also excludes
    its node.
    """
    excluded = set(exclude)
    grouped: Dict[int, List[List[int]]] = defaultdict(list)
    skip = {
        node
        for (node, _s, _c), cpus in cores.items()
        if whole_nodes and excluded.intersection(cpus)
    }
    for (node, _socket, _core), cpus in sorted(cores.items()):
        if node in skip or excluded.intersection(cpus):
            continue
        grouped[node].append(cpus)
    return dict(grouped)


def select(
    nodes: Dict[int, List[List[int]]], want: int, whole_nodes: bool
) -> List[int]:
    """Select up to ``want`` logical CPUs using whole cores and few nodes.

    Never exceed ``want``. A partial core or, with ``whole_nodes``, a partial
    node may leave the selection short.
    """
    capacity = {node: sum(len(c) for c in cs) for node, cs in nodes.items()}

    if whole_nodes:
        # Ascending node id, so the choice is stable and the low nodes -- where
        # the excluded host CPUs live -- are already out of the running.
        chosen: List[int] = []
        for node in sorted(nodes):
            if len(chosen) + capacity[node] > want:
                continue
            chosen.extend(cpu for cpus in nodes[node] for cpu in cpus)
        return sorted(chosen)

    # A request that fits inside one node takes exactly one node -- the smallest
    # that can hold it, which leaves the wider nodes whole for whatever else the
    # shared machine is running.  Otherwise fill the widest nodes first, so the
    # allocation is spread over as few nodes as it can be.
    fits = sorted((capacity[node], node) for node in nodes if capacity[node] >= want)
    order = (
        [fits[0][1]] if fits else sorted(nodes, key=lambda n: (-capacity[n], n))
    )

    chosen = []
    for node in order:
        for cpus in nodes[node]:
            if len(chosen) + len(cpus) <= want:
                chosen.extend(cpus)
        if len(chosen) >= want:
            break
    return sorted(chosen)


def report(
    cores: Cores, nodes: Dict[int, List[List[int]]], chosen: Sequence[int], want: int
) -> None:
    """Print the CPU selection, host identity, and whole-node sizes."""
    picked = set(chosen)
    total = sum(len(c) for c in cores.values())
    all_nodes = {node for node, _s, _c in cores}
    siblings = sorted({len(c) for c in cores.values()})
    used = sorted({node for node, cs in nodes.items() if any(set(c) <= picked for c in cs)})
    physical = sum(1 for c in cores.values() if set(c) <= picked)

    print(
        f"host              {total} logical CPUs, {len(cores)} physical cores, "
        f"{len(all_nodes)} NUMA node(s), "
        f"{'/'.join(str(s) for s in siblings)} thread(s) per core"
    )
    print(f"requested         {want} logical CPUs")
    print(
        f"selected          {len(chosen)} logical CPUs on {physical} physical cores, "
        f"NUMA node(s) {','.join(str(n) for n in used) or '-'}"
    )
    print(f"left to the host  {total - len(chosen)} logical CPUs")

    per_node = sorted({sum(len(c) for c in cs) for cs in nodes.values()})
    if len(per_node) == 1 and len(nodes) > 1:
        step = per_node[0]
        sizes = ", ".join(f"{k} nodes = {k * step}" for k in range(1, len(nodes) + 1))
        print(f"whole-node sizes  {sizes}")

    print()
    print("# for .env")
    print(f"BENCH_CPUSET={compress(chosen)}")
    print()
    print("# multi-threaded regime: threads must match the cpuset size")
    print(f"docker compose run --rm benchmark threads={len(chosen)}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pick a whole-core, NUMA-aligned cpuset for the benchmark container, "
            "from lscpu -e or lscpu -p output."
        ),
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=256,
        help="Logical CPUs to allocate to the container (default: 256).",
    )
    parser.add_argument(
        "--whole-nodes",
        action="store_true",
        help=(
            "Never take part of a NUMA node, so the allocation does not share "
            "memory bandwidth with the rest of the machine.  Rounds down to the "
            "nearest whole node."
        ),
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="0",
        help=(
            "CPUs to leave to the host, comma-separated.  Their whole physical "
            "core is dropped, and under --whole-nodes their whole node "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--input",
        type=argparse.FileType("r"),
        default=sys.stdin,
        metavar="FILE",
        help="File holding the lscpu output (default: stdin).",
    )
    args = parser.parse_args(argv)

    if args.cpus < 1:
        parser.error("--cpus must be at least 1")
    exclude = [int(f) for f in args.exclude.split(",") if f.strip()]

    cores = parse_topology(args.input)
    nodes = by_node(cores, exclude, args.whole_nodes)
    if not nodes:
        raise SystemExit("every core holds an excluded CPU; nothing left to select")
    chosen = select(nodes, args.cpus, args.whole_nodes)
    if not chosen:
        raise SystemExit(
            f"nothing fits in {args.cpus} CPUs; the smallest unit available is "
            f"{min(sum(len(c) for c in cs) for cs in nodes.values()) if args.whole_nodes else min(len(c) for cs in nodes.values() for c in cs)} CPUs"
        )
    report(cores, nodes, chosen, args.cpus)

    if len(chosen) != args.cpus:
        unit = "node" if args.whole_nodes else "physical core"
        print(
            f"\nnote: {len(chosen)} CPUs selected rather than {args.cpus}, since only "
            f"whole {unit}s are taken.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
