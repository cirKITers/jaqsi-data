"""Process-wide thread pinning for the benchmark run.

Every simulator here reaches a different threading layer: OpenMP for Qulacs and
qiskit-aer, a BLAS pool for the numpy-based paths, numba for qibojit, and XLA's
own pool for JAX.  Left alone they disagree.  qibojit is the clearest case: it
sizes numba from ``len(psutil.Process().cpu_affinity())`` when its backend is
constructed, ignoring every environment variable, so on a 16-core machine it
takes all 16 threads while nothing else is pinned and the timings stop being
comparable.

:func:`pin_threads` therefore narrows the process CPU affinity first, which is
the one setting all of those layers agree to read, and sets the environment
variables on top of it.  It has to run *before* the numerical libraries are
imported, since each of them sizes its pool once at import time.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Environment variables read at import time by the threading layers in use.
#
# ``NUMBA_NUM_THREADS`` is deliberately absent.  It is a hard *cap*, and
# qibojit's backend constructor requests one thread per core it can see; if the
# cap is lower, numba raises and the backend cannot be built at all.  Narrowing
# the affinity instead changes what that constructor sees, so it asks for the
# right number in the first place.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "QULACS_NUM_THREADS",
)


def _pin_affinity(threads: int) -> bool:
    """Narrow the process to *threads* CPUs, returning whether it worked.

    Cores are taken from the set already permitted, so a narrower allocation
    from a batch scheduler is respected rather than overridden.
    """
    if not hasattr(os, "sched_setaffinity"):
        return False
    try:
        allowed = sorted(os.sched_getaffinity(0))
        if threads > len(allowed):
            logger.warning(
                f"Requested {threads} threads but only {len(allowed)} CPU(s) are "
                f"available to this process; leaving the affinity alone."
            )
            return False
        os.sched_setaffinity(0, set(allowed[:threads]))
        return True
    except OSError:
        return False


def pin_threads(threads: int) -> None:
    """Pin every threading layer in the process to *threads*.

    Call this before importing jax, numpy, qulacs, qiskit-aer or qibo.  Values
    already present in the environment are overwritten, so the configuration
    wins over whatever the shell happened to export.
    """
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")

    pinned = _pin_affinity(threads)

    for var in _THREAD_VARS:
        os.environ[var] = str(threads)

    # XLA sizes its own CPU pool and ignores OMP_NUM_THREADS; it takes its cue
    # from the affinity mask instead.  At one thread the Eigen pool is disabled
    # outright rather than merely sized to one.  Every entry in XLA_FLAGS has to
    # start with "--" or XLA aborts at startup.
    if threads == 1:
        existing = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = " ".join(
            [existing, "--xla_cpu_multi_thread_eigen=false"]
        ).strip()

    how = "CPU affinity and environment" if pinned else "environment only"
    logger.info(f"Pinned all threading layers to {threads} thread(s) via {how}.")


def num_threads() -> Optional[int]:
    """Return the pinned thread count, or ``None`` when nothing was pinned.

    Reads the environment rather than a module global so that the answer is the
    setting the libraries actually saw, including when a caller exported it
    before starting the process.
    """
    value = os.environ.get("OMP_NUM_THREADS")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None
