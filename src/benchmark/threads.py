"""Pin CPU affinity and thread pools for comparable benchmarks.

Simulators use different threading layers, and qibojit ignores thread
environment variables when sizing its pool. Pin affinity and set the
environment before importing numerical libraries, when they typically
size their pools.
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
    """Restrict affinity to ``threads`` already permitted CPUs.

    Return whether pinning succeeded; preserve a batch scheduler's allocation.
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
    """Pin process affinity and thread pools to ``threads`` CPUs.

    Call before importing numerical libraries. Configured values replace
    existing thread environment variables.
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

    # XLA's intra-op threading barely speeds up one statevector, while the
    # samples of a batch are independent.  One CPU device per pinned thread
    # lets jaqsi split the batch over the same cores every simulator gets.
    # Read when JAX initialises; other JAX simulators run on the first device.
    os.environ["JAX_NUM_CPU_DEVICES"] = str(threads)

    how = "CPU affinity and environment" if pinned else "environment only"
    logger.info(f"Pinned all threading layers to {threads} thread(s) via {how}.")


def num_threads() -> Optional[int]:
    """Return the configured thread count, or ``None`` if unset.

    Read the environment to reflect the value seen by libraries.
    """
    value = os.environ.get("OMP_NUM_THREADS")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None
