"""Tests for process-wide thread pinning.

Pinning is what makes the timings comparable across simulators, so these tests
cover the mechanism rather than the resulting performance.  Every test restores
the process CPU affinity, because pinning is global and would otherwise leak
into the rest of the session.
"""

from __future__ import annotations

import os

import pytest

from benchmark.threads import _THREAD_VARS, num_threads, pin_threads


@pytest.fixture(autouse=True)
def restore_affinity():
    """Put the process CPU affinity back after each test."""
    if not hasattr(os, "sched_getaffinity"):
        yield
        return
    before = os.sched_getaffinity(0)
    yield
    os.sched_setaffinity(0, before)


@pytest.fixture(autouse=True)
def restore_env(monkeypatch):
    """Isolate the thread environment variables from the rest of the session."""
    for var in (*_THREAD_VARS, "XLA_FLAGS"):
        monkeypatch.delenv(var, raising=False)
    yield


class TestPinThreads:
    @pytest.mark.parametrize("threads", [1, 2])
    def test_sets_every_thread_variable(self, threads):
        pin_threads(threads)
        for var in _THREAD_VARS:
            assert os.environ[var] == str(threads)

    def test_does_not_cap_numba(self):
        """NUMBA_NUM_THREADS is a cap that makes qibojit's constructor raise."""
        pin_threads(1)
        assert "NUMBA_NUM_THREADS" not in os.environ

    @pytest.mark.skipif(
        not hasattr(os, "sched_setaffinity"), reason="affinity is Linux-only"
    )
    def test_narrows_cpu_affinity(self):
        pin_threads(1)
        assert len(os.sched_getaffinity(0)) == 1

    @pytest.mark.skipif(
        not hasattr(os, "sched_setaffinity"), reason="affinity is Linux-only"
    )
    def test_takes_cores_from_the_permitted_set(self):
        """A narrower allocation from a batch scheduler must be respected."""
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < 2:
            pytest.skip("needs at least two CPUs")
        os.sched_setaffinity(0, {allowed[-1]})
        pin_threads(1)
        assert os.sched_getaffinity(0) == {allowed[-1]}

    @pytest.mark.skipif(
        not hasattr(os, "sched_setaffinity"), reason="affinity is Linux-only"
    )
    def test_over_request_warns_and_leaves_affinity_alone(self, caplog):
        before = os.sched_getaffinity(0)
        with caplog.at_level("WARNING"):
            pin_threads(len(before) + 1000)
        assert os.sched_getaffinity(0) == before
        assert "available to this process" in caplog.text

    def test_single_thread_disables_the_xla_pool(self):
        pin_threads(1)
        assert "--xla_cpu_multi_thread_eigen=false" in os.environ["XLA_FLAGS"]

    def test_every_xla_flag_starts_with_dashes(self):
        """XLA aborts at startup on any entry that does not begin with '--'."""
        pin_threads(1)
        assert all(f.startswith("--") for f in os.environ["XLA_FLAGS"].split())

    def test_multi_thread_leaves_the_xla_pool_enabled(self):
        pin_threads(2)
        assert "--xla_cpu_multi_thread_eigen=false" not in os.environ.get(
            "XLA_FLAGS", ""
        )

    def test_preserves_existing_xla_flags(self, monkeypatch):
        monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/x")
        pin_threads(1)
        assert "--xla_dump_to=/tmp/x" in os.environ["XLA_FLAGS"]

    @pytest.mark.parametrize("threads", [0, -1])
    def test_rejects_non_positive(self, threads):
        with pytest.raises(ValueError, match="at least 1"):
            pin_threads(threads)


class TestNumThreads:
    def test_none_when_unpinned(self):
        assert num_threads() is None

    def test_reports_the_pinned_count(self):
        pin_threads(3)
        assert num_threads() == 3

    def test_none_on_a_non_numeric_value(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "all")
        assert num_threads() is None
