"""Resume only results whose runtime and workload settings are known."""

import json

import pytest

from benchmark.config import load_config
from benchmark import provenance


@pytest.fixture
def environment(monkeypatch):
    value = {"python": "test", "packages": {"jax": {"version": "1"}}}
    monkeypatch.setattr(provenance, "_environment", lambda: value)
    return value


def test_records_config_and_allows_sweep_extension(tmp_path, environment):
    path = tmp_path / "bench.csv"
    cfg = load_config(overrides=["output.identifier=test"])
    provenance.record_provenance(path, cfg, has_results=False)
    cfg.qubits.max += 1
    cfg.execution.batch_size += 1
    cfg.threads += 1
    provenance.record_provenance(path, cfg, has_results=True)
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["environment"] == environment
    assert len(metadata["runs"]) == 2
    assert (
        metadata["runs"][1]["config"]["execution"]["batch_size"]
        == cfg.execution.batch_size
    )
    assert "cpu_affinity" in metadata["runs"][1]


@pytest.mark.parametrize(
    "override",
    [
        "seed=12",
        "optimal_config=false",
        "closed_form=false",
        "envelope=drag",
        "warmup=false",
        "precision=1e-5",
        "depolarizing=0.02",
        "execution.n_iters=21",
    ],
)
def test_rejects_changed_workload(tmp_path, environment, override):
    path = tmp_path / "bench.csv"
    cfg = load_config()
    provenance.record_provenance(path, cfg, has_results=False)
    before = path.with_suffix(".json").read_bytes()
    with pytest.raises(ValueError, match="settings changed"):
        provenance.record_provenance(
            path, load_config(overrides=[override]), has_results=True
        )
    assert path.with_suffix(".json").read_bytes() == before


def test_rejects_changed_installed_environment(tmp_path, environment):
    path = tmp_path / "bench.csv"
    cfg = load_config()
    provenance.record_provenance(path, cfg, has_results=False)
    environment["packages"]["jax"]["version"] = "2"
    with pytest.raises(ValueError, match="environment changed"):
        provenance.record_provenance(path, cfg, has_results=True)


def test_legacy_results_cannot_be_resumed(tmp_path, environment):
    with pytest.raises(ValueError, match="provenance is missing"):
        provenance.record_provenance(
            tmp_path / "old.csv", load_config(), has_results=True
        )


def test_changed_thread_environment_requires_a_new_run(
    tmp_path, environment, monkeypatch
):
    path = tmp_path / "bench.csv"
    cfg = load_config()
    provenance.record_provenance(path, cfg, has_results=False)
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false --test-change")
    with pytest.raises(ValueError, match="thread environment changed"):
        provenance.record_provenance(path, cfg, has_results=True)


def test_changed_jax_device_count_requires_a_new_run(
    tmp_path, environment, monkeypatch
):
    path = tmp_path / "bench.csv"
    cfg = load_config()
    provenance.record_provenance(path, cfg, has_results=False)
    monkeypatch.setenv("JAX_NUM_CPU_DEVICES", "7")
    with pytest.raises(ValueError, match="thread environment changed"):
        provenance.record_provenance(path, cfg, has_results=True)


def test_source_identity_tracks_local_edits(tmp_path, monkeypatch):
    from importlib.machinery import ModuleSpec

    source = tmp_path / "__init__.py"
    source.write_text("value = 1\n")
    monkeypatch.setattr(
        provenance.importlib.util,
        "find_spec",
        lambda name: ModuleSpec(name, None, origin=str(source)),
    )
    before = provenance._source_identity("example")
    source.write_text("value = 2\n")
    after = provenance._source_identity("example")
    assert before["path"] == after["path"] == str(tmp_path)
    assert before["sha256"] != after["sha256"]
