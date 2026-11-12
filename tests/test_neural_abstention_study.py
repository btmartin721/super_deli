"""Independent cohort laws, seed registry and complete sealed engineering lifecycle."""

# ruff: noqa: E402
import json
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")

from super_deli.benchmarking.abstention_audit import reference
from super_deli.benchmarking.abstention_config import CASES, AbstentionStudy, cohort_names
from super_deli.benchmarking.abstention_data import generate, seed_registry


@pytest.mark.parametrize("case", CASES)
def test_independent_cohort_laws(case):
    task = dict(case=case, n=32, fixture=True, seed_path=[834919, CASES.index(case)])
    for cohort in cohort_names(task):
        a, b = generate(task, cohort), reference(task, cohort)
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-12, atol=1e-12)
        if cohort in ("near", "far", "overlap"):
            assert np.unique(a["truth"]).tolist() == [4]


def test_registry_materializes_before_generation():
    value = seed_registry(AbstentionStudy())
    assert value["materialized_before_generation"]
    assert value["cohorts"] == 550 and value["seed_paths"] == 2750


def test_complete_observation_binding_rejects_added_and_changed_files(tmp_path):
    from super_deli.benchmarking.abstention_source import bind_observation_inventory
    from super_deli.benchmarking.view_weight_archive import check_binding

    folder = tmp_path / "invented_observation"
    folder.mkdir()
    (folder / "selected.json").write_text('{"invented":true}')
    (folder / "additional_evidence.txt").write_text("Retain the entire observation inventory.")
    files = {}
    bind_observation_inventory(tmp_path, "invented_observation", files)
    binding = dict(fixture=True, replays=[dict(path="invented_observation")], source_sha256=files)
    check_binding(tmp_path, binding)
    extra = folder / "unexpected.txt"
    extra.write_text("Added after binding")
    with pytest.raises(ValueError, match="inventory"):
        check_binding(tmp_path, binding)
    extra.unlink()
    (folder / "selected.json").write_text("Changed after binding")
    with pytest.raises(ValueError, match="changed"):
        check_binding(tmp_path, binding)


def test_complete_sealed_invented_fixture(tmp_path):
    pytest.importorskip("torch")
    code = (
        "from super_deli.benchmarking.abstention_fixture import smoke;"
        "import sys,json;print(json.dumps(smoke(sys.argv[1],workers=2)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "fixture")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-5000:]
    x = json.loads(result.stdout.splitlines()[-1])
    assert x["observations"] == 1 and x["workflows"] == 4
    assert x["verified_optimizer_updates"] == 1500 and x["failed_workflows"] == 0


def test_support_failure_keeps_training_and_optimizer_evidence(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from super_deli.benchmarking import abstention_run
    from super_deli.benchmarking.abstention_audit import audit_unit
    from super_deli.io.config import write_json

    def unavailable(*args):
        raise ValueError("invented unresolved support")

    task = dict(
        key="engineering_failure", case="linear_standard", n=32, fixture=True, seed_path=[84178]
    )
    phase = tmp_path / "phase"
    phase.mkdir()
    folder = phase / task["key"]
    folder.mkdir()
    write_json(task, folder / "task.json")
    job = dict(
        kind="fresh", task=task, phase_root=str(phase), stop_file=str(tmp_path / "STOP.json")
    )
    monkeypatch.setattr(abstention_run, "fit_support", unavailable)
    rows = abstention_run.run_fresh(job, folder)
    write_json(rows, folder / "fits.json")
    assert len(rows) == 4 and all(r["status"] == "failed" for r in rows)
    assert not (tmp_path / "STOP.json").exists()
    checked = audit_unit(folder, job, tmp_path, cache={})[0]
    assert checked["neural_fits"] == 3 and checked["verified_updates"] == 1500
    assert checked["policy_evaluations"] == 0
