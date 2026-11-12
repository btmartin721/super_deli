"""Immutable seed budgets, exact tradeoff boundaries and complete invented lifecycle."""

# ruff: noqa: E402
import json
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")

from super_deli.benchmarking.local_reference_config import LocalReferenceStudy
from super_deli.benchmarking.local_reference_study import seed_registry


def test_new_registry_and_complete_budget():
    from super_deli.benchmarking.abstention_config import AbstentionStudy
    from super_deli.benchmarking.abstention_data import seed_registry as old_registry

    new = seed_registry(LocalReferenceStudy())
    old = old_registry(AbstentionStudy())
    assert (
        new["materialized_before_generation"]
        and new["cohorts"] == 550
        and new["seed_paths"] == 2750
    )

    def paths(v):
        return {tuple(p) for r in v["records"] for p in r["seeds"].values()}

    assert not paths(new) & paths(old)


@pytest.mark.parametrize("case", ["linear_weak", "linear_standard", "combined", "confounded_size"])
def test_true_law_reference_with_independent_multivariate_density(tmp_path, case):
    from super_deli.benchmarking.abstention_config import cohort_names
    from super_deli.benchmarking.abstention_data import generate
    from super_deli.benchmarking.local_reference_audit import audit_oracle
    from super_deli.benchmarking.local_reference_oracle import evaluate
    from super_deli.io.config import write_json

    task = dict(case=case, n=32, fixture=True, seed_path=[983471, case.__len__()])
    cohorts = {name: generate(task, name) for name in cohort_names(task)}
    arrays, report = evaluate(cohorts, task)
    np.savez_compressed(tmp_path / "oracle.npz", **arrays)
    write_json(report, tmp_path / "oracle-result.json")
    assert audit_oracle(tmp_path, cohorts, task) == 19 * (len(cohorts) - 2)
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["iid_score"][0] += 0.001
    np.savez_compressed(tmp_path / "oracle.npz", **changed)
    with pytest.raises(AssertionError):
        audit_oracle(tmp_path, cohorts, task)


def test_benefit_uses_exact_fraction_boundaries():
    from super_deli.benchmarking.local_reference_report import benefit, rate

    def known(n, accepted, good):
        return dict(n=n, accepted=accepted, correct_accepted_fraction=good / n)

    row = dict(
        status="available",
        local_status="available",
        metrics=dict(
            distance=dict(
                iid=known(1000, 800, 800),
                near=dict(n=1000, accepted=100),
                far=dict(n=1000, accepted=100),
            ),
            local=dict(
                iid=known(1000, 900, 891),
                near=dict(n=1000, accepted=120),
                far=dict(n=1000, accepted=120),
            ),
        ),
    )
    assert benefit(row) is True
    row["metrics"]["local"]["near"]["accepted"] = 121
    assert benefit(row) is False
    row["local_status"] = "failed"
    assert benefit(row) is None
    result = rate([True, False, None])
    assert result["failure_bounds"] == [1 / 3, 2 / 3] and result["unavailable"] == 1


def test_complete_sealed_invented_lifecycle(tmp_path):
    pytest.importorskip("torch")
    code = (
        "import sys,json;"
        "from super_deli.benchmarking.local_reference_fixture import study_fixture;"
        "print(json.dumps(study_fixture(sys.argv[1],workers=2)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "fixture")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-6000:]
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["observations"] == summary["archived_observations"] == 1
    assert summary["main_model_workflows"] == 4 and summary["verified_optimizer_updates"] == 3000
    assert summary["failed_workflows"] == 0 and summary["primary_rates"] == 8
    from super_deli.benchmarking.local_reference_audit import audit_unit
    from super_deli.benchmarking.local_reference_run import replay
    from super_deli.io.config import write_json

    source = tmp_path / "fixture/invented-source"
    for arm in ("logistic", "residual_0"):
        folder = tmp_path / ("replay_" + arm)
        folder.mkdir()
        entry = dict(path="main/invented_predecessor", arm=arm, n=32)
        task = dict(key="replay_" + arm, case="legacy", n=32)
        write_json(task, folder / "task.json")
        result = replay(entry, source, folder)
        assert result[0]["neural_fits"] == 0 and result[0]["replay_cohorts"] == 5
        verified = audit_unit(folder, dict(kind="replay", task=task, entry=entry), source, cache={})
        assert verified[0]["replay_cohorts"] == 5


def test_complete_observation_inventory_including_nonreplay_paths(tmp_path):
    from super_deli.benchmarking.local_reference_source import check_source
    from super_deli.core.provenance import fingerprint_file

    folder = tmp_path / "invented_observation"
    folder.mkdir()
    (folder / "kept.txt").write_text("invented")
    binding = dict(
        fixture=True,
        source_sha256={
            "invented_observation/kept.txt": fingerprint_file(folder / "kept.txt").sha256
        },
        replays=[],
        observations=[dict(path="invented_observation")],
    )
    check_source(tmp_path, binding)
    (folder / "extra.txt").write_text("added after binding")
    with pytest.raises(ValueError, match="inventory"):
        check_source(tmp_path, binding)


def test_original_seal_rejects_changed_source_even_if_new_hash_is_rebound(tmp_path):
    from super_deli.benchmarking.local_reference_source import check_source
    from super_deli.benchmarking.neural_false_split_state import seal
    from super_deli.core.provenance import fingerprint_file
    from super_deli.io.config import write_json

    task = dict(key="invented", case="linear_standard", n=32)
    folder = tmp_path / "main/invented"
    folder.mkdir(parents=True)
    write_json(dict(invented=True), tmp_path / "frozen.json")
    (folder / "kept.txt").write_text("original invented evidence")
    seal(
        tmp_path,
        "main",
        dict(kind="fresh", task=task),
        dict(seconds=0, attempted=4, neural_fits=0, failed_points=0),
    )
    binding = dict(
        fixture=False,
        source_sha256={"main/invented/kept.txt": fingerprint_file(folder / "kept.txt").sha256},
        replays=[],
        observations=[dict(path="main/invented", task=task)],
    )
    check_source(tmp_path, binding)
    (folder / "kept.txt").write_text("changed invented evidence")
    binding["source_sha256"]["main/invented/kept.txt"] = fingerprint_file(
        folder / "kept.txt"
    ).sha256
    with pytest.raises(ValueError, match="evidence changed"):
        check_source(tmp_path, binding)
