"""Integrity, ownership, interruption and independent budget accounting checks."""

# ruff: noqa: E402

import json
import os
import socket
import time

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.benchmarking.neural_false_split_config import NeuralFalseSplitStudy, counts, tasks
from super_deli.benchmarking.neural_false_split_state import (
    StudyLock,
    atomic_json,
    seal,
    verify_seal,
)


def test_live_lock_and_foreign_lock_are_never_removed(tmp_path):
    atomic_json(tmp_path / "frozen.json", {"fixture": True})
    with StudyLock(tmp_path):
        original = (tmp_path / "study.lock").read_bytes()
        with pytest.raises(ValueError, match="active"):
            with StudyLock(tmp_path):
                pytest.fail("Live owner cannot be replaced")
        assert (tmp_path / "study.lock").read_bytes() == original
    atomic_json(tmp_path / "study.lock", {"kind": "unrelated_file"})
    original = (tmp_path / "study.lock").read_bytes()
    with pytest.raises(ValueError, match="Recognized"):
        with StudyLock(tmp_path):
            pytest.fail("Foreign lock cannot be replaced")
    assert (tmp_path / "study.lock").read_bytes() == original


def test_completed_unit_tampering_is_rejected(tmp_path):
    atomic_json(tmp_path / "frozen.json", {"fixture": True})
    task = tasks(NeuralFalseSplitStudy(fixture=True), "main")[0]
    job = dict(kind="fresh", task=task)
    out = tmp_path / "main" / task["key"]
    out.mkdir(parents=True)
    (out / "data.txt").write_text("invented original")
    seal(tmp_path, "main", job, dict(seconds=1, attempted=6, neural_fits=6, failed_points=0))
    verify_seal(tmp_path, "main", job)
    (out / "data.txt").write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        verify_seal(tmp_path, "main", job)


def test_interruption_with_corrupt_partial_json_is_unavailable(tmp_path, monkeypatch):
    from super_deli.benchmarking import neural_false_split_study as study
    from super_deli.benchmarking.neural_false_split_report import assessment

    atomic_json(tmp_path / "frozen.json", {"fixture": True})
    task = tasks(NeuralFalseSplitStudy(fixture=True), "main")[0]
    job = dict(kind="fresh", task=task)
    out = tmp_path / "main" / task["key"] / "neural/calibration"
    out.mkdir(parents=True)
    (out / "summary.json").write_text('{"partial":')
    atomic_json(
        tmp_path / "state/active" / ("main_" + task["key"] + ".json"),
        dict(
            key=task["key"],
            started=time.monotonic() - 1,
            elapsed=None,
            pid=os.getpid(),
            host=socket.gethostname(),
            status="running",
        ),
    )
    monkeypatch.setattr(study, "process_alive", lambda _: False)
    value = study.recover_interrupted(tmp_path, "main", job)
    assert value["interrupted"] and value["result"]["attempted"] is None
    row = assessment(tmp_path, task, "neural")
    assert row["p_value"] is row["ordinary_split"] is row["global_rejection"] is None
    assert (out / "summary.json").read_text() == '{"partial":'


def test_failed_outcomes_remain_in_primary_and_paired_bounds():
    from super_deli.benchmarking.neural_calibration_report import rate
    from super_deli.benchmarking.neural_false_split_report import paired_interval

    r = rate([True, False, None, None], family_size=8)
    assert r["requested"] == 4 and r["unavailable"] == 2
    assert r["identification_bounds"] == [0.25, 0.75]
    lo, hi = paired_interval(0, 0, 4, 4, 0.05 / 4)
    assert lo == -1 and hi == 1


def test_resource_projection_requires_high_draw_stratum(tmp_path, monkeypatch):
    from super_deli.benchmarking import neural_false_split_budget as b

    c = NeuralFalseSplitStudy(fixture=True)
    task = tasks(c, "preflight")[0]
    atomic_json(
        tmp_path / "replay-binding.json",
        dict(
            replays=[
                dict(
                    key="replay",
                    task=task,
                    path="invented",
                )
            ]
        ),
    )
    for name in ("frozen.json", "preflight-execution.json", "preflight-audit.json"):
        atomic_json(tmp_path / name, {})
    rows = []
    for t in tasks(c, "preflight")[:-1]:
        out = tmp_path / "preflight" / t["key"]
        out.mkdir(parents=True)
        (out / "invented.txt").write_text("fixture")
        rows.append(dict(key=t["key"], seconds=1))
    e = dict(
        completed=True,
        unknown_counter_units=0,
        records=rows,
        attempted=counts(c, "preflight")["workflows"],
        neural_fits=counts(c, "preflight")["neural_fits"],
    )
    a = dict(
        records=[], seconds=1, verified_optimizer_updates=counts(c, "preflight")["verified_updates"]
    )
    monkeypatch.setattr(
        b.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 16 * 1024**3})()
    )
    r = b.project(
        tmp_path, {"counts": {"preflight": counts(c, "preflight")}}, c, e, a, staging_bytes=0
    )
    assert not r["passed"] and any("B3" in name for name in r["missing_strata"])


def test_seed_policy_audit_does_not_trust_stored_execution(tmp_path):
    from super_deli.benchmarking.neural_false_split_audit import check_policy
    from super_deli.benchmarking.neural_false_split_run import calibration_policy

    t = tasks(NeuralFalseSplitStudy(fixture=True), "main")[0]
    for arm in ("neural", "pca4"):
        value = json.loads(json.dumps(calibration_policy(t, arm).to_dict()))
        value["random_seed"] += 1
        atomic_json(tmp_path / arm / "execution.json", dict(configuration=value, point_only=False))
    with pytest.raises(ValueError, match="prespecified calibration policy"):
        check_policy(tmp_path, t)
