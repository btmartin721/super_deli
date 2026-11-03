"""Invented lifecycle, control audit, old-report parity and retained guard failures."""

# ruff: noqa: E402

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from super_deli.benchmarking.separate_config import SeparateStudy, tasks
from super_deli.io.config import read_json


def test_invented_lifecycle(tmp_path):
    from super_deli.benchmarking.separate_archive import bind, create_fixture
    from super_deli.benchmarking.separate_report import report
    from super_deli.benchmarking.separate_study import (
        audit_phase,
        check_frozen,
        freeze,
        gate,
        run_phase,
    )

    source = create_fixture(tmp_path / "source")
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering only.\n")
    bind(source, root, fixture=True)
    freeze(root, SeparateStudy(fixture=True))
    assert run_phase(root, source, "preflight", workers=1)["attempted"] == 26
    pre = audit_phase(root, source, "preflight")
    assert pre["verified_optimizer_updates"] == 15000 and pre["branch_parity_checks"] == 6
    assert gate(root, staging_bytes=0)["passed"]
    assert run_phase(root, source, "main", workers=1)["attempted"] == 27
    main = audit_phase(root, source, "main")
    assert main["verified_optimizer_updates"] == 15500 and main["branch_parity_checks"] == 6
    assert report(root)["failed_points"] == 0
    (root / "approved-plan.md").write_text("tampered")
    with pytest.raises(ValueError, match="Frozen"):
        check_frozen(root)


def test_numeric_guard_stops_remaining_arms(tmp_path, monkeypatch):
    from super_deli.benchmarking import separate_run

    calls = []

    def fail(task, data, canonical, out, arm):
        calls.append(arm)
        return {
            "arm": arm,
            "planned": 2,
            "attempted": 1,
            "neural_fits_attempted": 1,
            "status": "failed",
            "numerical_guard_failed": True,
        }

    monkeypatch.setattr(separate_run, "run_arm", fail)
    t = tasks(SeparateStudy(fixture=True), "main")[0]
    result = separate_run.run_dataset(
        {"phase_root": str(tmp_path), "task": t, "stop_file": str(tmp_path / "STOP.json")}
    )
    assert len(calls) == 1 and result["numerical_guard_failed"]
    rows = read_json(tmp_path / t["key"] / "fits.json")
    assert all(r["status"] == "unattempted" for r in rows[1:])


def test_summaries_keep_unknown_pairs_and_stress_separate():
    from super_deli.benchmarking.neural_summary import summarize_rates

    rows = []
    for case in ("gaussian_independent", "discordant"):
        for arm in ("ref", "alt"):
            for i in range(2):
                rows.append(
                    {
                        "key": str(i),
                        "case": case,
                        "n": 64,
                        "arm": arm,
                        "status": "failed" if arm == "alt" and i == 1 else "available",
                        "selected_k": 2,
                        "endpoint": True,
                        "endpoint_name": None if case == "discordant" else "nontrivial_return",
                        "coverage": 1.0,
                        "ari": None,
                        "assigned_ari": None,
                        "linear_morphology": {"r2": [0.0, 0.0]},
                        "linear_ecology": {"r2": [0.0, 0.0]},
                    }
                )
    groups, primary, paired = summarize_rates(
        rows,
        arms=("ref", "alt"),
        joint=("ref", "alt"),
        comparisons=[{"reference": "ref", "alternative": "alt", "labels": {}}],
        primary_cases=("gaussian_independent",),
    )
    assert len(primary) == 2 and len(paired) == 1 and paired[0]["unknown_pairs"] == 1
    assert all(
        g["endpoint"] is None and g["failure_bounds"] is None
        for g in groups
        if g["case"] == "discordant"
    )
