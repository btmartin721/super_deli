"""Pure-Python constraints, gate overlap and unavailable-result regression tests."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.benchmarking.bootstrap_diagnostic_gates import (
    gates,
    truth_sizes,
    unique_support,
    validation_gain,
)
from super_deli.models.bootstrap_diagnostic_config import BootstrapDiagnosticConfig

ROOT = Path(__file__).resolve().parents[1]


def config(name="study"):
    return BootstrapDiagnosticConfig.from_dict(
        json.loads((ROOT / f"examples/bootstrap-instability/{name}.json").read_text())
    )


def fixture():
    full = [
        dict(
            k=k,
            status="eligible",
            aic=100 - k,
            hard_cluster_sizes=[10] * k,
            distinct_unit_sizes=[10] * k,
        )
        for k in range(1, 5)
    ]
    folds = [
        dict(
            fold=i,
            status="complete",
            original_training_indices=[j for j in range(50) if j // 10 != i],
            original_validation_indices=list(range(10 * i, 10 * i + 10)),
            candidates=copy.deepcopy(full),
            log_densities={str(k): [float(k - 1)] * 10 for k in range(1, 5)},
        )
        for i in range(5)
    ]
    selection = dict(
        status="complete",
        screened_k=4,
        candidates=[dict(k=k, passes=True, pooled_validation_gain=k - 1) for k in range(1, 5)],
    )
    return full, folds, selection


def test_plan_exact_budgets_and_disjoint_probe():
    c = config()
    assert sum(r["planned_searches"] for r in c.plan()) == 13680
    assert sum(r["planned_oracles"] for r in c.plan()) == 1080
    assert sum(r["planned_searches"] for r in c.probe().plan()) == 1368
    assert sum(r["planned_oracles"] for r in c.probe().plan()) == 108
    main = {r[k] for r in c.plan() for k in ("data_seed", "split_seed", "draw_stream_seed")}
    probe = {
        r[k] for r in c.probe().plan() for k in ("data_seed", "split_seed", "draw_stream_seed")
    }
    assert main.isdisjoint(probe)
    assert c == BootstrapDiagnosticConfig.from_dict(c.to_dict())
    assert len(c.draw_seeds(c.plan()[0])) == 25


@pytest.mark.parametrize(
    "field,value",
    [
        ("extended_draws", 26),
        ("extended_starts", 10),
        ("extended_starts", 101),
        ("n_draws", 0),
        ("schema_version", 2),
    ],
)
def test_invalid_settings(field, value):
    with pytest.raises(ValueError):
        replace(config(), **{field: value})


def test_support_order_and_generating_feasibility_are_distinct():
    assert unique_support([4, 2, 4, 9, 2, 0]) == [4, 2, 9, 0]
    labels = [i // 10 for i in range(40)]
    indices = list(range(4)) + list(range(10, 20)) + list(range(20, 30)) + list(range(30, 40))
    assert len(set(indices)) >= 20
    assert truth_sizes(labels, indices) == [4, 10, 10, 10]
    assert truth_sizes(labels, indices * 2) == [4, 10, 10, 10]
    assert truth_sizes(None, indices) is None


def test_equal_validation_units_and_pooled_occurrences():
    folds = [
        dict(original_validation_indices=[0, 0, 1], log_densities={"1": [0, 0, 0], "4": [2, 2, -3]})
    ]
    assert validation_gain(4, folds) == pytest.approx(1 / 3)
    assert validation_gain(4, folds, unique=True) == pytest.approx(-0.5)
    assert validation_gain(3, folds) is None
    folds[0]["log_densities"]["4"] = [1]
    with pytest.raises(ValueError):
        validation_gain(4, folds)


def test_gate_overlap_and_no_mutation():
    full, folds, selection = fixture()
    baseline = copy.deepcopy((full, folds, selection))
    result = gates(full, folds, list(range(50)), selection, minimum=5)
    assert result["selected_k"] == 4
    assert result["candidates"][-1]["first_failing_gate"] == "selected"
    assert (full, folds, selection) == baseline
    full[-1]["status"] = "small_component"
    folds[0]["candidates"][-1]["status"] = "small_component"
    selection["screened_k"] = 3
    selection["candidates"][-1].update(passes=False, pooled_validation_gain=None)
    result = gates(full, folds, list(range(50)), selection, minimum=5)
    row = result["candidates"][-1]
    assert row["first_failing_gate"] == "full_eligibility"
    assert not row["full_eligible"] and not row["all_folds_eligible"]
    assert result["counterfactuals"]["remove_full_size"]["selected_k"] == 3
    assert result["counterfactuals"]["remove_fold_size"]["selected_k"] == 3


def test_unfitted_counterfactual_and_failures_never_become_k1():
    full, folds, selection = fixture()
    full[-1]["status"] = "insufficient_samples"
    selection["screened_k"] = 3
    selection["candidates"][-1].update(passes=False, pooled_validation_gain=None)
    result = gates(full, folds, list(range(50)), selection, minimum=5)
    assert result["counterfactuals"]["remove_full_size"]["status"] == "unavailable"
    assert result["counterfactuals"]["remove_full_size"]["missing_candidates"] == [4]
    failed = gates([], [], [], dict(status="withheld", screened_k=None, error="failed"), minimum=5)
    assert failed["status"] == "unavailable" and failed["selected_k"] is None


def test_inconsistent_selection_fails_closed():
    full, folds, selection = fixture()
    selection["screened_k"] = 2
    with pytest.raises(ValueError):
        gates(full, folds, list(range(50)), selection, minimum=5)


@pytest.mark.parametrize("budget,passed", [(3 * 1024**3, True), (1024**2, False)])
def test_measured_gate_retains_failed_decision(tmp_path, monkeypatch, budget, passed):
    from super_deli.benchmarking import bootstrap_diagnostic_preflight as module
    from super_deli.benchmarking.bootstrap_diagnostic_preflight import storage_preflight
    from super_deli.io.config import read_json, write_json

    c = replace(config(), max_artifact_bytes=budget)
    probe = tmp_path / "probe"
    probe.mkdir()
    (tmp_path / "audit").mkdir()
    write_json(c.probe().to_dict(), probe / "configuration.json")
    rows = []
    for i, cell in enumerate(c.cells):
        write_json({}, probe / f"manifest_{i}.json")
        rows.append(dict(cell=cell.name, size_bytes=10000, manifest=f"manifest_{i}.json"))
    (probe / "observations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(
        dict(
            status="passed",
            audited_searches=1368,
            audited_oracles=108,
            unattempted_fits=0,
            unavailable_diagnostics=0,
        ),
        tmp_path / "audit/summary.json",
    )
    write_json(
        dict(status="verified", audited_searches=1368, audited_oracles=108), tmp_path / "check.json"
    )
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda p: type("Usage", (), dict(free=100 * 1024**3))()
    )
    target = tmp_path / "gate.json"
    kwargs = dict(
        probe_study=probe,
        probe_audit=tmp_path / "audit",
        probe_check=tmp_path / "check.json",
        artifact_root=tmp_path,
        output_file=target,
    )
    if passed:
        result = storage_preflight(c, **kwargs)
        assert result["status"] == "passed"
        assert result["main_searches_including_replays"] == 13860
        assert result["main_oracles"] == 1080
        assert result["replay_archive_allowance"] > 0
    else:
        with pytest.raises(RuntimeError, match="Storage gate failed"):
            storage_preflight(c, **kwargs)
        assert read_json(target)["status"] == "failed"


def reporting():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bootstrap_report", ROOT / "scripts/report_bootstrap_instability.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reporting_keeps_unavailable_denominators_and_empty_signal():
    result = reporting().summarize([dict(status="unavailable", error="retained")], 1, reference_k=4)
    assert result["available"] == 0 and result["requested"] == 1
    assert result["k4"]["conditional_rate"] is None
    assert result["k4"]["unknown_outcome_bounds"] == [0, 1]
    assert result["signal_and_numerical_trace"]["k4_candidate_denominator"] == 0
    assert result["signal_and_numerical_trace"]["full_search_axis_r2"]["joint"] == dict(
        a=None, b=None
    )
    assert result["mean_assignment_changes_when_same_k"] is None


def test_reporting_exports_axis_trace_and_separates_training_from_oob():
    result = reporting().trace(
        [
            dict(
                selection=dict(screened_k=4),
                global_oob=[2, 3],
                searches=dict(
                    full=dict(
                        joint_axis_r2=dict(a=0.8, b=0.9),
                        candidates=[
                            dict(
                                k=4,
                                status="eligible",
                                all_original_ari=0.7,
                                global_oob_ari=0.5,
                                training_occurrence_ari=0.8,
                                entropy=0.2,
                                covariance_condition_numbers=[3],
                                iterations=4,
                                converged=True,
                            )
                        ],
                    )
                ),
            )
        ]
    )
    assert result["full_search_axis_r2"]["joint"] == dict(a=0.8, b=0.9)
    assert result["selected_all_original_ari"] == 0.7
    assert result["selected_global_oob_ari"] == 0.5
    assert result["selected_training_occurrence_ari"] == 0.8
    assert result["global_oob_counts"] == [2]
