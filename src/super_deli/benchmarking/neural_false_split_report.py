"""Fixed Gaussian rates and a separately labelled selected-case precision diagnostic."""

import math
from collections import Counter
from pathlib import Path

from super_deli.benchmarking.gaussian_graph_report import difference_interval
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_calibration_report import rate
from super_deli.benchmarking.neural_false_split_config import ARMS, CASES, archived_tasks, tasks
from super_deli.benchmarking.neural_false_split_state import StudyLock, atomic_json
from super_deli.benchmarking.neural_false_split_study import check_frozen
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def assessment(root, task, arm):
    seal = read_json(root / "ledger/main" / (task["key"] + ".json"))
    if seal["interrupted"]:
        return dict(
            key=task["key"],
            case=task["case"],
            arm=arm,
            n=task["n"],
            interrupted=True,
            point_available=False,
            selected_k=None,
            ordinary_split=None,
            global_rejection=None,
            p_value=None,
            failed_draws=None,
            conditional_tail_interval=None,
        )
    out = root / "main" / task["key"] / arm / "calibration"
    point = read_json(out / "observed.json") if (out / "observed.json").exists() else {}
    summary = read_json(out / "summary.json") if (out / "summary.json").exists() else {}
    inf = summary.get("inference", {}) if not seal["interrupted"] else {}
    available = not seal["interrupted"] and point.get("status") == "complete"
    return dict(
        key=task["key"],
        case=task["case"],
        arm=arm,
        n=task["n"],
        interrupted=seal["interrupted"],
        point_available=available,
        selected_k=point.get("selected_k") if available else None,
        ordinary_split=(point["selected_k"] > 1) if available else None,
        global_rejection=inf.get("reject_joint_raw_gaussian"),
        p_value=inf.get("p_value"),
        failed_draws=inf.get("failed"),
        conditional_tail_interval=inf.get("conditional_null_tail_interval_95"),
    )


def paired_interval(positive, negative, missing, n, alpha):
    return [
        difference_interval(positive, negative + missing, n, alpha)[0],
        difference_interval(positive + missing, negative, n, alpha)[1],
    ]


def precision_summary(root, source, config, binding):
    job = next(j for j in archived_tasks(config, binding) if j["kind"] == "sensitivity")
    original = source / job["source_path"]
    output = []
    for arm in ARMS:
        out = root / "main" / job["task"]["key"] / arm / "calibration"
        old = read_json(original / arm / "calibration/summary.json")
        interrupted = read_json(root / "ledger/main" / (job["task"]["key"] + ".json"))[
            "interrupted"
        ]
        new = (
            read_json(out / "summary.json")
            if not interrupted and (out / "summary.json").exists()
            else {}
        )
        record = dict(arm=arm, original={}, sensitivity={})
        for key, summary in (("original", old), ("sensitivity", new if not interrupted else {})):
            inf = summary.get("inference", {})
            record[key] = {
                k: inf.get(k)
                for k in (
                    "requested",
                    "successful",
                    "failed",
                    "p_value",
                    "exceedances",
                    "conditional_null_tail_interval_95",
                    "status",
                )
            }
        output.append(record)
    return dict(
        source_case=binding["replays"][0]["path"],
        arms=output,
        excluded_from_fresh_rates=True,
        historical_pvalues_unchanged=True,
        interpretation="One selected archived case; Monte Carlo precision sensitivity only",
    )


def report(root, source):
    from super_deli.benchmarking.neural_false_split_study import (
        all_jobs,
        group_execution,
        group_names,
    )
    from super_deli.benchmarking.view_weight_archive import check_binding

    root, source = Path(root), Path(source)
    frozen, config = check_frozen(root)
    with StudyLock(root):
        require(not (root / "STOP.json").exists(), "Retained guard prohibits reporting")
        binding = read_json(root / "replay-binding.json")
        check_binding(source, binding)
        for phase in ("preflight", "main"):
            audit = read_json(root / (phase + "-audit.json"))
            require(
                audit["passed"]
                and audit["execution_sha256"]
                == fingerprint_file(root / (phase + "-execution.json")).sha256,
                "Complete independent phase audit",
            )
            for group in group_names(config, phase):
                require(
                    read_json(root / "executions" / phase / (group + ".json"))
                    == group_execution(root, config, phase, group),
                    "Unchanged audited unit inventories",
                )
        rows = [assessment(root, t, arm) for t in tasks(config, "main") for arm in ARMS]
        primary, groups, paired = [], [], []
        for case in CASES:
            for arm in ARMS:
                cell = [r for r in rows if (r["case"], r["arm"]) == (case, arm)]
                group = dict(
                    case=case,
                    arm=arm,
                    n=cell[0]["n"],
                    requested=len(cell),
                    selected_k_counts=dict(
                        Counter(str(r["selected_k"]) for r in cell if r["point_available"])
                    ),
                )
                for endpoint in ("ordinary_split", "global_rejection"):
                    value = rate([r[endpoint] for r in cell], family_size=8)
                    p = value["all_request_known_rate"]
                    value["monte_carlo_standard_error"] = (
                        math.sqrt(p * (1 - p) / len(cell)) if not value["unavailable"] else None
                    )
                    group[endpoint] = value
                    primary.append(
                        dict(case=case, arm=arm, n=cell[0]["n"], endpoint=endpoint, **value)
                    )
                groups.append(group)
            for endpoint in ("ordinary_split", "global_rejection"):
                left = {
                    r["key"]: r[endpoint]
                    for r in rows
                    if r["case"] == case and r["arm"] == "neural"
                }
                right = {
                    r["key"]: r[endpoint] for r in rows if r["case"] == case and r["arm"] == "pca4"
                }
                common = [k for k in left if left[k] is not None and right[k] is not None]
                positive = sum(left[k] and not right[k] for k in common)
                negative = sum(right[k] and not left[k] for k in common)
                n, missing = len(left), len(left) - len(common)
                paired.append(
                    dict(
                        case=case,
                        endpoint=endpoint,
                        requested=n,
                        common_available=len(common),
                        unknown_pairs=missing,
                        neural_only=positive,
                        pca_only=negative,
                        difference_known_contribution=(positive - negative) / n,
                        identification_bounds=[
                            (positive - negative - missing) / n,
                            (positive - negative + missing) / n,
                        ],
                        marginal_95_interval=difference_interval(positive, negative, n, 0.05),
                        simultaneous_95_interval=difference_interval(
                            positive, negative, n, 0.05 / 4
                        ),
                        simultaneous_failure_interval=paired_interval(
                            positive, negative, missing, n, 0.05 / 4
                        ),
                    )
                )
        precision = precision_summary(root, source, config, binding)
        summary = dict(
            schema_version=1,
            fixture=config.fixture,
            observations=len(tasks(config, "main")),
            observations_per_law=1 if config.fixture else 200,
            n=32 if config.fixture else 64,
            primary_rates=8,
            secondary_paired_comparisons=4,
            counts=frozen["counts"],
            main_units=len(all_jobs(root, config, "main")),
            audits={
                phase: {
                    k: v
                    for k, v in read_json(root / (phase + "-audit.json")).items()
                    if k != "records"
                }
                for phase in ("preflight", "main")
            },
            defaults_changed=False,
            historical_pvalues_changed=False,
            author_contact=False,
            calibration_scope=(
                "Approximate fitted raw-Gaussian departure; not selected-K/species significance"
            ),
            interpretation=(
                "Tested Gaussian laws, N=64, one fixed initialization; no general size guarantee"
            ),
        )
        for name, value in (
            ("assessments", rows),
            ("primary-rates", primary),
            ("groups", groups),
            ("paired-comparisons", paired),
            ("precision-diagnostic", precision),
            ("summary", summary),
        ):
            atomic_json(root / (name + ".json"), value)
        return summary
