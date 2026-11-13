"""Conditional archived-head feasibility, repeated curves and a frozen stopping flag."""

from pathlib import Path

from super_deli.benchmarking.abstention_config import ARMS
from super_deli.benchmarking.local_reference_report import rate
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json, verify_seal
from super_deli.benchmarking.observed_feasibility_config import THRESHOLDS, counts, settings
from super_deli.benchmarking.observed_feasibility_study import engine
from super_deli.benchmarking.observed_feasibility_tree import at_threshold
from super_deli.benchmarking.residual_report import distribution
from super_deli.io.config import read_json

POLICIES = ("unscreened", "probability", "distance", "local", "diagnostic")


def outcome(row, policy):
    return row["joint_targets"][policy] if row["status"] == "available" else None


def points(rows, case, n, arm):
    return [
        next(r for r in o["assessments"] if r["arm"] == arm)
        for o in rows
        if (o["task"]["case"], o["task"]["n"]) == (case, n)
    ]


def summarize(rows):
    groups = []
    classwise = []
    primary = []
    paired = []
    curve_groups = []
    qualification = []
    cells = sorted({(v["task"]["case"], v["task"]["n"]) for v in rows})
    for case, n in cells:
        for arm in ARMS:
            selected = points(rows, case, n, arm)
            good = [v for v in selected if v["status"] == "available"]
            identity = dict(case=case, n=n, arm=arm)
            qualification.append(
                dict(
                    **identity,
                    requested=len(selected),
                    qualified=sum(v["selection"]["status"] == "selected" for v in good),
                    no_operating_point=sum(
                        v["selection"]["status"] == "available_no_operating_point" for v in good
                    ),
                    unavailable=len(selected) - len(good),
                    thresholds=distribution([v["selection"]["threshold"] for v in good]),
                )
            )
            for policy in POLICIES:
                values = [v["metrics"][policy] for v in good if v["metrics"][policy] is not None]
                group = dict(
                    **identity,
                    policy=policy,
                    requested=len(selected),
                    available_operating_points=len(values),
                    no_operating_point=len(good) - len(values),
                    numerical_unavailable=len(selected) - len(good),
                    cohorts={},
                )
                for c in ("iid", "near", "far", "overlap", "shifted"):
                    metrics = [v[c] for v in values if c in v]
                    if not metrics:
                        continue
                    keys = sorted(
                        {
                            k
                            for m in metrics
                            for k, value in m.items()
                            if type(value) in (int, float)
                        }
                    )
                    group["cohorts"][c] = {
                        k: distribution([m.get(k) for m in metrics]) for k in keys
                    }
                    if c in ("iid", "shifted"):
                        for label in range(4):
                            members = [m["per_class"][label] for m in metrics]
                            classwise.append(
                                dict(
                                    **identity,
                                    policy=policy,
                                    cohort=c,
                                    reference=label,
                                    requested=len(selected),
                                    metrics={
                                        k: distribution([m.get(k) for m in members])
                                        for k in (
                                            "n",
                                            "accepted",
                                            "accepted_accuracy",
                                            "true_reference_included",
                                        )
                                    },
                                )
                            )
                groups.append(group)
                if arm in ("logistic", "residual_0") and policy != "unscreened":
                    primary.append(
                        dict(
                            **identity,
                            policy=policy,
                            measure="external_joint_target",
                            **rate([outcome(v, policy) for v in selected]),
                        )
                    )
            if arm in ("logistic", "residual_0"):
                for baseline in ("probability", "distance", "local"):
                    a = [outcome(v, "diagnostic") for v in selected]
                    b = [outcome(v, baseline) for v in selected]
                    common = [
                        (x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None
                    ]
                    row = dict(
                        **identity,
                        baseline=baseline,
                        requested=len(selected),
                        available=len(common),
                        diagnostic_only=sum(x and not y for x, y in common),
                        baseline_only=sum(y and not x for x, y in common),
                        both=sum(x and y for x, y in common),
                        neither=sum(not x and not y for x, y in common),
                        changes={},
                    )
                    for c, k in (
                        ("iid", "accepted_fraction"),
                        ("iid", "accepted_accuracy"),
                        ("iid", "incorrect_accepted_fraction"),
                        ("near", "accepted_fraction"),
                        ("far", "accepted_fraction"),
                    ):
                        v = [
                            p["metrics"]["diagnostic"][c][k] - p["metrics"][baseline][c][k]
                            for p in good
                            if p["metrics"]["diagnostic"] is not None
                            and p["metrics"]["diagnostic"][c][k] is not None
                            and p["metrics"][baseline][c][k] is not None
                        ]
                        row["changes"][c + "_" + k] = distribution(v)
                    paired.append(row)
            for i, t in enumerate(THRESHOLDS):
                curve = dict(**identity, threshold=t, requested=len(selected), cohorts={})
                for c in ("iid", "near", "far", "overlap", "shifted"):
                    values = [
                        at_threshold(v["evaluation_curves"][c], i)
                        for v in good
                        if c in v["evaluation_curves"]
                    ]
                    if values:
                        keys = sorted({k for v in values for k in v})
                        curve["cohorts"][c] = {
                            k: distribution([v.get(k) for v in values]) for k in keys
                        }
                curve_groups.append(curve)
    return dict(
        groups=groups,
        classwise=classwise,
        primary_rates=primary,
        paired_contrasts=paired,
        curve_groups=curve_groups,
        qualification=qualification,
    )


def focal_flag(rows):
    spec = settings()["focal"]
    selected = points(rows, spec["case"], spec["n"], spec["arm"])
    if not selected:
        return dict(
            assessed=False, flag=False, reason="Invented fixture lacks scientific focal cell"
        )
    good = [r for r in selected if r["status"] == "available"]
    qualified = sum(r["selection"]["status"] == "selected" for r in good)
    successes = sum(r["joint_targets"]["diagnostic"] for r in good)
    wins = sum(r["joint_targets"]["diagnostic"] and not r["joint_targets"]["local"] for r in good)
    reverse = sum(
        r["joint_targets"]["local"] and not r["joint_targets"]["diagnostic"] for r in good
    )
    flag = len(selected) == 10 and qualified >= 8 and successes >= 5 and wins >= 3 and reverse <= 1
    return dict(
        assessed=True,
        case=spec["case"],
        n=spec["n"],
        arm=spec["arm"],
        requested=len(selected),
        qualified=qualified,
        external_joint_successes=successes,
        diagnostic_only=wins,
        reverse=reverse,
        unavailable=len(selected) - len(good),
        flag=flag,
        decision=(
            "Favourable measured-space gap under explicit unfamiliar exposure; "
            "separate deployment proposal required"
        )
        if flag
        else (
            "No prespecified focal gap demonstrated; close this support-rule tuning cycle "
            "without claiming impossibility"
        ),
        interpretation=(
            "Descriptive allocation rule, not a test or attainable bound; "
            "other cells cannot replace this focal cell"
        ),
    )


def report(root, source):
    root = Path(root)
    study = engine(root, source)
    _, config = study.check()
    study.binding()
    audit = read_json(root / "main-audit.json")
    require(audit["passed"], "Completed independent main audit")
    rows = []
    for job in study.jobs(config, "main"):
        kept = verify_seal(root, "main", job)
        if job["kind"] == "fresh":
            require(not kept["interrupted"], "Numerically complete diagnostic observations")
            rows.append(
                dict(
                    task=job["task"],
                    source_path=job["entry"]["path"],
                    assessments=read_json(root / "main" / job["task"]["key"] / "assessment.json"),
                )
            )
    expected = counts(config, "main")
    execution = read_json(root / "main-execution.json")
    require(
        len(rows) == expected["observations"]
        and execution["completed"]
        and execution["detector_fits"] == expected["detector_fits"]
        and execution["trees"] == expected["trees"]
        and execution["failed_points"] == execution["unknown_counter_units"] == 0,
        "Retained complete scientific budget",
    )
    values = summarize(rows)
    flag = focal_flag(rows)
    summary = dict(
        completed=True,
        observations=len(rows),
        source_heads_retrained=0,
        main_detector_fits=expected["detector_fits"],
        preflight_detector_fits=counts(config, "preflight")["detector_fits"],
        main_trees=expected["trees"],
        preflight_trees=counts(config, "preflight")["trees"],
        primary_rates=len(values["primary_rates"]),
        paired_contrasts=len(values["paired_contrasts"]),
        failed_workflows=0,
        focal_flag=flag["flag"],
        default_changes=False,
        pvalue_changes=False,
        author_contact=False,
        scope=(
            "Favourable measured-feature diagnostic conditional on archived heads; "
            "unknown training examples supplied; no deployment/species guarantee"
        ),
    )
    atomic_json(root / "observations.json", rows)
    for name, value in values.items():
        atomic_json(root / (name.replace("_", "-") + ".json"), value)
    atomic_json(root / "stopping-assessment.json", flag)
    atomic_json(root / "summary.json", summary)
    atomic_json(
        root / "study-design.json",
        dict(
            settings=settings(),
            configuration=config.to_dict(),
            counts={p: counts(config, p) for p in ("preflight", "main")},
            independent_units=(
                "Archived head observations with fresh independent diagnostic roles; "
                "seeds/policies/thresholds repeat observations"
            ),
            score_information=(
                "Measured traits only; specified unfamiliar examples and extra labelled specimens "
                "are deliberately favourable information"
            ),
        ),
    )
    return summary
