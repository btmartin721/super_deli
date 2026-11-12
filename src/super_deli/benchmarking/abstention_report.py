"""Dataset-level paired coverage/error tradeoffs; no automatic policy selection."""

from pathlib import Path

from super_deli.benchmarking.abstention_config import ARMS, CASES, POLICIES
from super_deli.benchmarking.abstention_study import engine
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json, verify_seal
from super_deli.benchmarking.residual_report import distribution, interval
from super_deli.io.config import read_json


def rate(values):
    n = len(values)
    missing, count = sum(v is None for v in values), sum(v is True for v in values)
    return dict(
        count=count,
        requested=n,
        unavailable=missing,
        failure_bounds=[count / n, (count + missing) / n],
        descriptive_95_interval=[
            interval(count, n, 0.05)[0],
            interval(count + missing, n, 0.05)[1],
        ],
        simultaneous_95_interval=[
            interval(count, n, 0.05 / 32)[0],
            interval(count + missing, n, 0.05 / 32)[1],
        ],
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
        verify_seal(root, "main", job)
        if job["kind"] == "fresh":
            rows.append(
                dict(
                    task=job["task"],
                    fits=read_json(root / "main" / job["task"]["key"] / "fits.json"),
                )
            )
    groups, primary, paired = [], [], []
    for case in ("linear_standard",) if config.fixture else CASES:
        for n in (32,) if config.fixture else (64, 256):
            selected = [r for r in rows if r["task"]["case"] == case and r["task"]["n"] == n]
            group = dict(case=case, n=n, requested=len(selected), arms=[])
            for arm in ARMS:
                values = [
                    next(r for r in observation["fits"] if r["arm"] == arm)
                    for observation in selected
                ]
                for policy in POLICIES:
                    summaries = dict(
                        arm=arm,
                        policy=policy,
                        available=sum(r["status"] == "available" for r in values),
                        cohorts={},
                    )
                    for cohort in ("iid", "near", "far", "overlap", "shifted"):
                        measurements = [
                            r.get("metrics", {}).get(policy, {}).get(cohort)
                            if r["status"] == "available"
                            else None
                            for r in values
                        ]
                        if not any(v is not None for v in measurements):
                            continue
                        names = sorted(
                            {
                                k
                                for m in measurements
                                if m
                                for k, v in m.items()
                                if type(v) in (int, float)
                            }
                        )
                        summaries["cohorts"][cohort] = {
                            name: distribution([m.get(name) if m else None for m in measurements])
                            for name in names
                        }
                        summaries["cohorts"][cohort]["withheld_reasons"] = {
                            reason: distribution(
                                [m["reasons"][reason] / m["n"] if m else None for m in measurements]
                            )
                            for reason in (
                                "assigned",
                                "withheld_outside_reference_support",
                                "withheld_ambiguous_reference_support",
                                "withheld_classifier_support_conflict",
                            )
                        }
                    group["arms"].append(summaries)
                    outcomes = [
                        r["joint_targets"][policy] if r["status"] == "available" else None
                        for r in values
                    ]
                    summaries["joint_target"] = rate(outcomes)
                    if arm in ("logistic", "residual_0") and policy != "unscreened":
                        primary.append(
                            dict(case=case, n=n, arm=arm, policy=policy, **rate(outcomes))
                        )
                if arm in ("logistic", "residual_0"):
                    available = [r for r in values if r["status"] == "available"]
                    paired.append(
                        dict(
                            case=case,
                            n=n,
                            arm=arm,
                            requested=len(values),
                            available=len(available),
                            distance_only=sum(
                                r["joint_targets"]["distance"]
                                and not r["joint_targets"]["probability"]
                                for r in available
                            ),
                            probability_only=sum(
                                r["joint_targets"]["probability"]
                                and not r["joint_targets"]["distance"]
                                for r in available
                            ),
                            both=sum(
                                r["joint_targets"]["probability"] and r["joint_targets"]["distance"]
                                for r in available
                            ),
                        )
                    )
            groups.append(group)
    require(
        config.fixture or len(primary) == 32 and len(paired) == 16,
        "Complete prespecified rate/contrast families",
    )
    execution = read_json(root / "main-execution.json")
    result = dict(
        observations=len(rows),
        groups=len(groups),
        primary_rates=len(primary),
        paired_contrasts=len(paired),
        workflows=execution["attempted"],
        neural_trainings=execution["neural_fits"],
        failed_workflows=execution["failed_points"],
        verified_optimizer_updates=audit["verified_optimizer_updates"],
        independent_datasets_per_cell=1 if config.fixture else 10,
        default_changes=False,
        pvalue_changes=False,
        author_contact=False,
        species_significance="not_computed",
        interpretation=[
            "Compatibility and abstention are not species discovery",
            "Known-class inclusion assumptions do not guarantee unfamiliar rejection",
            "Indistinguishable unfamiliar groups cannot be separated using these measurements",
            "No policy, seed or default automatically selected",
        ],
    )
    for name, value in (
        ("assessments.json", rows),
        ("groups.json", groups),
        ("primary-rates.json", primary),
        ("paired-contrasts.json", paired),
        ("summary.json", result),
    ):
        atomic_json(root / name, value)
    return result
