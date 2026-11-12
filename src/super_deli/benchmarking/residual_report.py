"""Separate reference assignment/discovery summaries; no significance recalibration."""

from pathlib import Path

import numpy as np
from scipy.stats import beta

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json, verify_seal
from super_deli.benchmarking.residual_config import CASES
from super_deli.benchmarking.residual_study import engine
from super_deli.io.config import read_json


def interval(k, n, alpha):
    return [
        0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k)),
    ]


def rate(values):
    n = len(values)
    missing = sum(v is None for v in values)
    count = sum(v is True for v in values)
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
            interval(count, n, 0.05 / 40)[0],
            interval(count + missing, n, 0.05 / 40)[1],
        ],
    )


def distribution(values):
    good = [float(v) for v in values if v is not None]
    return dict(
        available=len(good),
        mean=float(np.mean(good)) if good else None,
        median=float(np.median(good)) if good else None,
        minimum=min(good) if good else None,
        maximum=max(good) if good else None,
    )


def report(root, source):
    root = Path(root)
    study = engine(root, source)
    frozen, config = study.check()
    study.binding()
    audit = read_json(root / "main-audit.json")
    require(audit["passed"], "Main independent audit required")
    rows = []
    for job in study.jobs(config, "main"):
        verify_seal(root, "main", job)
        if job["kind"] == "fresh":
            folder = root / "main" / job["task"]["key"]
            rows.append(dict(task=job["task"], fits=read_json(folder / "fits.json")))
    groups, rates = [], []
    for case in CASES if not config.fixture else ("linear_standard",):
        for n in (64, 256) if not config.fixture else (32,):
            observations = [r for r in rows if r["task"]["case"] == case and r["task"]["n"] == n]
            grouped = dict(case=case, n=n, requested=len(observations), arms=[])
            by_arm = {
                arm: []
                for arm in [r["arm"] for r in observations[0]["fits"]]
                if not arm.startswith("permuted_")
            }
            for observation in observations:
                for r in observation["fits"]:
                    if r["arm"] in by_arm:
                        by_arm[r["arm"]].append(r)
            for arm, values in by_arm.items():
                group = dict(
                    arm=arm,
                    task=values[0]["task"],
                    available=sum(r["status"] == "available" for r in values),
                    seconds=distribution([r["seconds"] for r in values]),
                    parameter_count=distribution([r.get("parameter_count") for r in values]),
                    cohorts={},
                )
                for cohort in ("iid", "shifted", "unfamiliar"):
                    metrics = [r.get("metrics", {}).get(cohort) for r in values]
                    if not any(m is not None for m in metrics):
                        continue
                    names = sorted(
                        {k for m in metrics if m for k, v in m.items() if type(v) in (int, float)}
                    )
                    group["cohorts"][cohort] = {
                        name: distribution([m.get(name) if m else None for m in metrics])
                        for name in names
                    }
                grouped["arms"].append(group)
                if case not in ("gaussian", "gradient") and arm in ("pca4", "neural_0"):
                    rates.append(
                        dict(
                            case=case,
                            n=n,
                            arm=arm,
                            metric="four_group_recovery",
                            **rate(
                                [
                                    bool(r["metrics"]["iid"]["four_group_recovery"])
                                    if r["status"] == "available"
                                    else None
                                    for r in values
                                ]
                            ),
                        )
                    )
                if case == "gaussian" and arm in ("pca4", "neural_0"):
                    rates.append(
                        dict(
                            case=case,
                            n=n,
                            arm=arm,
                            metric="gaussian_split",
                            **rate(
                                [
                                    bool(r["selected_k"] > 1)
                                    if r["status"] == "available"
                                    else None
                                    for r in values
                                ]
                            ),
                        )
                    )
            if case not in ("gaussian", "gradient"):
                paired = []
                for observation in observations:
                    fits = {r["arm"]: r for r in observation["fits"]}
                    a, b = fits["residual_0"], fits["logistic"]
                    paired.append(
                        a["metrics"]["iid"]["balanced_accuracy"]
                        - b["metrics"]["iid"]["balanced_accuracy"]
                        if a["status"] == b["status"] == "available"
                        else None
                    )
                grouped["paired_primary_accuracy_gain"] = distribution(paired)
                grouped["paired_accuracy_gain_values"] = paired
                rates.append(
                    dict(
                        case=case,
                        n=n,
                        arm="residual_0_vs_logistic",
                        metric="five_percentage_point_gain",
                        **rate([bool(v >= 0.05) if v is not None else None for v in paired]),
                    )
                )
            groups.append(grouped)
    require(config.fixture or len(rates) == 40, "All forty prespecified rates")
    execution = read_json(root / "main-execution.json")
    result = dict(
        observations=len(rows),
        groups=len(groups),
        primary_rates=len(rates),
        workflows=execution["attempted"],
        neural_trainings=execution["neural_fits"],
        failed_workflows=execution["failed_points"],
        verified_optimizer_updates=audit["verified_optimizer_updates"],
        independent_datasets_per_cell=1 if config.fixture else 5,
        neural_seeds=[0, 101, 202],
        primary_seed=0,
        default_changes=False,
        pvalue_changes=False,
        species_significance="not_computed",
        author_contact=False,
        interpretation=[
            "Supervised reference assignment and unsupervised discovery are separate tasks",
            "Allometry and saturation are illustrative measurement laws, not empirical biology",
            "Unfamiliar-class confidence does not implement unknown-class rejection",
            "Five datasets/cell support a descriptive pilot; no automatic model promotion",
        ],
    )
    atomic_json(root / "assessments.json", rows)
    atomic_json(root / "groups.json", groups)
    atomic_json(root / "primary-rates.json", rates)
    atomic_json(root / "summary.json", result)
    return result
