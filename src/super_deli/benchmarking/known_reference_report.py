"""Parent-level held-out-group operating rates, paired bounds and stopping decision."""

from pathlib import Path

from super_deli.benchmarking.known_reference_config import ARMS, POLICIES, counts, settings
from super_deli.benchmarking.known_reference_study import engine
from super_deli.benchmarking.local_reference_report import rate
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json, verify_seal
from super_deli.benchmarking.residual_report import distribution, interval
from super_deli.io.config import read_json


def secondary_rate(values):
    """Secondary holdout/unscreened intervals are marginal, outside the primary family."""
    value = rate(values)
    value.pop("simultaneous_95_interval")
    return value


def fold_outcome(row, policy):
    return (
        row["joint_targets"][policy]
        if row["status"] == "available" and row["policy_status"][policy] == "available"
        else None
    )


def parent_outcome(fits, arm, policy):
    selected = [r for r in fits if r["arm"] == arm]
    require(
        len(selected) == 4 and {r["heldout"] for r in selected} == set(range(4)),
        "Four distinct held-out measurements per parent/head",
    )
    values = [fold_outcome(r, policy) for r in selected]
    k = sum(v is True for v in values)
    missing = sum(v is None for v in values)
    return True if k >= 3 else False if k + missing < 3 else None


def paired(a, b):
    require(len(a) == len(b) and len(a) > 0, "Requested paired parent denominators")
    common = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    n = len(a)
    missing = n - len(common)
    forward = sum(x and not y for x, y in common)
    reverse = sum(y and not x for x, y in common)

    def bounds(alpha):
        return [
            max(
                -1,
                interval(forward, n, alpha / 2)[0] - interval(reverse + missing, n, alpha / 2)[1],
            ),
            min(
                1, interval(forward + missing, n, alpha / 2)[1] - interval(reverse, n, alpha / 2)[0]
            ),
        ]

    return dict(
        requested=n,
        available=len(common),
        unavailable=missing,
        new_only=forward,
        baseline_only=reverse,
        both=sum(x and y for x, y in common),
        neither=sum(not x and not y for x, y in common),
        difference_failure_bounds=[
            (forward - reverse - missing) / n,
            (forward - reverse + missing) / n,
        ],
        descriptive_95_interval=bounds(0.05),
        simultaneous_95_interval=bounds(0.05 / 56),
        interval_method=(
            "Conservative paired discordance bounds; exact binomial endpoints and union bound"
        ),
    )


def summarize(rows):
    cells = sorted({(v["task"]["case"], v["task"]["n"]) for v in rows})
    groups = []
    fold_groups = []
    classwise = []
    primary = []
    contrasts = []
    for case, n in cells:
        parents = [v for v in rows if (v["task"]["case"], v["task"]["n"]) == (case, n)]
        for arm in ARMS:
            for policy in POLICIES:
                identity = dict(case=case, n=n, arm=arm, policy=policy)
                pooled = dict(
                    **identity,
                    requested_parents=len(parents),
                    cohorts={},
                    available_folds=distribution(
                        [
                            sum(
                                r["arm"] == arm
                                and r["status"] == "available"
                                and r["policy_status"][policy] == "available"
                                for r in parent["fits"]
                            )
                            for parent in parents
                        ]
                    ),
                    continuous_summary=(
                        "Each parent averages available folds; no pooled-fold independence"
                    ),
                )
                cohort_names = sorted(
                    {
                        name
                        for p in parents
                        for r in p["fits"]
                        if r["arm"] == arm
                        for name in r.get("metrics", {}).get(policy, {})
                    }
                )
                for name in cohort_names:
                    entries = [
                        r["metrics"][policy][name]
                        for p in parents
                        for r in p["fits"]
                        if r["arm"] == arm and r["policy_status"][policy] == "available"
                    ]
                    numeric = sorted(
                        {k for m in entries for k, v in m.items() if type(v) in (int, float)}
                    )
                    aggregate = {}
                    for metric in numeric:
                        means = []
                        for parent in parents:
                            values = [
                                r["metrics"][policy][name].get(metric)
                                for r in parent["fits"]
                                if r["arm"] == arm and r["policy_status"][policy] == "available"
                            ]
                            values = [v for v in values if v is not None]
                            means.append(sum(values) / len(values) if values else None)
                        aggregate[metric] = distribution(means)
                    pooled["cohorts"][name] = aggregate
                outcomes = [parent_outcome(p["fits"], arm, policy) for p in parents]
                pooled["three_of_four_joint_target"] = (
                    rate(outcomes) if policy != "unscreened" else secondary_rate(outcomes)
                )
                groups.append(pooled)
                if policy != "unscreened":
                    primary.append(
                        dict(
                            **identity,
                            measure="three_of_four_heldout_folds_joint_target",
                            **rate(outcomes),
                        )
                    )
                for h in range(4):
                    points = [
                        next(r for r in p["fits"] if r["arm"] == arm and r["heldout"] == h)
                        for p in parents
                    ]
                    available = [
                        r
                        for r in points
                        if r["status"] == "available" and r["policy_status"][policy] == "available"
                    ]
                    entry = dict(
                        **identity,
                        heldout=h,
                        requested=len(points),
                        available=len(available),
                        joint_target=secondary_rate([fold_outcome(r, policy) for r in points]),
                        cohorts={},
                    )
                    for name in cohort_names:
                        ms = [r["metrics"][policy][name] for r in available]
                        if not ms:
                            continue
                        keys = sorted(
                            {k for m in ms for k, v in m.items() if type(v) in (int, float)}
                        )
                        entry["cohorts"][name] = {
                            k: distribution([m.get(k) for m in ms]) for k in keys
                        }
                        if name in ("iid", "shifted"):
                            known = [c for c in range(4) if c != h]
                            for c, original in enumerate(known):
                                classwise.append(
                                    dict(
                                        **identity,
                                        heldout=h,
                                        cohort=name,
                                        original_reference=original,
                                        requested=len(points),
                                        metrics={
                                            k: distribution([m["per_class"][c].get(k) for m in ms])
                                            for k in (
                                                "n",
                                                "accepted",
                                                "accepted_accuracy",
                                                "true_reference_included",
                                            )
                                        },
                                    )
                                )
                    fold_groups.append(entry)
            a = [parent_outcome(p["fits"], arm, "density") for p in parents]
            for baseline in ("probability", "distance", "local"):
                b = [parent_outcome(p["fits"], arm, baseline) for p in parents]
                contrasts.append(
                    dict(case=case, n=n, arm=arm, baseline=baseline, kind="policy", **paired(a, b))
                )
        a = [parent_outcome(p["fits"], "residual_0", "density") for p in parents]
        b = [parent_outcome(p["fits"], "logistic", "density") for p in parents]
        contrasts.append(
            dict(
                case=case,
                n=n,
                kind="classifier",
                arm="residual_0",
                baseline="logistic",
                **paired(a, b),
            )
        )
    focal = [p for p in rows if (p["task"]["case"], p["task"]["n"]) == ("combined", 256)]
    if focal:
        a = [parent_outcome(p["fits"], "residual_0", "density") for p in focal]
        b = [parent_outcome(p["fits"], "residual_0", "local") for p in focal]
        pair = paired(a, b)
        success = sum(v is True for v in a)
        flag = (
            success >= 5
            and pair["new_only"] >= 3
            and pair["baseline_only"] <= 1
            and pair["unavailable"] == 0
        )
        stopping = dict(
            assessed=True,
            case="combined",
            n=256,
            arm="residual_0",
            requested=len(focal),
            parent_successes=success,
            **{k: pair[k] for k in ("new_only", "baseline_only", "unavailable")},
            flag=flag,
            decision="Favourable bounded known-only gap; no default promotion"
            if flag
            else "No prespecified focal gap; close this reference-support tuning cycle",
            interpretation=(
                "Descriptive allocation rule, no significance/species guarantee; stop regardless"
            ),
        )
    else:
        stopping = dict(
            assessed=False, flag=None, reason="Invented fixture excludes scientific focal cell"
        )
    return dict(
        groups=groups,
        fold_groups=fold_groups,
        classwise=classwise,
        primary_rates=primary,
        paired_contrasts=contrasts,
        stopping_assessment=stopping,
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
    require(
        len(rows) == counts(config, "main")["observations"], "All requested independent parents"
    )
    values = summarize(rows)
    require(
        config.fixture
        or (len(values["primary_rates"]), len(values["paired_contrasts"])) == (64, 56),
        "Frozen rate/contrast families",
    )
    execution = read_json(root / "main-execution.json")
    summary = dict(
        completed=True,
        observations=len(rows),
        ordinary_heldout_folds=4 * len(rows),
        main_classifier_fits=execution["attempted"],
        main_neural_trainings=execution["neural_fits"],
        audited_optimizer_updates=audit["verified_optimizer_updates"],
        failed_workflows=execution["failed_points"],
        unknown_training_examples=0,
        threshold_search=False,
        primary_rates=len(values["primary_rates"]),
        paired_contrasts=len(values["paired_contrasts"]),
        focal_flag=values["stopping_assessment"]["flag"],
        default_changes=False,
        pvalue_changes=False,
        author_contact=False,
        species_significance="not_computed",
        interpretation=(
            "Known-only reference support; folds repeat "
            "parents; no general unknown/species guarantee"
        ),
    )
    atomic_json(root / "observations.json", rows)
    for name, value in values.items():
        atomic_json(root / (name.replace("_", "-") + ".json"), value)
    atomic_json(
        root / "study-design.json",
        dict(
            settings=settings(),
            configuration=config.to_dict(),
            counts={p: counts(config, p) for p in ("preflight", "main")},
            independent_units="Fresh parent datasets; four exclusions are repeated measurements",
            training_panel_note="Original panel64/256 becomes known training48/192",
        ),
    )
    atomic_json(root / "summary.json", summary)
    return summary
