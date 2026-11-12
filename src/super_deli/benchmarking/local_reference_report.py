"""Fresh paired operating rates, archival descriptions and separate true-law sweeps."""

from fractions import Fraction
from pathlib import Path

from super_deli.benchmarking.abstention_config import ARMS
from super_deli.benchmarking.local_reference_config import POLICIES, counts
from super_deli.benchmarking.local_reference_study import engine
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json, verify_seal
from super_deli.benchmarking.residual_report import distribution, interval
from super_deli.io.config import read_json


def rate(values):
    n = len(values)
    missing = sum(v is None for v in values)
    k = sum(v is True for v in values)
    return dict(
        count=k,
        requested=n,
        unavailable=missing,
        failure_bounds=[k / n, (k + missing) / n],
        descriptive_95_interval=[interval(k, n, 0.05)[0], interval(k + missing, n, 0.05)[1]],
        simultaneous_95_interval=[
            interval(k, n, 0.05 / 64)[0],
            interval(k + missing, n, 0.05 / 64)[1],
        ],
    )


def available(row, policy):
    return row["status"] == "available" and (
        policy != "local" or row.get("local_status") == "available"
    )


def outcome(row, policy):
    return row["joint_targets"][policy] if available(row, policy) else None


def benefit(row):
    if not available(row, "local") or not available(row, "distance"):
        return None
    new, old = row["metrics"]["local"], row["metrics"]["distance"]
    n, o = new["iid"], old["iid"]
    if not n["accepted"] or not o["accepted"]:
        return False

    def accuracy(m):
        return Fraction(round(m["correct_accepted_fraction"] * m["n"]), m["accepted"])

    def coverage(m):
        return Fraction(m["accepted"], m["n"])

    return bool(
        coverage(n) - coverage(o) >= Fraction(1, 10)
        and accuracy(n) >= Fraction(9, 10)
        and accuracy(o) - accuracy(n) <= Fraction(1, 100)
        and all(coverage(new[c]) - coverage(old[c]) <= Fraction(1, 50) for c in ("near", "far"))
    )


def summarize(rows):
    groups = []
    classwise = []
    cells = sorted({(r["task"]["case"], r["task"]["n"]) for r in rows})
    for case, n in cells:
        selected = [r for r in rows if (r["task"]["case"], r["task"]["n"]) == (case, n)]
        for arm in ARMS:
            points = [next(v for v in r["fits"] if v["arm"] == arm) for r in selected]
            for policy in POLICIES:
                good = [r for r in points if available(r, policy)]
                identity = dict(case=case, n=n, arm=arm, policy=policy)
                item = dict(**identity, requested=len(points), available=len(good), cohorts={})
                for cohort in ("iid", "near", "far", "overlap", "shifted"):
                    values = [
                        r["metrics"][policy][cohort] for r in good if cohort in r["metrics"][policy]
                    ]
                    if not values:
                        continue
                    names = sorted(
                        {k for m in values for k, v in m.items() if type(v) in (int, float)}
                    )
                    item["cohorts"][cohort] = {
                        k: distribution([m.get(k) for m in values]) for k in names
                    }
                    item["cohorts"][cohort]["withheld_reasons"] = {
                        s: distribution([m["reasons"][s] / m["n"] for m in values])
                        for s in values[0]["reasons"]
                    }
                    if cohort in ("iid", "shifted"):
                        for c in range(4):
                            entries = [m["per_class"][c] for m in values]
                            classwise.append(
                                dict(
                                    **identity,
                                    cohort=cohort,
                                    reference=c,
                                    requested=len(points),
                                    metrics={
                                        k: distribution([v[k] for v in entries])
                                        for k in (
                                            "n",
                                            "accepted",
                                            "accepted_accuracy",
                                            "true_reference_included",
                                        )
                                    },
                                )
                            )
                groups.append(item)
    return groups, classwise


def oracle_summary(rows):
    output = []
    for case, n in sorted({(r["task"]["case"], r["task"]["n"]) for r in rows}):
        selected = [r for r in rows if (r["task"]["case"], r["task"]["n"]) == (case, n)]
        for i in range(19):
            row = dict(
                case=case,
                n=n,
                nominal_withholding=(i + 1) / 20,
                requested=len(selected),
                cohorts={},
            )
            for cohort in ("iid", "near", "far", "overlap", "shifted"):
                values = [
                    r["oracle"]["thresholds"][i]["cohorts"][cohort]
                    for r in selected
                    if cohort in r["oracle"]["thresholds"][i]["cohorts"]
                ]
                if values:
                    row["cohorts"][cohort] = {
                        k: distribution([v.get(k) for v in values])
                        for k in (
                            "accepted_fraction",
                            "accepted_accuracy",
                            "incorrect_accepted_fraction",
                            "raw_accuracy",
                        )
                        if any(k in v for v in values)
                    }
            output.append(row)
    return output


def report(root, source):
    root = Path(root)
    study = engine(root, source)
    frozen, config = study.check()
    study.binding()
    fresh = []
    archived = []
    for phase in ("preflight", "main"):
        audit = read_json(root / (phase + "-audit.json"))
        require(audit["passed"], "Both independent phase audits required")
        for job in study.jobs(config, phase):
            verify_seal(root, phase, job)
            if job["kind"] == "replay":
                continue
            folder = root / phase / job["task"]["key"]
            row = dict(
                task=job["task"],
                fits=read_json(folder / "fits.json"),
                oracle=read_json(folder / "oracle-result.json"),
            )
            if job["kind"] == "archive":
                archived.append(row)
            elif phase == "main":
                fresh.append(row)
    groups, classwise = summarize(fresh)
    archive_groups, archive_classes = summarize(archived)
    primary = []
    paired = []
    for case, n in sorted({(r["task"]["case"], r["task"]["n"]) for r in fresh}):
        selected = [r for r in fresh if (r["task"]["case"], r["task"]["n"]) == (case, n)]
        for arm in ("logistic", "residual_0"):
            points = [next(v for v in r["fits"] if v["arm"] == arm) for r in selected]
            for policy in ("probability", "distance", "local"):
                primary.append(
                    dict(
                        case=case,
                        n=n,
                        arm=arm,
                        measure="joint_target",
                        policy=policy,
                        **rate([outcome(p, policy) for p in points]),
                    )
                )
            primary.append(
                dict(
                    case=case,
                    n=n,
                    arm=arm,
                    measure="bounded_coverage_benefit",
                    policy="local_vs_distance",
                    **rate([benefit(p) for p in points]),
                )
            )
            for baseline in ("distance", "probability"):
                common = [p for p in points if available(p, "local") and available(p, baseline)]
                a = [outcome(p, "local") for p in common]
                b = [outcome(p, baseline) for p in common]
                changes = {}
                for cohort, key in (
                    ("iid", "accepted_fraction"),
                    ("iid", "accepted_accuracy"),
                    ("iid", "incorrect_accepted_fraction"),
                    ("near", "accepted_fraction"),
                    ("far", "accepted_fraction"),
                ):
                    changes[cohort + "_" + key] = distribution(
                        [
                            p["metrics"]["local"][cohort][key] - p["metrics"][baseline][cohort][key]
                            for p in common
                            if p["metrics"]["local"][cohort][key] is not None
                            and p["metrics"][baseline][cohort][key] is not None
                        ]
                    )
                paired.append(
                    dict(
                        case=case,
                        n=n,
                        arm=arm,
                        baseline=baseline,
                        requested=len(points),
                        available=len(common),
                        local_only=sum(x and not y for x, y in zip(a, b, strict=True)),
                        baseline_only=sum(y and not x for x, y in zip(a, b, strict=True)),
                        both=sum(x and y for x, y in zip(a, b, strict=True)),
                        neither=sum(not x and not y for x, y in zip(a, b, strict=True)),
                        changes=changes,
                    )
                )
    main = read_json(root / "main-execution.json")
    pre = read_json(root / "preflight-execution.json")
    require(
        main["attempted"] == counts(config, "main")["workflows"]
        and pre["attempted"] == counts(config, "preflight")["workflows"],
        "Exact requested model-workflow accounting",
    )
    summary = dict(
        completed=True,
        observations=len(fresh),
        archived_observations=len(archived),
        main_model_workflows=main["attempted"],
        main_classifier_trainings=counts(config, "main")["classifier_trainings"],
        preflight_classifier_trainings=counts(config, "preflight")["classifier_trainings"],
        main_neural_fits=main["neural_fits"],
        preflight_neural_fits=pre["neural_fits"],
        verified_optimizer_updates=sum(
            read_json(root / (p + "-audit.json"))["verified_optimizer_updates"]
            for p in ("preflight", "main")
        ),
        failed_workflows=main["failed_points"] + pre["failed_points"],
        primary_rates=len(primary),
        paired_comparisons=len(paired),
        fresh_policy_evaluations=sum(
            r["policy_evaluations"]
            for r in read_json(root / "main-audit.json")["records"]
            if not r["key"].startswith(("archive_", "replay_"))
        ),
        default_changes=False,
        pvalue_changes=False,
        author_contact=False,
        uncertainty=(
            "Independent training datasets; marginal exact and separate "
            "Bonferroni-64 intervals; descriptive pilot"
        ),
        scope="Reference support, not species discovery or guaranteed unfamiliar detection",
    )
    for name, value in (
        ("groups.json", groups),
        ("classwise.json", classwise),
        ("primary-rates.json", primary),
        ("paired-contrasts.json", paired),
        ("archival-groups.json", archive_groups),
        ("archival-classwise.json", archive_classes),
        ("oracle-groups.json", oracle_summary(fresh)),
        ("archival-oracle-groups.json", oracle_summary(archived)),
        ("observations.json", fresh),
        ("archival-observations.json", archived),
        (
            "study-design.json",
            dict(settings=frozen["settings"], counts=frozen["counts"], fixture=config.fixture),
        ),
        ("summary.json", summary),
    ):
        atomic_json(root / name, value)
    return summary
