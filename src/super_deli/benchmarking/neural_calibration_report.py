"""Separate ordinary splits, global rejection and group recovery with pilot uncertainty."""

from collections import Counter
from zipfile import ZipFile

from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.gaussian_graph_report import difference_interval
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_calibration_audit import zip_arrays
from super_deli.benchmarking.neural_calibration_config import ARMS, tasks
from super_deli.benchmarking.neural_calibration_study import check_frozen
from super_deli.benchmarking.unknown_k_report import interval
from super_deli.benchmarking.view_signal_report import spread
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def rate(values, *, family_size):
    n = len(values)
    known = [v for v in values if v is not None]
    count = sum(known)
    missing = n - len(known)
    return dict(
        requested=n,
        available=len(known),
        unavailable=missing,
        count=count,
        all_request_known_rate=count / n,
        available_only_rate=count / len(known) if known else None,
        identification_bounds=[count / n, (count + missing) / n],
        marginal_95_interval=interval(count, n, 0.05),
        simultaneous_95_interval=interval(count, n, 0.05 / family_size),
        simultaneous_failure_interval=[
            interval(count, n, 0.05 / family_size)[0],
            interval(count + missing, n, 0.05 / family_size)[1],
        ],
    )


def report(root):
    frozen, c = check_frozen(root)
    audit = read_json(root / "main-audit.json")
    require(audit["passed"] and not (root / "STOP.json").exists(), "Audited unchanged study")
    require(
        audit["execution_sha256"] == fingerprint_file(root / "main-execution.json").sha256,
        "Matching main evidence",
    )
    rows = []
    for t in tasks(c, "main"):
        folder = root / "main" / t["key"]
        truth = arrays(folder / "generating-reference.npz")["truth"]
        for arm in ARMS:
            out = folder / arm / "calibration"
            summary = read_json(out / "summary.json") if (out / "summary.json").exists() else {}
            point = read_json(out / "observed.json") if (out / "observed.json").exists() else {}
            inference = summary.get("inference", {})
            done = point.get("status") == "complete"
            ari = None
            if done:
                with ZipFile(out / "pipeline_artifacts.zip") as z:
                    a = zip_arrays(z, "observed/run/point.npz")
                ari = float(adjusted_rand_score(truth, a["labels"]))
            recovered = (
                (point["selected_k"] == 4 and ari >= 0.8)
                if done and t["case"] == "four_complementary"
                else None
            )
            p = inference.get("p_value")
            rows.append(
                dict(
                    key=t["key"],
                    case=t["case"],
                    n=t["n"],
                    arm=arm,
                    selected_k=point.get("selected_k"),
                    point_available=done,
                    ari=ari,
                    ordinary_split=bool(point["selected_k"] > 1) if done else None,
                    global_rejection=inference.get("reject_joint_raw_gaussian"),
                    p_value=p,
                    recovered=recovered,
                    joint_rejection_recovery=(p <= 0.05 and recovered)
                    if p is not None and recovered is not None
                    else None,
                    failed_draws=inference.get("failed"),
                    tail_interval=inference.get("conditional_null_tail_interval_95"),
                    null_fit_available=(out / "null_model.npz").exists(),
                )
            )
    groups = []
    primary = []
    paired = []
    for case in sorted({r["case"] for r in rows}):
        for arm in ARMS:
            cell = [r for r in rows if r["case"] == case and r["arm"] == arm]
            group = dict(
                case=case,
                n=cell[0]["n"],
                arm=arm,
                requested=len(cell),
                selected_k_counts=dict(
                    Counter(str(r["selected_k"]) for r in cell if r["point_available"])
                ),
                ari=spread([r["ari"] for r in cell]),
                p_value=spread([r["p_value"] for r in cell]),
                failed_draws=spread([r["failed_draws"] for r in cell]),
            )
            for endpoint in ("global_rejection", "ordinary_split"):
                v = rate([r[endpoint] for r in cell], family_size=16)
                group[endpoint] = v
                primary.append(dict(case=case, n=cell[0]["n"], arm=arm, endpoint=endpoint, **v))
            if case == "four_complementary":
                for endpoint in ("recovered", "joint_rejection_recovery"):
                    group[endpoint] = rate([r[endpoint] for r in cell], family_size=1)
            groups.append(group)
        for endpoint in ("global_rejection", "ordinary_split"):
            a = {r["key"]: r[endpoint] for r in rows if r["case"] == case and r["arm"] == "neural"}
            b = {r["key"]: r[endpoint] for r in rows if r["case"] == case and r["arm"] == "pca4"}
            common = [k for k in a if a[k] is not None and b[k] is not None]
            positive = sum(a[k] and not b[k] for k in common)
            negative = sum(b[k] and not a[k] for k in common)
            n = len(a)
            missing = n - len(common)
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
                    simultaneous_95_interval=difference_interval(positive, negative, n, 0.05 / 8),
                    interpretation=(
                        "Secondary paired neural-minus-PCA event difference; "
                        "repeated heads are not independent datasets"
                    ),
                )
            )
    for name, value in (
        ("assessments", rows),
        ("groups", groups),
        ("primary-rates", primary),
        ("paired-comparisons", paired),
    ):
        write_json(value, root / (name + ".json"))
    summary = dict(
        schema_version=1,
        observations=len(tasks(c, "main")),
        fixture=c.fixture,
        counts=frozen["counts"],
        primary_rates=len(primary),
        secondary_paired_comparisons=len(paired),
        audits={
            phase: {
                k: v for k, v in read_json(root / (phase + "-audit.json")).items() if k != "records"
            }
            for phase in ("preflight", "main")
        },
        calibration_scope=(
            "Approximate fitted joint raw-Gaussian departure test with full neural/PCA refits; "
            "not species/selected-K significance"
        ),
        interpretation=(
            "Ten independent datasets per law: initial pilot with wide uncertainty, "
            "not general size/power validation"
        ),
        defaults_changed=False,
        historical_pvalues_changed=False,
        author_contact=False,
    )
    write_json(summary, root / "summary.json")
    return summary
