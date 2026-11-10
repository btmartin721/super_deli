"""Distinguish rejection, correct group recovery and their joint event."""

import math
from collections import Counter
from zipfile import ZipFile

from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.gaussian_graph_report import difference_interval
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_calibration_audit import zip_arrays
from super_deli.benchmarking.neural_calibration_report import rate
from super_deli.benchmarking.neural_false_split_report import assessment, paired_interval
from super_deli.benchmarking.neural_false_split_state import StudyLock, atomic_json
from super_deli.benchmarking.neural_power_config import ARMS, tasks
from super_deli.benchmarking.neural_power_study import engine
from super_deli.benchmarking.view_signal_report import spread
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json

PRIMARY = ("global_rejection", "recovered", "joint_rejection_recovery")


def event_rate(values, family):
    row = rate(values, family_size=family)
    p = row["all_request_known_rate"]
    row["monte_carlo_standard_error"] = (
        math.sqrt(p * (1 - p) / len(values)) if None not in values else None
    )
    return row


def paired(rows, endpoint, *, family=18):
    a, b = ({r["key"]: r[endpoint] for r in rows if r["arm"] == arm} for arm in ARMS)
    common = [key for key in a if a[key] is not None and b[key] is not None]
    pos = sum(a[k] and not b[k] for k in common)
    neg = sum(b[k] and not a[k] for k in common)
    n, missing = len(a), len(a) - len(common)
    return dict(
        requested=n,
        common_available=len(common),
        unknown_pairs=missing,
        neural_only=pos,
        pca_only=neg,
        difference_known_contribution=(pos - neg) / n,
        identification_bounds=[(pos - neg - missing) / n, (pos - neg + missing) / n],
        marginal_95_interval=difference_interval(pos, neg, n, 0.05),
        simultaneous_95_interval=difference_interval(pos, neg, n, 0.05 / family),
        simultaneous_failure_interval=paired_interval(pos, neg, missing, n, 0.05 / family),
    )


def assess(root, config):
    rows = []
    for task in tasks(config, "main"):
        for arm in ARMS:
            row = assessment(root, task, arm)
            row.update(
                cell=task["cell"],
                signal_multiplier=task["signal_multiplier"],
                ari=None,
                correct_k=None,
                underselected=None,
                selected_search_boundary=None,
                recovered=None,
                joint_rejection_recovery=None,
            )
            four = task["case"] in ("four_complementary", "unequal_complementary")
            if row["point_available"]:
                row["selected_search_boundary"] = row["selected_k"] == 4
                if four:
                    truth = arrays(root / "main" / task["key"] / "generating-reference.npz")[
                        "truth"
                    ]
                    with ZipFile(
                        root / "main" / task["key"] / arm / "calibration/pipeline_artifacts.zip"
                    ) as z:
                        labels = zip_arrays(z, "observed/run/point.npz")["labels"]
                    row["ari"] = float(adjusted_rand_score(truth, labels))
                    row["correct_k"] = row["selected_k"] == 4
                    row["underselected"] = row["selected_k"] < 4
                    row["recovered"] = row["correct_k"] and row["ari"] >= 0.8
                    if row["global_rejection"] is not None:
                        row["joint_rejection_recovery"] = (
                            row["global_rejection"] and row["recovered"]
                        )
            rows.append(row)
    return rows


def aggregate(rows):
    groups, primary, comparisons = [], [], []
    for cell in sorted({r["cell"] for r in rows}):
        values = [r for r in rows if r["cell"] == cell]
        meta = {k: values[0][k] for k in ("cell", "case", "n", "signal_multiplier")}
        signal = meta["case"] == "four_complementary"
        for arm in ARMS:
            subset = [r for r in values if r["arm"] == arm]
            group = dict(
                **meta,
                arm=arm,
                requested=len(subset),
                selected_k_counts=dict(
                    Counter(str(r["selected_k"]) for r in subset if r["point_available"])
                ),
                ari=spread([r["ari"] for r in subset]),
                p_value=spread([r["p_value"] for r in subset]),
                failed_draws=spread([r["failed_draws"] for r in subset]),
            )
            endpoints = ("ordinary_split", "global_rejection", "selected_search_boundary")
            if meta["case"] in ("four_complementary", "unequal_complementary"):
                endpoints += ("correct_k", "underselected", "recovered", "joint_rejection_recovery")
            for endpoint in endpoints:
                family = 36 if signal and endpoint in PRIMARY else 1
                result = event_rate([r[endpoint] for r in subset], family)
                group[endpoint] = result
                if family == 36:
                    primary.append(dict(**meta, arm=arm, endpoint=endpoint, **result))
            groups.append(group)
        if signal:
            for endpoint in PRIMARY:
                comparisons.append(dict(**meta, endpoint=endpoint, **paired(values, endpoint)))
    return groups, primary, comparisons


def report(root, source):
    study = engine(root, source)
    root = study.root
    frozen, config = study.check()
    with StudyLock(root):
        study.binding()
        require(not (root / "STOP.json").exists(), "Retained guard requires amendment")
        audit = read_json(root / "main-audit.json")
        require(
            audit["passed"]
            and audit["execution_sha256"] == fingerprint_file(root / "main-execution.json").sha256,
            "Audited full study",
        )
        require(
            read_json(root / "main-execution.json") == study.execution(config, "main"),
            "All requested units verified",
        )
        rows = assess(root, config)
        groups, primary, comparisons = aggregate(rows)
        if not config.fixture:
            require(
                len(rows) == 200 and len(primary) == 36 and len(comparisons) == 18,
                "Frozen report grid",
            )
        for name, value in (
            ("assessments", rows),
            ("groups", groups),
            ("primary-rates", primary),
            ("paired-comparisons", comparisons),
        ):
            atomic_json(root / (name + ".json"), value)
        value = dict(
            schema_version=1,
            fixture=config.fixture,
            observations=len(tasks(config, "main")),
            primary_rates=len(primary),
            paired_comparisons=len(comparisons),
            counts=frozen["counts"],
            audits={
                p: {
                    k: v for k, v in read_json(root / (p + "-audit.json")).items() if k != "records"
                }
                for p in ("preflight", "main")
            },
            defaults_changed=False,
            historical_pvalues_changed=False,
            author_contact=False,
            search_boundary="K<=4; overpartitioning beyond four cannot be assessed",
            interpretation=(
                "Small-cell descriptive operating range; no neural superiority or species guarantee"
            ),
            calibration_scope=(
                "Approximate fitted raw-Gaussian departure, not selected-K significance"
            ),
        )
        atomic_json(root / "summary.json", value)
        return value
