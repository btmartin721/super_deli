"""Descriptive paired archival summaries; no new significance calculation."""

from collections import Counter
from pathlib import Path

import numpy as np

from super_deli.benchmarking.neural_analysis import arrays, interval, require
from super_deli.benchmarking.neural_false_split_state import atomic_json, verify_seal
from super_deli.benchmarking.weak_signal_config import ARMS
from super_deli.benchmarking.weak_signal_reference import metrics
from super_deli.benchmarking.weak_signal_study import engine
from super_deli.io.config import read_json


def distribution(values):
    known = [v for v in values if v is not None]
    return dict(
        available=len(known),
        median=float(np.median(known)) if known else None,
        minimum=float(min(known)) if known else None,
        maximum=float(max(known)) if known else None,
    )


def rate(values):
    n = len(values)
    unknown = sum(v is None for v in values)
    success = sum(v is True for v in values)
    return dict(
        count=success,
        requested=n,
        unavailable=unknown,
        failure_bounds=[success / n, (success + unknown) / n],
        descriptive_95_interval=[interval(success, n)[0], interval(success + unknown, n)[1]],
    )


def observation(root, task):
    folder = root / "main" / task["key"]
    original = root / "inputs" / task["key"]
    result = read_json(folder / "analysis.json")
    truth = arrays(original / "reference.npz")["truth"]
    record = dict(
        key=task["key"],
        case=task["case"],
        n=task["n"],
        signal_multiplier=task["signal_multiplier"],
        cell=task["cell"],
        references=result["references"],
        geometry=result["geometry"],
        heads={},
    )
    for arm in ARMS:
        old = read_json(original / (arm + "-head.json"))
        saved = arrays(original / (arm + "-head.npz"))
        attempt = {r["k"]: r for r in old["candidate_attempts"]}
        winner = attempt[old["selected_k"]]
        # Four-dimensional tied GMM adds five parameters per component.
        penalty = 5 * (4 - old["selected_k"]) * np.log(task["n"])
        gain = 2 * (attempt[4]["log_likelihood"] - winner["log_likelihood"])
        np.testing.assert_allclose(gain - penalty, winner["bic"] - attempt[4]["bic"], atol=1e-8)
        row = dict(
            native=metrics(truth, saved["labels"]),
            stored_k4=metrics(truth, saved["k4_labels"]),
            stored_k4_eligible=next(
                r["status"] == "eligible" for r in old["candidates"] if r["k"] == 4
            ),
            k4_twice_likelihood_gain_over_selected=float(gain),
            k4_additional_bic_penalty=float(penalty),
            k4_bic_minus_selected=float(attempt[4]["bic"] - winner["bic"]),
        )
        if task["truth_fits"]:
            fitted = read_json(folder / (arm + "-truth/result.json"))
            row["truth_available"] = fitted["status"] == "complete"
            if row["truth_available"]:
                labels = arrays(folder / (arm + "-truth/fit.npz"))["k4_labels"]
                row["truth_start"] = metrics(truth, labels)
                row["truth_eligible"] = fitted["eligible"]
                row["truth_log_likelihood_gain_over_native_k4"] = (
                    fitted["log_likelihood"] - attempt[4]["log_likelihood"]
                )
                row["truth_bic_minus_selected"] = fitted["bic"] - winner["bic"]
        record["heads"][arm] = row
    return record


def aggregate(rows):
    first = rows[0]
    group = {k: first[k] for k in ("cell", "case", "n", "signal_multiplier")}
    group["requested"] = len(rows)
    group["references"] = {}
    for space in ("latent", "raw", "pca4"):
        group["references"][space] = {
            name: distribution([r["references"][space][name] for r in rows])
            for name in (
                "accuracy",
                "ari",
                "expected_fixed_map_accuracy",
                "expected_fixed_map_error",
                "fraction_max_probability_below_08",
            )
        }
        if first["case"] == "four_complementary":
            group["references"][space]["recovered"] = rate(
                [r["references"][space]["recovered"] for r in rows]
            )
    group["heads"] = {}
    for arm in ARMS:
        heads = [r["heads"][arm] for r in rows]
        row = dict(
            selected_k_counts=dict(Counter(str(h["native"]["selected_k"]) for h in heads)),
            ordinary_split=rate([h["native"]["selected_k"] > 1 for h in heads]),
            stored_k4_eligible=rate([h["stored_k4_eligible"] for h in heads]),
            native_ari=distribution([h["native"]["ari"] for h in heads]),
            stored_k4_ari=distribution([h["stored_k4"]["ari"] for h in heads]),
            k4_twice_likelihood_gain=distribution(
                [h["k4_twice_likelihood_gain_over_selected"] for h in heads]
            ),
            k4_bic_penalty=distribution([h["k4_additional_bic_penalty"] for h in heads]),
        )
        if first["case"] == "four_complementary":
            row.update(
                native_recovered=rate([h["native"]["recovered"] for h in heads]),
                stored_k4_recovered=rate([h["stored_k4"]["recovered"] for h in heads]),
                truth_recovered=rate(
                    [h["truth_start"]["recovered"] if h["truth_available"] else None for h in heads]
                ),
                truth_eligible_recovered=rate(
                    [
                        h["truth_start"]["recovered"] and h["truth_eligible"]
                        if h["truth_available"]
                        else None
                        for h in heads
                    ]
                ),
                truth_ari=distribution(
                    [h["truth_start"]["ari"] if h["truth_available"] else None for h in heads]
                ),
                truth_ll_gain=distribution(
                    [h.get("truth_log_likelihood_gain_over_native_k4") for h in heads]
                ),
            )
        group["heads"][arm] = row
    group["geometry"] = {
        space: dict(
            axis_r2_median=np.median(
                [r["geometry"][space]["linear_recovery"]["r2"] for r in rows], axis=0
            ).tolist(),
            scatter=distribution([r["geometry"][space]["between_scatter_fraction"] for r in rows]),
        )
        for space in ("latent", "raw", "pca4", "neural")
    }
    return group


def report(root, source):
    root = Path(root)
    study = engine(root, source)
    _, config = study.check()
    study.binding()
    require(not (root / "STOP.json").exists(), "Retained guard prohibits conclusions")
    audit = read_json(root / "main-audit.json")
    execution = read_json(root / "main-execution.json")
    require(
        audit["passed"] and execution["completed"] and execution["unknown_counter_units"] == 0,
        "Complete independently audited observations",
    )
    records = []
    for job in study.jobs(config, "main"):
        require(
            not verify_seal(root, "main", job)["interrupted"],
            "Complete diagnostic accounting required",
        )
        records.append(observation(root, job["task"]))
    groups = [
        aggregate([r for r in records if r["cell"] == cell])
        for cell in sorted({r["cell"] for r in records})
    ]
    summary = dict(
        schema_version=1,
        fixture=config.fixture,
        observations=len(records),
        groups=len(groups),
        reference_evaluations=sum(r["references"] for r in audit["records"]),
        archived_candidate_densities=sum(
            r["archived_candidate_densities"] for r in audit["records"]
        ),
        component_fits=execution["attempted"],
        failed_diagnostics=execution["failed_points"],
        neural_training=0,
        new_datasets=0,
        calibration_draws=0,
        defaults_changed=False,
        pvalues_changed=False,
        author_contact=False,
        interpretation="Diagnostic references and truth starts; no ARI/species guarantee",
    )
    atomic_json(root / "observations.json", records)
    atomic_json(root / "groups.json", groups)
    atomic_json(root / "summary.json", summary)
    return summary
