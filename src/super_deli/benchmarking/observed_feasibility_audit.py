"""Independent measured laws, tree walks, frozen heads and operating-point counts."""

import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from super_deli.benchmarking.abstention_audit import (
    compare,
    decisions,
    independent_metrics,
    prediction,
    project,
    quadratic,
    reference,
)
from super_deli.benchmarking.abstention_config import ARMS
from super_deli.benchmarking.local_reference_audit import neighbor_reference
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.observed_feasibility_config import roles
from super_deli.io.config import read_json


def raw_roles(folder, task):
    data = {}
    seen = set()
    identities = read_json(folder / "identities.json")
    specs = roles(task)
    require(set(identities) == {r["name"] for r in specs}, "Complete identity role keys")
    for i, r in enumerate(specs):
        name = r["name"]
        d = arrays(folder / (name + ".npz"))
        original = dict(task, n=r["n"], seed_path=[*task["seed_path"], i])
        expected = reference(original, r["generator_cohort"])
        require(d.keys() == expected.keys(), "All generating fields retained")
        for k in d:
            np.testing.assert_allclose(d[k], expected[k], rtol=1e-12, atol=1e-12)
        ids = identities[name]
        tag = "observed750_" + "_".join(str(v) for v in task["seed_path"]) + "_" + name
        require(ids == [f"{tag}_{j:05d}" for j in range(r["n"])], "ID/task/role binding")
        require(
            len(ids) == len(set(ids)) == len(d["truth"]) and not seen.intersection(ids),
            "Independent diagnostic identities",
        )
        seen.update(ids)
        data[name] = d
    return data, seen


def audit_forest(folder, cohorts):
    forest, stored = arrays(folder / "forest.npz"), arrays(folder / "native-scores.npz")
    offsets = forest["offsets"]
    require(
        offsets.shape == (101,) and offsets[0] == 0 and np.all(np.diff(offsets) > 0),
        "One hundred numeric trees",
    )
    np.testing.assert_array_equal(forest["classes"], [0, 1])
    np.testing.assert_array_equal(forest["initial_log_odds"], [0.0])
    np.testing.assert_array_equal(forest["learning_rate"], [0.1])
    n = sum(len(cohorts["train_" + g]["truth"]) for g in ("known", "near", "far"))
    np.testing.assert_array_equal(forest["training_counts"], [n // 2, n // 2])
    for k in (
        "children_left",
        "children_right",
        "feature",
        "threshold",
        "value",
        "n_node_samples",
        "weighted_n_node_samples",
        "impurity",
    ):
        require(
            forest[k].shape == (int(offsets[-1]),) and np.isfinite(forest[k]).all(),
            "Complete finite numeric tree fields",
        )
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        require(
            0 < end - start <= 15 and forest["n_node_samples"][start] == n,
            "Depth-three full-sample tree envelope",
        )
        levels = {0: 0}
        for j in range(int(end - start)):
            left, right = (
                int(forest["children_left"][start + j]),
                int(forest["children_right"][start + j]),
            )
            require(j in levels and levels[j] <= 3, "Reachable bounded-depth tree")
            require(
                forest["weighted_n_node_samples"][start + j] == forest["n_node_samples"][start + j],
                "Unweighted diagnostic rows",
            )
            if left == -1:
                require(
                    right == -1 and forest["n_node_samples"][start + j] >= 10,
                    "Valid minimum-size leaf",
                )
            else:
                require(
                    j < left < end - start
                    and j < right < end - start
                    and left != right
                    and 0 <= forest["feature"][start + j] < 12,
                    "Measured-feature binary split",
                )
                require(left not in levels and right not in levels, "Unique tree parents")
                levels[left] = levels[right] = levels[j] + 1
                require(
                    forest["n_node_samples"][start + left] + forest["n_node_samples"][start + right]
                    == forest["n_node_samples"][start + j],
                    "Node row conservation",
                )
    elements = 0
    for name, d in cohorts.items():
        raw = np.concatenate((d["morphology"], d["ecology"]), axis=1)
        require(raw.shape[1] == 12 and np.isfinite(raw).all(), "Observed trait whitelist")
        with np.errstate(over="raise", invalid="raise"):
            x = np.array(raw, dtype=np.float32)
        require(np.isfinite(x).all(), "Finite native comparison inputs")
        stage = stored[name + "_stages"]
        require(stage.shape == (100, len(x)), "Every native stage/row retained")
        score = np.zeros(len(x))
        for i, (start, _end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            nodes = np.zeros(len(x), dtype=int)
            for depth in range(4):
                active = np.flatnonzero(forest["children_left"][start + nodes] != -1)
                if not len(active):
                    break
                require(depth < 3, "Prediction depth ceiling")
                indexes = start + nodes[active]
                features = forest["feature"][indexes]
                take_left = x[active, features] <= forest["threshold"][indexes]
                nodes[active] = np.where(
                    take_left, forest["children_left"][indexes], forest["children_right"][indexes]
                )
            require(np.all(forest["children_left"][start + nodes] == -1), "All rows reach a leaf")
            score += 0.1 * forest["value"][start + nodes]
            np.testing.assert_allclose(stage[i], score, rtol=1e-12, atol=1e-12)
        probability = np.empty(len(score))
        positive = score >= 0
        probability[positive] = 1 / (1 + np.exp(-score[positive]))
        e = np.exp(score[~positive])
        probability[~positive] = e / (1 + e)
        np.testing.assert_allclose(
            stored[name + "_known_scores"], probability, rtol=1e-12, atol=1e-12
        )
        elements += stage.size
    train = np.column_stack([stored["train_" + g + "_stages"] for g in ("known", "near", "far")])
    y = np.concatenate(
        [
            np.full(len(cohorts["train_" + g]["truth"]), int(g == "known"))
            for g in ("known", "near", "far")
        ]
    )
    loss = np.mean(np.logaddexp(0, train) - y[None, :] * train, axis=1)
    np.testing.assert_allclose(stored["native_training_log_loss"], loss, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(forest["native_train_score"], 2 * loss, rtol=1e-12, atol=1e-12)
    return elements


def count_reference(score, p, truth, known):
    predicted = np.argmax(p, axis=1)
    selected = [score >= j / 100 for j in range(101)]
    result = dict(n=len(score), accepted=[int(np.count_nonzero(a)) for a in selected])
    if known:
        correct = predicted == truth
        result.update(
            correct=[int(np.count_nonzero(a & correct)) for a in selected],
            raw_accuracy=float(np.mean(correct)),
            per_class=[],
        )
        for c in range(4):
            rows = truth == c
            result["per_class"].append(
                dict(
                    n=int(np.count_nonzero(rows)),
                    accepted=[int(np.count_nonzero(a & rows)) for a in selected],
                    correct=[int(np.count_nonzero(a & rows & correct)) for a in selected],
                )
            )
    return result


def selection_reference(cal):
    good = []
    a, b, c = cal["iid"], cal["near"], cal["far"]
    for i in range(101):
        count = a["accepted"][i]
        if (
            count
            and Fraction(count, a["n"]) >= Fraction(4, 5)
            and Fraction(a["correct"][i], count) >= Fraction(9, 10)
            and Fraction(b["accepted"][i], b["n"]) <= Fraction(1, 10)
            and Fraction(c["accepted"][i], c["n"]) <= Fraction(1, 10)
        ):
            good.append(i)
    if not good:
        return dict(status="available_no_operating_point", index=None, threshold=None)
    i = sorted(good, key=lambda j: (a["accepted"][j], j))[-1]
    return dict(status="selected", index=i, threshold=i / 100)


def point_reference(curve, i):
    n = curve["n"]
    a = curve["accepted"][i]
    v = dict(n=n, accepted=a, accepted_fraction=a / n)
    if "correct" in curve:
        k = curve["correct"][i]
        v.update(
            correct_accepted_fraction=k / n,
            incorrect_accepted_fraction=(a - k) / n,
            accepted_accuracy=k / a if a else None,
        )
    if "per_class" in curve:
        v.update(
            raw_accuracy=curve["raw_accuracy"],
            per_class=[
                dict(reference=c, **point_reference(x, i)) for c, x in enumerate(curve["per_class"])
            ],
        )
    return v


def target_reference(m):
    if m is None:
        return False
    a = m["iid"]
    k = a["accepted"]
    return bool(
        k
        and Fraction(k, a["n"]) >= Fraction(4, 5)
        and Fraction(round(a["correct_accepted_fraction"] * a["n"]), k) >= Fraction(9, 10)
        and all(Fraction(m[c]["accepted"], m[c]["n"]) <= Fraction(1, 10) for c in ("near", "far"))
    )


def audit_assessment(folder, base, cohorts):
    rows = read_json(folder / "assessment.json")
    require([r["arm"] for r in rows] == list(ARMS), "All four archived heads")
    saved = arrays(folder / "reference-predictions.npz")
    scores = arrays(folder / "native-scores.npz")
    metric, local = arrays(base / "shared-support.npz"), arrays(base / "local-support.npz")
    train = arrays(base / "generating-reference.npz")
    ztrain = project(np.column_stack((train["morphology"], train["ecology"])), metric)
    precision = np.linalg.inv(metric["covariance"])
    original_ids = set()
    for parent in [
        base,
        *[
            base / c
            for c in ("calibration", "iid", "near", "far", "overlap", "shifted")
            if (base / c).exists()
        ],
    ]:
        original_ids.update(
            v["specimen_id"] for v in read_json(parent / "raw/dataset.json")["specimens"]
        )
    require(
        not original_ids.intersection(
            u for values in read_json(folder / "identities.json").values() for u in values
        ),
        "No original specimen enters new roles",
    )
    geometry = {}
    for name, d in cohorts.items():
        if name.startswith("test_"):
            x = np.concatenate((d["morphology"], d["ecology"]), axis=1)
            q = quadratic(x, metric["centers"], precision, metric)
            near, _, _ = neighbor_reference(project(x, metric), ztrain, train["truth"], precision)
            geometry[name] = (q, near)
    evaluations = 0
    curve_evaluations = 0
    for row in rows:
        arm = row["arm"]
        a = arrays(base / arm / ("model.npz" if arm == "logistic" else "run/model.npz"))
        calibration = {}
        test = {}
        fitted = arrays(base / arm / "calibration.npz")
        expected = {k: {} for k in ("unscreened", "probability", "distance", "local")}
        for name, d in cohorts.items():
            if name.startswith("train_"):
                continue
            x = np.concatenate((d["morphology"], d["ecology"]), axis=1)
            p = prediction(x, a, arm != "logistic")
            np.testing.assert_allclose(
                saved[arm + "_" + name + "_probabilities"], p, rtol=1e-11, atol=1e-11
            )
            # Decisions/counts are reconstructed from the exact native frozen scores.
            p = saved[arm + "_" + name + "_probabilities"]
            known = name in ("calibration_known", "test_iid", "test_shifted")
            values = count_reference(scores[name + "_known_scores"], p, d["truth"], known)
            if name.startswith("calibration_"):
                calibration[
                    "iid" if name.endswith("known") else name.removeprefix("calibration_")
                ] = values
                continue
            key = name.removeprefix("test_")
            test[key] = values
            for policy in expected:
                if policy == "unscreened":
                    compatible = None
                    accepted = np.ones(len(p), dtype=bool)
                    status = np.full(len(p), "assigned")
                else:
                    q = local["thresholds"] if policy == "local" else fitted[policy + "_thresholds"]
                    value = (
                        1 - p if policy == "probability" else geometry[name][int(policy == "local")]
                    )
                    compatible, accepted, status = decisions(p, value, q)
                check = independent_metrics(d["truth"], p, compatible, accepted, status, known)
                compare(row["metrics"][policy][key], check)
                expected[policy][key] = check
                np.testing.assert_array_equal(
                    saved[arm + "_" + name + "_" + policy + "_accepted"], accepted
                )
                np.testing.assert_array_equal(
                    saved[arm + "_" + name + "_" + policy + "_decisions"], status
                )
                evaluations += 1
        compare(row["calibration_curves"], calibration)
        compare(row["evaluation_curves"], test)
        choice = selection_reference(calibration)
        compare(row["selection"], choice)
        fixed = (
            {c: point_reference(v, choice["index"]) for c, v in test.items()}
            if choice["status"] == "selected"
            else None
        )
        compare(row["metrics"]["diagnostic"], fixed)
        expected["diagnostic"] = fixed
        compare(row["joint_targets"], {p: target_reference(m) for p, m in expected.items()})
        curve_evaluations += 101 * (len(calibration) + len(test))
    return dict(
        policy_cohort_evaluations=evaluations
        + sum(len(r["metrics"]["diagnostic"] or {}) for r in rows),
        threshold_cohort_records=curve_evaluations,
    )


def audit_unit(folder, job, source, *, cache):
    started = time.monotonic()
    folder, source = Path(folder), Path(source)
    kind = job["kind"]
    require(read_json(folder / "task.json") == job["task"], "Frozen work-unit task")
    result = read_json(folder / "unit-result.json")
    require(result["status"] == "complete" and result["kind"] == kind, "Complete typed work unit")
    detail = {}
    if kind == "fresh":
        cohorts, _ = raw_roles(folder, job["task"])
        elements = audit_forest(folder, cohorts)
        detail = audit_assessment(folder, source / job["entry"]["path"], cohorts)
    elif kind == "detector_replay":
        original = folder.parent / job["original_key"]
        cohorts, _ = raw_roles(original, read_json(original / "task.json"))
        elements = audit_forest(folder, cohorts)
        for name in ("forest.npz", "native-scores.npz"):
            a, b = arrays(folder / name), arrays(original / name)
            require(a.keys() == b.keys(), "Full fit replay schema")
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
    else:
        elements = 0
        base = source / job["entry"]["path"]
        arm = job["entry"]["arm"]
        old = arrays(base / arm / "policy-predictions.npz")
        p = arrays(folder / "replay-predictions.npz")
        local = arrays(base / arm / "local-predictions.npz")
        replay = arrays(folder / "local-replay.npz")
        for key in p:
            np.testing.assert_array_equal(p[key], old[key])
        for key in replay:
            np.testing.assert_array_equal(replay[key], local[key])
        detail["replay_cohorts"] = 5
    expected_fit = int(kind != "frozen_replay")
    require(
        result["detector_fits"] == expected_fit and result["trees"] == 100 * expected_fit,
        "Known diagnostic fit/tree counts",
    )
    return [
        dict(
            key=job["task"]["key"],
            kind=kind,
            seconds=time.monotonic() - started,
            neural_fits=0,
            verified_updates=0,
            detector_fits=expected_fit,
            trees=100 * expected_fit,
            verified_stage_elements=elements,
            **detail,
        )
    ]
