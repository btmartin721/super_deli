"""Independent raw quadratics, neighbor ranks, decisions and true-law densities."""

import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from super_deli.benchmarking.abstention_audit import (
    audit_fresh,
    compare,
    decisions,
    independent_metrics,
    prediction,
    project,
    quantiles,
    raw_check,
)
from super_deli.benchmarking.abstention_config import ARMS, cohort_names
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.io.config import read_json


def neighbor_reference(query, train, y, precision):
    scores = []
    neighbors = []
    ties = []
    for c in range(4):
        indices = np.flatnonzero(y == c)
        delta = query[:, None, :] - train[indices][None, :, :]
        q = np.einsum("nki,ij,nkj->nk", delta, precision, delta)
        order = np.argsort(q, axis=1, kind="stable")[:, :5]
        fifth = np.take_along_axis(q, order[:, -1:], axis=1)[:, 0]
        scores.append(fifth)
        neighbors.append(indices[order])
        ties.append(np.sum(q == fifth[:, None], axis=1))
    return np.stack(scores, axis=1), np.stack(neighbors, axis=1), np.stack(ties, axis=1)


def audit_oracle(out, cohorts, task):
    stored = arrays(out / "oracle.npz")
    report = read_json(out / "oracle-result.json")
    a = 1.5 if task["case"] in ("linear_weak", "confounded_size") else 3.0
    means = []
    for first, second in ((-a, -a), (-a, a), (a, -a), (a, a)):
        size = (0.8 if first > 0 else -0.8) if task["case"] == "confounded_size" else 0.0
        means.append([first, 0, 0, second, size])
    size = 0.8 if task["case"] == "confounded_size" else 0.0
    other = [[a, 0, 0, 0, size], [6, 0, 0, 6, size]]
    covariance = np.diag([1, 1, 1, 1, 0.25])

    def independently(data):
        x = np.concatenate(
            (data["latent_morphology"], data["latent_ecology"], data["size"][:, None]), axis=1
        )
        known = np.column_stack(
            [multivariate_normal.logpdf(x, mean=m, cov=covariance) for m in means]
        )
        unseen = np.column_stack(
            [multivariate_normal.logpdf(x, mean=m, cov=covariance) for m in other]
        )
        ratio = logsumexp(unseen, axis=1) - logsumexp(known, axis=1) + np.log(2.0)
        p = np.exp(known - logsumexp(known, axis=1)[:, None])
        return ratio, p

    calibration, _ = independently(cohorts["calibration"])
    ranks = np.array([((len(calibration) + 1) * (20 - i) + 19) // 20 for i in range(1, 20)])
    thresholds = np.sort(calibration)[ranks - 1]
    np.testing.assert_array_equal(stored["ranks"], ranks)
    np.testing.assert_allclose(stored["thresholds"], thresholds, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(stored["calibration_score"], calibration, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(stored["nominal_withholding"], np.arange(1, 20) / 20)
    require(len(report["thresholds"]) == 19, "All nineteen diagnostic levels")
    evaluations = 0
    for name, data in cohorts.items():
        if name in ("train", "calibration"):
            continue
        score, p = independently(data)
        np.testing.assert_allclose(stored[name + "_score"], score, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(stored[name + "_probabilities"], p, rtol=1e-10, atol=1e-10)
        for i, q in enumerate(thresholds):
            accepted = score <= q
            status = np.where(accepted, "assigned", "withheld_outside_reference_support")
            expected = independent_metrics(
                data["truth"], p, None, accepted, status, name in ("iid", "shifted")
            )
            compare(report["thresholds"][i]["cohorts"][name], expected)
            compare(report["thresholds"][i]["nominal_withholding"], (i + 1) / 20)
            compare(report["thresholds"][i]["threshold"], q)
            evaluations += 1
    return evaluations


def audit_local(base, out, task, rows):
    cohorts = {
        name: raw_check(base if name == "train" else base / name, task, name)[0]
        for name in cohort_names(task)
    }
    oracle_count = audit_oracle(out, cohorts, task)
    if (out / "local-support-failure.json").exists():
        require(
            read_json(out / "local-support-failure.json")["error"], "Retained local-support failure"
        )
        require(
            all(r["local_status"] == "failed" for r in rows),
            "No local success after support failure",
        )
        return dict(local_policy_evaluations=0, oracle_evaluations=oracle_count)
    metric, local = arrays(base / "shared-support.npz"), arrays(out / "local-support.npz")
    train = cohorts["train"]
    cal = cohorts["calibration"]

    def x(d):
        return np.column_stack((d["morphology"], d["ecology"]))

    z = project(x(train), metric)
    zc = project(x(cal), metric)
    precision = np.linalg.inv(metric["covariance"])
    np.testing.assert_array_equal(local["training_labels"], train["truth"])
    np.testing.assert_array_equal(local["calibration_labels"], cal["truth"])
    np.testing.assert_allclose(
        local["whitened_training"] @ metric["cholesky"].T, z, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        local["whitened_calibration"] @ metric["cholesky"].T, zc, rtol=1e-10, atol=1e-10
    )
    score, _, _ = neighbor_reference(zc, z, train["truth"], precision)
    q, counts, ranks = quantiles(score, cal["truth"])
    for key, value in (("calibration_scores", score), ("thresholds", q)):
        np.testing.assert_allclose(local[key], value, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(local["calibration_counts"], counts)
    np.testing.assert_array_equal(local["calibration_ranks"], ranks)
    geometry = arrays(out / "local-geometry.npz")
    evaluations = 0
    for name, data in cohorts.items():
        if name in ("train", "calibration"):
            continue
        direct, indices, ties = neighbor_reference(
            project(x(data), metric), z, train["truth"], precision
        )
        np.testing.assert_allclose(geometry[name + "_scores"], direct, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(geometry[name + "_neighbor_indices"], indices)
        np.testing.assert_array_equal(geometry[name + "_cutoff_tie_counts"], ties)
        for row in rows:
            if row["status"] != "available" or row["local_status"] != "available":
                continue
            arm = row["arm"]
            original = arrays(base / arm / "policy-predictions.npz")
            p = original[name + "_probabilities"]
            compatible, accepted, status = decisions(p, direct, q)
            output = arrays(out / arm / "local-predictions.npz")
            for key, value in (
                ("compatible", compatible),
                ("accepted", accepted),
                ("decisions", status),
                ("accepted_indices", np.where(accepted, p.argmax(1), -1)),
            ):
                np.testing.assert_array_equal(output[name + "_" + key], value)
            compare(
                row["metrics"]["local"][name],
                independent_metrics(
                    data["truth"], p, compatible, accepted, status, name in ("iid", "shifted")
                ),
            )
            evaluations += 1
    for row in rows:
        if row["local_status"] != "available":
            continue
        m = row["metrics"]["local"]
        expected = (
            m["iid"]["accepted_fraction"] >= 0.8
            and m["iid"]["accepted_accuracy"] is not None
            and m["iid"]["accepted_accuracy"] >= 0.9
            and m["near"]["accepted_fraction"] <= 0.1
            and m["far"]["accepted_fraction"] <= 0.1
        )
        require(row["joint_targets"]["local"] == expected, "Unchanged absolute joint target")
    return dict(local_policy_evaluations=evaluations, oracle_evaluations=oracle_count)


def audit_unit(folder, job, source, *, cache):
    started = time.monotonic()
    task = job["task"]
    require(read_json(folder / "task.json") == task, "Frozen work-unit task")
    if job["kind"] == "replay":
        entry = job["entry"]
        base = Path(source) / entry["path"]
        arm = entry["arm"]
        original = arrays(base / arm / "policy-predictions.npz")
        saved = arrays(folder / "replay-predictions.npz")
        a = arrays(base / arm / ("model.npz" if arm == "logistic" else "run/model.npz"))
        for name in ("calibration", "iid", "near", "far", "overlap"):
            d = arrays(base / name / "generating-reference.npz")
            x = np.column_stack((d["morphology"], d["ecology"]))
            np.testing.assert_array_equal(
                saved[name + "_probabilities"], original[name + "_probabilities"]
            )
            np.testing.assert_allclose(
                saved[name + "_probabilities"],
                prediction(x, a, arm != "logistic"),
                rtol=1e-10,
                atol=1e-10,
            )
        kept_policies = arrays(folder / "replay-policies.npz")
        for key, value in kept_policies.items():
            np.testing.assert_array_equal(value, original[key])
        expected_keys = {
            key
            for key in original
            if any(
                key.startswith(c + "_" + policy + "_")
                for c in ("iid", "near", "far", "overlap")
                for policy in ("unscreened", "distance", "probability")
            )
        }
        require(set(kept_policies) == expected_keys, "All original policy states replayed")
        result = dict(
            neural_fits=0,
            verified_updates=0,
            external_cohorts=0,
            policy_evaluations=0,
            support_fits=0,
            oracle_evaluations=0,
            replay_cohorts=5,
        )
    else:
        archive = job["kind"] == "archive"
        base = Path(source) / job["entry"]["path"] if archive else folder
        original_task = job["entry"]["task"] if archive else task
        # This verifies retained historical traces, without replaying optimizer updates.
        checked = audit_fresh(base, original_task, cache)
        rows = read_json(folder / "fits.json")
        require([r["arm"] for r in rows] == list(ARMS), "Four fixed classifiers per observation")
        for row in rows:
            if archive:
                require(
                    row["neural_fits"] == 0 and row["archived"], "No archived retraining counted"
                )
        local = audit_local(base, folder, original_task, rows)
        result = {
            **checked,
            "neural_fits": 0 if archive else checked["neural_fits"],
            "verified_updates": 0 if archive else checked["verified_updates"],
            "historical_update_records_checked": checked["verified_updates"] if archive else 0,
            "policy_evaluations": checked["policy_evaluations"] + local["local_policy_evaluations"],
            "oracle_evaluations": local["oracle_evaluations"],
        }
    return [dict(key=task["key"], seconds=time.monotonic() - started, **result)]
