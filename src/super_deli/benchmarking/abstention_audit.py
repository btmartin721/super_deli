"""Independent laws, raw joins, quadratic scores, ranks, decisions and denominators."""

import csv
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from super_deli.benchmarking.abstention_config import ARMS, POLICIES, cohort_names
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_power_data import reference as original_reference
from super_deli.benchmarking.residual_audit import audit_arm
from super_deli.benchmarking.residual_derivatives import evaluate
from super_deli.io.config import read_json
from super_deli.models.reference_abstention_config import DECISIONS


def reference(task, cohort):
    index = ("train", "calibration", "iid", "near", "far", "overlap", "shifted").index(cohort)
    n = (
        task["n"]
        if cohort == "train"
        else (80 if task["fixture"] else 160)
        if index == 1
        else (64 if task["fixture"] else 512)
        if index in (3, 4, 5)
        else (64 if task["fixture"] else 2048)
    )
    strength = 0.5 if task["case"] in ("linear_weak", "confounded_size") else 1.0
    seed = task["seed_path"] + [index]
    params = dict(
        case="four_complementary",
        n=n,
        signal_multiplier=strength,
        **{
            k: seed + [j]
            for j, k in enumerate(
                ("data_seed", "noise_seed", "permutation_seed", "ecology_order_seed")
            )
        },
    )
    d = {
        k: v
        for k, v in original_reference(params).items()
        if k not in ("morphology_truth", "ecology_truth")
    }
    if index in (3, 4, 5):
        target = (
            (3 * strength, 0)
            if index == 3
            else (6.0, 6.0)
            if index == 4
            else (3 * strength, 3 * strength)
        )
        old_y = d["truth"].copy()
        lm = np.array([[1.0, 0], [0, 1], [1, 1], [1, -1]])
        le = np.vstack((lm, [[1, 2], [2, 1], [1, -2], [-2, 1]]))
        for role, j, load in (("morphology", 0, lm), ("ecology", 1, le)):
            for i, label in enumerate(old_y):
                bit = label // 2 if j == 0 else label % 2
                d["latent_" + role][i, j] += target[j] - (
                    -3 * strength if bit == 0 else 3 * strength
                )
            load = load / np.sqrt(np.einsum("ij,ij->i", load, load))[:, None]
            sl = slice(0, 4) if j == 0 else slice(4, 12)
            d[role] = np.einsum("ij,kj->ik", d["latent_" + role], load) + d["noise"][:, sl]
        d["truth"] = np.full(n, 4, dtype=int)
    d["base_morphology"], d["base_ecology"] = d["morphology"].copy(), d["ecology"].copy()
    original_size = np.random.default_rng(seed + [4]).standard_normal(n) * 0.5
    size = np.array([original_size[i] for i in d["order"]])
    if task["case"] == "confounded_size" and cohort != "shifted":
        size += [0.8 if index in (3, 4, 5) or label // 2 else -0.8 for label in d["truth"]]
    d["size"] = size
    if task["case"] in ("combined", "confounded_size"):
        d["morphology"] = np.column_stack(
            [
                np.exp(0.35 * d["base_morphology"][:, j] + s * size)
                for j, s in enumerate((0.8, 1.0, 1.2, 1.4))
            ]
        )
    if task["case"] == "combined":
        d["ecology"] = 20 + 5 * np.tanh(0.5 * d["base_ecology"])
    if task["case"] == "confounded_size":
        d["ecology"] = np.column_stack(
            [d["base_ecology"][:, j] + (0.4 if j % 2 == 0 else -0.4) * size for j in range(8)]
        )
    return d


def raw_check(folder, task, cohort):
    actual, expected = arrays(folder / "generating-reference.npz"), reference(task, cohort)
    require(actual.keys() == expected.keys(), "All raw generative fields retained")
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    ds = read_json(folder / "raw/dataset.json")
    units = [v["specimen_id"] for v in ds["specimens"]]
    require(units == sorted(set(units)), "Unique canonical raw identities")
    for modality in ds["modalities"]:
        role = modality["name"]
        with (folder / "raw" / modality["path"]).open(newline="") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        obs = {row[0]: np.array(row[1:], dtype=float) for row in rows[1:]}
        require(len(obs) == len(rows) - 1 == len(units), "Unique measurement rows")
        matched = {v["unit_id"]: obs[v["observation_id"]] for v in modality["observations"]}
        require(set(matched) == set(units), "Complete explicit specimen/measurement joins")
        np.testing.assert_array_equal(np.vstack([matched[u] for u in units]), actual[role])
    if cohort in ("train", "calibration"):
        from super_deli.io.reference_labels import read_reference_labels

        mapping = read_reference_labels(folder / "references.csv")
        require(
            [mapping[u] for u in units] == [f"reference_{i}" for i in actual["truth"]],
            "ID-bound known labels only",
        )
    return actual, units


def project(x, a):
    idx = a["kept_indices"]
    return (x[:, idx] - a["means"][idx]) / a["scales"][idx] * a["column_weights"]


def prediction(x, a, residual):
    z = project(x, a)
    if residual:
        return evaluate(z, np.zeros(len(z), dtype=int), a)[1]
    logits = np.einsum("ij,kj->ik", z, a["coefficients"]) + a["intercept"]
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def quadratic(x, centers, precision, a):
    delta = project(x, a)[:, None, :] - centers
    return np.einsum("nci,ij,ncj->nc", delta, precision, delta)


def quantiles(scores, y):
    counts = np.bincount(y, minlength=4)
    require(np.all(counts >= 20), "Independent calibration class counts")
    ranks = np.array([(19 * (int(m) + 1) + 19) // 20 for m in counts])
    return np.array([sorted(scores[y == c, c])[ranks[c] - 1] for c in range(4)]), counts, ranks


def independent_metrics(y, p, compatible, accepted, status, known):
    n = len(y)
    chosen = np.array([max(range(4), key=lambda j: row[j]) for row in p])
    result = dict(
        n=n,
        accepted=sum(bool(v) for v in accepted),
        accepted_fraction=sum(bool(v) for v in accepted) / n,
        reasons={s: sum(v == s for v in status) for s in DECISIONS},
    )
    if not known:
        return result
    correct = [int(t) == int(c) for t, c in zip(y, chosen, strict=True)]
    total = sum(accepted)
    good = sum(c and a for c, a in zip(correct, accepted, strict=True))
    result.update(
        accepted_accuracy=good / total if total else None,
        correct_accepted_fraction=good / n,
        incorrect_accepted_fraction=(total - good) / n,
        known_withheld_fraction=(n - total) / n,
        raw_accuracy=sum(correct) / n,
        per_class=[],
    )
    for c in range(4):
        members = [i for i in range(n) if y[i] == c]
        selected = [i for i in members if accepted[i]]
        result["per_class"].append(
            dict(
                reference=c,
                n=len(members),
                accepted=len(selected),
                accepted_accuracy=sum(correct[i] for i in selected) / len(selected)
                if selected
                else None,
                true_reference_included=sum(bool(compatible[i, c]) for i in members) / len(members)
                if compatible is not None
                else None,
            )
        )
    accuracy = [v["accepted_accuracy"] for v in result["per_class"]]
    result["accepted_balanced_accuracy"] = (
        sum(accuracy) / 4 if all(v is not None for v in accuracy) else None
    )
    if compatible is not None:
        result.update(
            true_reference_included=sum(bool(compatible[i, int(y[i])]) for i in range(n)) / n,
            mean_compatible_references=sum(sum(row) for row in compatible) / n,
        )
    return result


def compare(actual, expected):
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), "Complete metric keys")
        for k in expected:
            compare(actual[k], expected[k])
    elif isinstance(expected, list):
        require(len(actual) == len(expected), "Metric list lengths")
        for a, b in zip(actual, expected, strict=True):
            compare(a, b)
    elif expected is None or isinstance(expected, (str, bool)):
        require(actual == expected, "Categorical/undefined metric parity")
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def decisions(p, score, q):
    compatible = score <= q
    accepted, status = [], []
    for i, row in enumerate(compatible):
        supported = [j for j, v in enumerate(row) if v]
        chosen = int(p[i].argmax())
        s = (
            DECISIONS[1]
            if not supported
            else DECISIONS[2]
            if len(supported) > 1
            else DECISIONS[0]
            if supported[0] == chosen
            else DECISIONS[3]
        )
        status.append(s)
        accepted.append(s == DECISIONS[0])
    return compatible, np.asarray(accepted), np.asarray(status)


def audit_fresh(folder, task, cache):
    cohorts, seen = {}, set()
    for name in cohort_names(task):
        d, ids = raw_check(folder if name == "train" else folder / name, task, name)
        require(not seen & set(ids), "No biological unit crosses cohort roles")
        seen.update(ids)
        cohorts[name] = d
    data, cal = cohorts.pop("train"), cohorts.pop("calibration")
    x, xc = [np.column_stack((d["morphology"], d["ecology"])) for d in (data, cal)]
    if (folder / "support-failure.json").exists():
        failure = read_json(folder / "support-failure.json")
        require(failure["status"] == "unavailable" and failure["error"], "Retained support failure")
        rows = read_json(folder / "fits.json")
        require(
            [r["arm"] for r in rows] == list(ARMS),
            "All classifiers accounted after support failure",
        )
        checked = []
        for row in rows:
            require(row["status"] == "failed" and row.get("error"), "Policy results withheld")
            trained = read_json(folder / row["arm"] / "execution.json")
            checked.append(audit_arm(folder / row["arm"], trained, task, data, {}, cache))
        return dict(
            neural_fits=sum(r["neural_fits"] for r in checked),
            verified_updates=sum(r["verified_updates"] for r in checked),
            external_cohorts=0,
            policy_evaluations=0,
            support_fits=1,
        )
    s = arrays(folder / "shared-support.npz")
    z = project(x, s)
    centers = np.stack([z[data["truth"] == c].mean(0) for c in range(4)])
    residual = z - centers[data["truth"]]
    cov = sum(np.outer(row, row) for row in residual) / len(residual)
    shrunk = 0.9 * cov + 0.1 * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    for k, v in (("centers", centers), ("unshrunk_covariance", cov), ("covariance", shrunk)):
        np.testing.assert_allclose(s[k], v, rtol=1e-10, atol=1e-10)
    require(np.linalg.eigvalsh(shrunk).min() > 0, "Positive-definite frozen shrinkage")
    np.testing.assert_allclose(s["cholesky"] @ s["cholesky"].T, shrunk, rtol=1e-10, atol=1e-10)
    precision = np.linalg.inv(shrunk)
    dc = quadratic(xc, centers, precision, s)
    np.testing.assert_allclose(s["calibration_distances"], dc, rtol=1e-10, atol=1e-10)
    q, counts, ranks = quantiles(dc, cal["truth"])
    np.testing.assert_allclose(s["distance_thresholds"], q, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(s["calibration_labels"], cal["truth"])
    np.testing.assert_array_equal(s["calibration_counts"], counts)
    np.testing.assert_array_equal(s["calibration_ranks"], ranks)
    rows = read_json(folder / "fits.json")
    require([r["arm"] for r in rows] == list(ARMS), "Four unchanged classifier fits once")
    totals = dict(
        neural_fits=0, verified_updates=0, external_cohorts=0, policy_evaluations=0, support_fits=1
    )
    for row in rows:
        arm = row["arm"]
        trained = read_json(folder / arm / "execution.json")
        checked = audit_arm(folder / arm, trained, task, data, {}, cache)
        totals["neural_fits"] += checked["neural_fits"]
        totals["verified_updates"] += checked["verified_updates"]
        if row["status"] != "available":
            require(row.get("error"), "Retained unavailable policy result")
            continue
        a = arrays(folder / arm / ("model.npz" if arm == "logistic" else "run/model.npz"))
        fitted, output = (
            arrays(folder / arm / "calibration.npz"),
            arrays(folder / arm / "policy-predictions.npz"),
        )
        for k in s:
            np.testing.assert_array_equal(s[k], fitted[k])
        cp = prediction(xc, a, arm != "logistic")
        np.testing.assert_allclose(fitted["calibration_probabilities"], cp, rtol=1e-10, atol=1e-10)
        pq, _, _ = quantiles(1 - cp, cal["truth"])
        np.testing.assert_allclose(fitted["probability_thresholds"], pq, rtol=1e-10, atol=1e-10)
        for name, d in cohorts.items():
            xx = np.column_stack((d["morphology"], d["ecology"]))
            p, ds = output[name + "_probabilities"], output[name + "_distances"]
            np.testing.assert_allclose(
                p, prediction(xx, a, arm != "logistic"), rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                ds, quadratic(xx, centers, precision, s), rtol=1e-10, atol=1e-10
            )
            for policy in POLICIES:
                compatible, accepted, status = (
                    (None, np.ones(len(p), dtype=bool), np.full(len(p), DECISIONS[0]))
                    if policy == "unscreened"
                    else decisions(
                        p, ds if policy == "distance" else 1 - p, fitted[policy + "_thresholds"]
                    )
                )
                prefix = name + "_" + policy + "_"
                np.testing.assert_array_equal(output[prefix + "accepted"], accepted)
                np.testing.assert_array_equal(
                    output[prefix + "accepted_indices"], np.where(accepted, p.argmax(1), -1)
                )
                np.testing.assert_array_equal(output[prefix + "decisions"], status)
                if compatible is not None:
                    np.testing.assert_array_equal(output[prefix + "compatible"], compatible)
                expected = independent_metrics(
                    d["truth"], p, compatible, accepted, status, name in ("iid", "shifted")
                )
                compare(row["metrics"][policy][name], expected)
                totals["policy_evaluations"] += 1
            totals["external_cohorts"] += 1
        for policy in POLICIES:
            m = row["metrics"][policy]
            expected = (
                m["iid"]["accepted_fraction"] >= 0.8
                and m["iid"]["accepted_accuracy"] is not None
                and m["iid"]["accepted_accuracy"] >= 0.9
                and m["near"]["accepted_fraction"] <= 0.1
                and m["far"]["accepted_fraction"] <= 0.1
            )
            require(row["joint_targets"][policy] == expected, "Prespecified joint operating target")
    return totals


def audit_unit(folder, job, source, *, cache):
    started = time.monotonic()
    require(read_json(folder / "task.json") == job["task"], "Frozen work-unit task")
    if job["kind"] == "fresh":
        result = audit_fresh(folder, job["task"], cache)
    else:
        entry = job["entry"]
        old = Path(source) / entry["path"]
        a = arrays(
            old / entry["arm"] / ("model.npz" if entry["arm"] == "logistic" else "run/model.npz")
        )
        output = arrays(folder / "replay-predictions.npz")
        original = arrays(old / entry["arm"] / "external-predictions.npz")
        for name in ("iid", "unfamiliar"):
            d = arrays(old / name / "generating-reference.npz")
            x = np.column_stack((d["morphology"], d["ecology"]))
            np.testing.assert_array_equal(
                output[name + "_probabilities"], original[name + "_probabilities"]
            )
            np.testing.assert_allclose(
                output[name + "_probabilities"],
                prediction(x, a, entry["arm"] != "logistic"),
                rtol=1e-10,
                atol=1e-10,
            )
        result = dict(
            neural_fits=0,
            verified_updates=0,
            external_cohorts=0,
            policy_evaluations=0,
            support_fits=0,
            replay_cohorts=2,
        )
    return [dict(key=job["task"]["key"], seconds=time.monotonic() - started, **result)]
