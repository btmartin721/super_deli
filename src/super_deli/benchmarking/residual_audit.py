"""Independent raw-law, identity, preprocessing, optimizer and external-score audits."""

import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.coordinate_audit import gaussian_reference
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_calibration_audit import audit_point
from super_deli.benchmarking.neural_power_data import reference as source_reference
from super_deli.benchmarking.residual_config import arms
from super_deli.benchmarking.residual_derivatives import evaluate
from super_deli.benchmarking.residual_run import head_files
from super_deli.io.config import read_json


def reference_data(task, cohort):
    index = ("train", "iid", "shifted", "unfamiliar").index(cohort)
    n = (
        task["n"]
        if index == 0
        else (32 if task["fixture"] else 512)
        if index == 3
        else (32 if task["fixture"] else 2048)
    )
    case = (
        "gaussian_independent"
        if task["case"] == "gaussian"
        else "continuous_gradient"
        if task["case"] == "gradient"
        else "four_complementary"
    )
    multiplier = 0.5 if task["case"] in ("linear_weak", "confounded_size") else 1.0
    seed = task["seed_path"] + [index]
    original = dict(
        case=case,
        n=n,
        signal_multiplier=multiplier,
        **{
            key: seed + [i]
            for i, key in enumerate(
                ("data_seed", "noise_seed", "permutation_seed", "ecology_order_seed")
            )
        },
    )
    d = {
        k: v
        for k, v in source_reference(original).items()
        if k not in ("morphology_truth", "ecology_truth")
    }
    if cohort == "unfamiliar":
        label = d["truth"].copy()
        for role, axis in (("morphology", 0), ("ecology", 1)):
            for i in range(n):
                bit = label[i] // 2 if axis == 0 else label[i] % 2
                d["latent_" + role][i, axis] += 6 - (-3 if bit == 0 else 3) * multiplier
        lm = np.array([[1.0, 0], [0, 1], [1, 1], [1, -1]])
        le = np.vstack((lm, [[1, 2], [2, 1], [1, -2], [-2, 1]]))
        for role, loading, sl in (("morphology", lm, slice(0, 4)), ("ecology", le, slice(4, 12))):
            loading /= np.sqrt(np.einsum("ij,ij->i", loading, loading))[:, None]
            d[role] = np.einsum("ij,kj->ik", d["latent_" + role], loading) + d["noise"][:, sl]
        d["truth"] = np.full(n, 4, dtype=int)
    d["base_morphology"], d["base_ecology"] = d["morphology"].copy(), d["ecology"].copy()
    size_original = np.random.default_rng(seed + [4]).standard_normal(n) * 0.5
    size = np.array([size_original[i] for i in d["order"]])
    if task["case"] == "confounded_size" and cohort != "shifted":
        size += np.array(
            [0.8 if cohort == "unfamiliar" or c // 2 != 0 else -0.8 for c in d["truth"]]
        )
    d["size"] = size
    if task["case"] in ("allometric", "combined", "confounded_size"):
        d["morphology"] = np.column_stack(
            [
                np.exp(0.35 * d["base_morphology"][:, j] + exponent * size)
                for j, exponent in enumerate((0.8, 1.0, 1.2, 1.4))
            ]
        )
    if task["case"] in ("saturating", "combined"):
        d["ecology"] = 20 + 5 * np.tanh(0.5 * d["base_ecology"])
    if task["case"] == "confounded_size":
        d["ecology"] = np.column_stack(
            [d["base_ecology"][:, j] + (0.4 if j % 2 == 0 else -0.4) * size for j in range(8)]
        )
    return d


def check_raw(directory, task, cohort):
    actual = arrays(directory / "generating-reference.npz")
    expected = reference_data(task, cohort)
    require(actual.keys() == expected.keys(), "Complete independently generated measurement fields")
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    ds = read_json(directory / "raw/dataset.json")
    units = [v["specimen_id"] for v in ds["specimens"]]
    require(len(set(units)) == len(units), "Unique raw cohort IDs")
    import csv

    for modality in ds["modalities"]:
        role = modality["name"]
        with (directory / "raw" / modality["path"]).open(newline="") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        values = {r[0]: [float(x) for x in r[1:]] for r in rows[1:]}
        observed = np.array([values[r["observation_id"]] for r in modality["observations"]])
        require(
            [r["unit_id"] for r in modality["observations"]] == units,
            "Exact source-to-specimen joins",
        )
        np.testing.assert_array_equal(observed, actual[role])
        require(
            rows[0][1:] == [f["name"] for f in modality["features"]],
            "Feature order bound to raw header",
        )
    return actual, units


def preprocessing(x, a):
    means, scales = np.mean(x, axis=0), np.std(x, axis=0)
    kept = np.flatnonzero(scales > 0)
    scales[scales == 0] = 1
    counts = np.array([sum(kept < 4), sum(kept >= 4)])
    weight = np.sqrt(len(kept) / (2 * counts))
    for key, expected in (
        ("means", means),
        ("scales", scales),
        ("kept_indices", kept),
        ("view_weights", weight),
        ("column_weights", weight[(kept >= 4).astype(int)]),
        ("retained_view_counts", counts),
    ):
        np.testing.assert_allclose(a[key], expected, rtol=1e-12, atol=1e-12)
    return (x[:, kept] - a["means"][kept]) / a["scales"][kept] * a["column_weights"]


def reference_scores(x, a, arm):
    from super_deli.benchmarking.neural_analysis import neural_forward

    scores = []
    for role, sl in (("morphology", slice(0, 4)), ("ecology", slice(4, 12))):
        b = {k.split("__", 1)[1]: v for k, v in a.items() if k.startswith(role + "__")}
        raw = x[:, sl]
        z = (raw[:, b["kept_indices"]] - b["means"][b["kept_indices"]]) / b["scales"][
            b["kept_indices"]
        ]
        value = (
            np.einsum("ij,kj->ik", z - b["pca_mean"], b["components"])
            if arm == "pca4"
            else neural_forward(z, b, "encoder")
        )
        scores.append((value - b["latent_means"]) / b["latent_scales"])
    return np.concatenate(scores, axis=1)


def check_metrics(row, cohort, values, probability, labels, *, supervised):
    metric = row["metrics"][cohort]
    if cohort == "unfamiliar":
        key = (
            "mean_maximum_reference_confidence"
            if supervised
            else "mean_maximum_component_membership"
        )
        np.testing.assert_allclose(
            [metric[key], metric["fraction_above_point_nine"]],
            [np.max(probability, axis=1).mean(), (np.max(probability, axis=1) > 0.9).mean()],
            rtol=1e-10,
            atol=1e-10,
        )
    elif supervised:
        y = values["truth"]
        confusion = np.array(
            [[np.count_nonzero((y == i) & (labels == j)) for j in range(4)] for i in range(4)]
        )
        target = (y[:, None] == np.arange(4)).astype(float)
        accuracy = sum(confusion[i, i] / sum(confusion[i]) for i in range(4)) / 4
        loss = sum(
            -np.log(max(probability[i, int(label)], np.finfo(float).tiny))
            for i, label in enumerate(y)
        ) / len(y)
        brier = np.sum(np.square(probability - target)) / len(y)
        np.testing.assert_array_equal(confusion, metric["confusion"])
        np.testing.assert_allclose(
            [accuracy, loss, brier],
            [metric["balanced_accuracy"], metric["log_loss"], metric["brier"]],
            rtol=1e-10,
            atol=1e-10,
        )
    elif "truth" in values:
        ari = float(adjusted_rand_score(values["truth"], labels))
        np.testing.assert_allclose(ari, metric["ari"], atol=1e-12)
        if "four_group_recovery" in metric:
            require(
                metric["four_group_recovery"] == (row["selected_k"] == 4 and ari >= 0.8),
                "Frozen recovery definition",
            )


def audit_arm(directory, row, task, data, cohorts, cache):
    counts = dict(
        neural_fits=row["neural_fits"],
        verified_updates=0,
        candidate_densities=0,
        external_cohorts=0,
    )
    if row["status"] != "available":
        require(row.get("error"), "Retained ordinary fit failure")
        return counts
    x = np.column_stack((data["morphology"], data["ecology"]))
    supervised = row["task"] == "reference_assignment"
    if supervised:
        residual = "residual_" in row["arm"]
        a = arrays(directory / ("run/model.npz" if residual else "model.npz"))
        z = preprocessing(x, a)
        if residual:
            checks = arrays(directory / "optimizer-checks.npz")
            require(
                checks["errors"].shape == (500, 4)
                and checks["step_sha256"].shape == (500,)
                and np.isfinite(checks["errors"]).all(),
                "All actual residual optimizer updates audited",
            )
            require(
                all(len(s) == 64 for s in checks["step_sha256"]),
                "Complete residual trajectory hashes",
            )
            for key, name in (
                ("reference_losses", "losses"),
                ("reference_regularization", "regularization"),
            ):
                np.testing.assert_allclose(checks[key], a[name], rtol=1e-10, atol=1e-10)
            counts["verified_updates"] = 500
            trained = read_json(directory / "run/summary.json")
            units = trained["unit_ids"]
        else:
            units = [
                u["specimen_id"]
                for u in read_json(directory.parent / "raw/dataset.json")["specimens"]
            ]
            np.testing.assert_array_equal(a["classes"], np.arange(4))
        from super_deli.io.reference_labels import read_reference_labels

        label_file = directory.parent / (
            "permuted-references.csv" if row["arm"].startswith("permuted_") else "references.csv"
        )
        label_map = read_reference_labels(label_file)
        expected_y = (
            np.random.default_rng([*task["seed_path"], 99]).permutation(data["truth"])
            if row["arm"].startswith("permuted_")
            else data["truth"]
        )
        require(
            [label_map[u] for u in units] == [f"reference_{v}" for v in expected_y],
            "Labels bind only the correct training identities",
        )
        if residual:
            loss, _ = evaluate(z, expected_y, a)
            np.testing.assert_allclose(loss, a["losses"][-1], rtol=1e-10, atol=1e-10)
    else:
        kind = "pca4" if row["arm"] == "pca4" else "neural"
        a = arrays(directory / "run/point.npz")
        summary = read_json(directory / "run/summary.json")
        config = row["configuration"]
        events = [] if kind == "pca4" else read_json(directory / "optimizer-checks.json")
        evidence = {} if kind == "pca4" else arrays(directory / "optimizer-checks.npz")
        checked, densities = audit_point(
            directory,
            kind,
            0,
            x,
            a,
            {**summary, "status": "complete"},
            events=events,
            evidence=evidence,
            used=set(),
            config=config,
            cache=cache,
            replay=True,
        )
        counts.update(verified_updates=checked, candidate_densities=densities)
        head = head_files(directory, kind)
        head_arrays, metadata = arrays(head / "fit.npz"), read_json(head / "execution.json")
        np.testing.assert_array_equal(
            a["scores"], np.column_stack([a[r + "__scores"] for r in ("morphology", "ecology")])
        )
    predicted = arrays(directory / "external-predictions.npz")
    for name, external in cohorts.items():
        raw = np.column_stack((external["morphology"], external["ecology"]))
        if supervised:
            kept = a["kept_indices"]
            zz = (raw[:, kept] - a["means"][kept]) / a["scales"][kept] * a["column_weights"]
            if residual:
                _, expected = evaluate(zz, np.zeros(len(zz), dtype=int), a)
            else:
                logits = np.einsum("ij,kj->ik", zz, a["coefficients"]) + a["intercept"]
                expected = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        else:
            score = reference_scores(raw, a, kind)
            expected = gaussian_reference(score, head_arrays, metadata["selected_k"], "tied")[0][
                :, metadata["component_order"]
            ]
        p, labels = predicted[name + "_probabilities"], predicted[name + "_labels"]
        np.testing.assert_allclose(p, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(labels, p.argmax(1))
        check_metrics(row, name, external, p, labels, supervised=supervised)
        counts["external_cohorts"] += 1
    return counts


def audit_unit(folder, job, source, *, cache):
    started = time.monotonic()
    task = job["task"]
    require(read_json(folder / "task.json") == task, "Unchanged unit task")
    if job["kind"] == "replay":
        data = arrays(folder / "generating-reference.npz")
        rows = read_json(folder / "fits.json")
        result = audit_arm(folder / "point", rows[0], task, data, {}, cache)
        entry = job["entry"]
        old = Path(source) / entry["path"]
        actual, expected = (
            arrays(folder / "point/run/point.npz"),
            arrays(old / (entry["arm"] + "-point.npz")),
        )
        require(actual.keys() == expected.keys(), "Every archived point array replayed")
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        result["replay_arrays"] = len(actual)
    else:
        data, training_ids = check_raw(folder, task, "train")
        names = (
            ["iid"]
            + (["shifted"] if task["case"] == "confounded_size" else [])
            + (["unfamiliar"] if task["labeled"] else [])
        )
        seen, cohorts = set(training_ids), {}
        for name in names:
            cohort, units = check_raw(folder / name, task, name)
            require(not seen & set(units), "No specimen crosses cohorts")
            seen.update(units)
            cohorts[name] = cohort
        rows = read_json(folder / "fits.json")
        require([r["arm"] for r in rows] == arms(task), "All fixed arms once, with no replacements")
        results = [audit_arm(folder / r["arm"], r, task, data, cohorts, cache) for r in rows]
        result = {
            k: sum(r.get(k, 0) for r in results)
            for k in ("neural_fits", "verified_updates", "candidate_densities", "external_cohorts")
        }
    return [dict(key=task["key"], seconds=time.monotonic() - started, **result)]
