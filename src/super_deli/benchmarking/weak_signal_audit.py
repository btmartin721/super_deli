"""Independent densities, affine laws, native candidates and truth-start accounting."""

import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from super_deli.benchmarking.coordinate_audit import (
    audit_mixture,
    gaussian_reference,
    reference_scores,
)
from super_deli.benchmarking.matched_config import SETTINGS
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_power_data import reference
from super_deli.benchmarking.recovery_analysis import _ari
from super_deli.benchmarking.separate_audit import audit_pca
from super_deli.benchmarking.weak_signal_config import ARMS
from super_deli.io.config import read_json
from super_deli.models.separate_neural import unpack_branch


def independent_laws(task, point):
    k = 4 if task["case"] == "four_complementary" else 1
    means = np.zeros((k, 4))
    if k == 4:
        for j in range(k):
            means[j, 0] = 3 * task["signal_multiplier"] * (-1 if j // 2 == 0 else 1)
            means[j, 3] = 3 * task["signal_multiplier"] * (-1 if j % 2 == 0 else 1)
    loadings = np.zeros((12, 4))
    a = np.zeros((4, 12))
    b = np.zeros(4)
    for role, start, width, axis in (("morphology", 0, 4, 0), ("ecology", 4, 8, 2)):
        loading = np.asarray(SETTINGS["loading_" + role])
        for j, row in enumerate(loading):
            loadings[start + j, axis : axis + 2] = row / np.sqrt(sum(row**2))
        part = unpack_branch(point, role)
        for j in range(2):
            for column, original in enumerate(part["kept_indices"]):
                a[axis + j, start + original] = (
                    part["components"][j, column]
                    / part["scales"][original]
                    / part["latent_scales"][j]
                )
            b[axis + j] = (
                -a[axis + j, start : start + width] @ part["means"]
                - (part["components"][j] @ part["pca_mean"] + part["latent_means"][j])
                / part["latent_scales"][j]
            )
    raw = means @ loadings.T
    covariance = 0.0225 * np.eye(12) + sum(
        np.outer(loadings[:, j], loadings[:, j]) for j in range(4)
    )
    return (
        dict(
            latent=(means, np.eye(4)),
            raw=(raw, covariance),
            pca4=(raw @ a.T + b, a @ covariance @ a.T),
        ),
        a,
        b,
    )


def audit_reference(x, truth, saved, record, expected):
    mean, sigma = expected
    np.testing.assert_allclose(saved["means"], mean, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(saved["covariance"], sigma, atol=1e-11, rtol=1e-12)
    component = np.column_stack([multivariate_normal.logpdf(x, m, sigma) for m in mean])
    p = np.exp(component - logsumexp(component, axis=1)[:, None])
    np.testing.assert_allclose(saved["membership"], p, atol=1e-11, rtol=1e-11)
    np.testing.assert_array_equal(saved["labels"], p.argmax(1))
    labels = saved["labels"]
    np.testing.assert_allclose(record["ari"], _ari(truth, labels), atol=1e-12)
    require(record["accuracy"] == float(np.mean(labels == truth)), "Reference accuracy")
    require(
        record["recovered"] == bool(len(set(labels)) == 4 and _ari(truth, labels) >= 0.8),
        "Reference recovery",
    )
    if len(mean) == 1:
        expected_accuracy = 1.0
    else:
        split = 4 if x.shape[1] == 12 else 2
        require(np.max(abs(sigma[:split, split:])) < 1e-12, "Independent view blocks")
        correct = []
        for first, second, sl in ((0, 2, slice(0, split)), (0, 1, slice(split, x.shape[1]))):
            difference = mean[second, sl] - mean[first, sl]
            length = np.sqrt(difference @ np.linalg.inv(sigma[sl, sl]) @ difference)
            correct.append(1 - norm.cdf(-length / 2))
        expected_accuracy = float(np.prod(correct))
    np.testing.assert_allclose(record["expected_fixed_map_accuracy"], expected_accuracy, atol=1e-12)
    np.testing.assert_allclose(
        record["expected_fixed_map_error"], 1 - expected_accuracy, atol=1e-12
    )
    np.testing.assert_allclose(record["mean_max_probability"], p.max(1).mean(), atol=1e-12)
    require(
        record["fraction_max_probability_below_08"] == float(np.mean(p.max(1) < 0.8)),
        "Reference ambiguity denominator",
    )


def audit_truth(folder, x, truth):
    row = read_json(folder / "result.json")
    initial = arrays(folder / "initialization.npz")
    means = np.array([np.mean(x[truth == k], axis=0) for k in range(4)])
    covariance = sum(
        np.outer(v - means[g], v - means[g]) for v, g in zip(x, truth, strict=True)
    ) / len(x) + 1e-6 * np.eye(x.shape[1])
    np.testing.assert_allclose(initial["means"], means, atol=1e-12)
    np.testing.assert_allclose(initial["covariance"], covariance, atol=1e-12)
    np.testing.assert_array_equal(initial["weights"], np.bincount(truth, minlength=4) / len(truth))
    require(row["attempted"] == 1 and row["n_init"] == 1, "One isolated truth start")
    saved = arrays(folder / "fit.npz")
    if row["status"] == "complete":
        p, ll, bic, _ = gaussian_reference(x, saved, 4, "tied")
        np.testing.assert_allclose(p, saved["k4_membership"], atol=1e-8)
        np.testing.assert_allclose(
            [ll, bic], [row["log_likelihood"], row["bic"]], atol=1e-5, rtol=0
        )
        np.testing.assert_array_equal(saved["k4_labels"], p.argmax(1))
        sizes = np.bincount(saved["k4_labels"], minlength=4)
        require(row["converged"] and 1 <= row["iterations"] <= 500, "Converged diagnostic fit")
        require(
            row["hard_cluster_sizes"] == sizes.tolist()
            and row["eligible"] == bool(sizes.min() >= 5),
            "Descriptive eligibility",
        )
    else:
        require(row["status"] == "failed" and bool(row.get("error")), "Retained failed diagnostic")
    return row


def audit_unit(folder, job, source, *, cache):
    started = time.monotonic()
    task = job["task"]
    root = folder.parents[1]
    original = root / "inputs" / task["key"]
    data = arrays(original / "reference.npz")
    expected = reference(task)
    require(data.keys() == expected.keys(), "Complete generating reference")
    for name in data:
        np.testing.assert_allclose(data[name], expected[name], atol=1e-12, rtol=1e-12)
    x = np.column_stack((data["morphology"], data["ecology"]))
    points = {a: arrays(original / (a + "-point.npz")) for a in ARMS}
    laws, affine, offset = independent_laws(task, points["pca4"])
    np.testing.assert_allclose(x @ affine.T + offset, points["pca4"]["scores"], atol=1e-10)
    for arm in ARMS:
        np.testing.assert_allclose(
            reference_scores(data, points[arm], "pca4" if arm == "pca4" else "separate_0"),
            points[arm]["scores"],
            atol=1e-10,
        )
    for role, sl in (("morphology", slice(0, 4)), ("ecology", slice(4, 12))):
        audit_pca(x[:, sl], unpack_branch(points["pca4"], role))
    result = read_json(folder / "analysis.json")
    require(result["task"] == task, "Frozen task identity")
    coordinates = dict(
        latent=np.column_stack((data["latent_morphology"], data["latent_ecology"])),
        raw=x,
        pca4=points["pca4"]["scores"],
    )
    for name, values in coordinates.items():
        audit_reference(
            values,
            data["truth"],
            arrays(folder / (name + "-reference.npz")),
            result["references"][name],
            laws[name],
        )
    for space, values in {**coordinates, "neural": points["neural"]["scores"]}.items():
        stored = result["geometry"][space]
        target = coordinates["latent"] - coordinates["latent"].mean(0)
        centered = values - values.mean(0)
        residual = target - centered @ np.linalg.pinv(centered) @ target
        r2 = 1 - np.sum(residual**2, axis=0) / np.sum(target**2, axis=0)
        np.testing.assert_allclose(stored["linear_recovery"]["r2"], r2, atol=1e-10)
        between = sum(
            np.sum(data["truth"] == g) * np.sum(centered[data["truth"] == g].mean(0) ** 2)
            for g in np.unique(data["truth"])
        )
        np.testing.assert_allclose(
            stored["between_scatter_fraction"], between / np.sum(centered**2), atol=1e-12
        )
    checked = attempts = failures = 0
    for arm in ARMS:
        old = read_json(original / (arm + "-head.json"))
        saved = arrays(original / (arm + "-head.npz"))
        count, _ = audit_mixture(points[arm]["scores"], {**old, "clusterer": "gmm_shared"}, saved)
        checked += count
        if task["diagnostic_replay"]:
            replay = arrays(folder / (arm + "-replay.npz"))
            require(replay.keys() == saved.keys(), "All replay arrays")
            for name in saved:
                np.testing.assert_array_equal(replay[name], saved[name])
            require(
                read_json(folder / (arm + "-replay.json"))
                == {k: v for k, v in old.items() if k != "seconds"},
                "Exact replay head",
            )
            attempts += len(old["candidate_attempts"])
        if task["truth_fits"]:
            row = audit_truth(folder / (arm + "-truth"), points[arm]["scores"], data["truth"])
            require(row == result["fits"][arm]["truth_fit"], "Bound truth-start endpoint")
            attempts += 1
            failures += row["status"] != "complete"
    return [
        dict(
            key=task["key"],
            neural_fits=0,
            verified_updates=0,
            references=3,
            archived_candidate_densities=checked,
            component_fits=attempts,
            failed_points=failures,
            seconds=time.monotonic() - started,
        )
    ]
