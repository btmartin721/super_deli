"""Independent source-law reconstruction and all-candidate density arithmetic."""

import numpy as np
from scipy.special import logsumexp

from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.io.config import read_json


def reference(task):
    n, case = task["n"], task["case"]
    rng = np.random.default_rng(task["data_seed"])
    truth = None
    if case == "gaussian":
        z = rng.normal(size=(n, 2))
        truth = np.zeros(n, dtype=int)
    elif case == "gaussian_8d":
        z = rng.normal(size=(n, 8))
        truth = np.zeros(n, dtype=int)
    elif case == "heavy_tail":
        z = rng.standard_t(df=3, size=(n, 2)) / np.sqrt(3)
    elif case == "cline":
        a = rng.uniform(-3, 3, n)
        z = np.array([a, 0.5 * a + rng.normal(0, 0.3, n)]).T
    elif case == "moons":
        a, b = rng.uniform(0, np.pi, (2, n // 2))
        z = np.vstack(
            [np.array([np.cos(a), np.sin(a)]).T, np.array([1 - np.cos(b), 0.5 - np.sin(b)]).T]
        )
        z += rng.normal(0, 0.08, z.shape)
        truth = np.repeat([0, 1], n // 2)
    elif case == "eight_spherical":
        centers = [8 * np.array([np.cos(j * np.pi / 4), np.sin(j * np.pi / 4)]) for j in range(8)]
        z = np.vstack([rng.normal(c, 0.6, size=(n // 8, 2)) for c in centers])
        truth = np.repeat(np.arange(8), n // 8)
    else:
        centers = [[-3, -3], [-3, 3], [3, -3], [3, 3]]
        count = [n // 4] * 4
        cov = [np.eye(2)] * 4
        if case == "four_unequal":
            count = [int(0.4 * n), int(0.3 * n), int(0.2 * n)]
            count.append(n - sum(count))
            cov = [0.25 * np.eye(2), np.eye(2)]
            for theta, diagonal in ((35, [2.25, 0.36]), (70, [1.44, 0.16])):
                angle = theta * np.pi / 180
                rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                cov.append(rot @ np.diag(diagonal) @ rot.T)
        require(case in {"four_spherical", "four_unequal"}, "Known reference law")
        z = np.vstack(
            [
                rng.multivariate_normal(c, v, size=q)
                for c, v, q in zip(centers, cov, count, strict=True)
            ]
        )
        truth = np.concatenate([np.full(q, k) for k, q in enumerate(count)])
    if case in {"gaussian", "four_spherical", "four_unequal", "moons", "cline"}:
        original = rng.permutation(n)
        z = z[original]
        if truth is not None:
            truth = truth[original]
    order = np.random.default_rng(task["permutation_seed"]).permutation(n)
    z = z[order]
    truth = truth[order] if truth is not None else None
    if case == "gaussian_8d":
        clean = z.copy()
    elif case == "moons":
        a, b = z.T
        clean = np.array([a, b, a * b, a * a, b * b, np.sin(a), np.cos(b), np.sin(a + b)]).T
    else:
        mapping = np.array(
            [[1, 0], [0, 1], [1, 1], [1, -1], [0.5, 1], [-1, 0.5], [2, 0.5], [0.5, -2]]
        )
        clean = np.einsum("ij,kj->ik", z, mapping)
    noise = np.random.default_rng(task["noise_seed"]).normal(0, 0.15, clean.shape)
    result = {
        "latent": z,
        "clean": clean,
        "noise": noise,
        "noisy": clean + noise,
        "noise_mask": np.full(clean.shape, True, dtype=bool),
        "order": order,
    }
    if truth is not None:
        result["truth"] = truth
    return result


def audit_candidates(folder, n):
    point_seed = int(np.random.SeedSequence(71323).spawn(1)[0].generate_state(1)[0])
    rows = read_json(folder / "candidate-trace.json")
    a = arrays(folder / "candidate-trace.npz")
    candidates = (
        read_json(folder / "candidates.json") if (folder / "candidates.json").exists() else []
    )
    require(
        [r["k"] for r in rows] == list(range(1, len(rows) + 1)), "No replaced/extra candidate fits"
    )
    for row in rows:
        k = row["k"]
        require(
            row["n_init"] == 10
            and row["random_state"]
            == int(np.random.SeedSequence([point_seed, k]).generate_state(1)[0]),
            "GMM settings",
        )
        if row["status"] == "failed":
            require(bool(row["error"]), "Preserved candidate failure")
            continue
        scores = a["scores"]
        joint = []
        weights = a[f"k{k}_weights"]
        means = a[f"k{k}_means"]
        covs = a[f"k{k}_covariances"]
        require(np.all(weights > 0) and abs(weights.sum() - 1) < 1e-10, "Mixture weights")
        for w, mu, cov in zip(weights, means, covs, strict=True):
            require(np.linalg.eigvalsh(cov).min() > 0, "Candidate covariance")
            delta = scores - mu
            joint.append(
                np.log(w)
                - 0.5
                * (
                    2 * np.log(2 * np.pi)
                    + np.linalg.slogdet(cov)[1]
                    + np.einsum("ij,jk,ik->i", delta, np.linalg.inv(cov), delta)
                )
            )
        joint = np.column_stack(joint)
        density = logsumexp(joint, axis=1)
        membership = np.exp(joint - density[:, None])
        labels = membership.argmax(1)
        np.testing.assert_allclose(membership, a[f"k{k}_membership"], atol=1e-10)
        np.testing.assert_array_equal(labels, a[f"k{k}_labels"])
        bic = (6 * k - 1) * np.log(n) - 2 * density.sum()
        np.testing.assert_allclose(bic, row["bic"], atol=1e-6)
        if candidates:
            c = next(c for c in candidates if c["k"] == k)
            np.testing.assert_allclose(bic, c["bic"], atol=1e-6)
            sizes = np.bincount(labels, minlength=k)
            require(
                c["status"] == ("eligible" if sizes.min() >= 5 else "small_component"),
                "Size admissibility",
            )
            require(
                c["hard_cluster_sizes"] == sizes.tolist() and row["converged"],
                "Candidate convergence/sizes",
            )
    if candidates:
        require(len(rows) == 4, "Complete search retained")
