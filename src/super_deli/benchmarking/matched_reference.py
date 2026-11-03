"""Independent raw-law, balancing, optimizer and projection reference equations."""

import numpy as np

from super_deli.benchmarking.denoising_numerics import gradients, reconstruction
from super_deli.benchmarking.neural_analysis import neural_forward, require


def reference_data(task):
    n = task["n"]
    case = task["case"]
    rng = np.random.default_rng(task["data_seed"])
    m_noise = rng.standard_normal((n, 2))
    e_noise = rng.standard_normal((n, 2))
    corners = np.array([[-3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [3.0, 3.0]])
    truth = mt = et = None
    if case == "gaussian_independent":
        m, e = m_noise, e_noise
        truth = np.full(n, 0, dtype=int)
    elif case in ("gaussian_correlated", "heavy_tail"):
        latent = (
            rng.standard_normal((n, 2))
            if case == "gaussian_correlated"
            else rng.standard_t(df=3, size=(n, 2)) / 3**0.5
        )
        m, e = latent + 0.5 * m_noise, latent + 0.5 * e_noise
        if case == "gaussian_correlated":
            truth = np.full(n, 0, dtype=int)
    elif case == "discordant":
        mt = np.repeat([0, 1, 2, 3], n // 4)
        et = np.tile(np.repeat([0, 1, 2, 3], n // 16), 4)
        m, e = corners[mt] + m_noise, corners[et] + e_noise
    elif case == "moons":
        t = rng.uniform(0, np.pi, size=(2, n // 2))
        latent = np.vstack(
            (
                np.array([np.cos(t[0]), np.sin(t[0])]).T,
                np.array([1 - np.cos(t[1]), 0.5 - np.sin(t[1])]).T,
            )
        )
        latent += rng.standard_normal(latent.shape) * 0.08
        m, e = latent + 0.1 * m_noise, latent + 0.1 * e_noise
        truth = np.repeat([0, 1], n // 2)
    else:
        truth = np.repeat([0, 1, 2, 3], n // 4)
        if case == "four_complementary":
            m = m_noise.copy()
            e = e_noise.copy()
            m[:, 0] += corners[truth, 0]
            e[:, 1] += corners[truth, 1]
            mt = truth // 2
            et = truth % 2
        else:
            require(case in {"four_redundant", "morphology_only"}, "Known source law")
            m = corners[truth] + m_noise
            e = corners[truth] + e_noise if case == "four_redundant" else e_noise
    lm = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=float)
    le = np.vstack((lm, [[1, 2], [2, 1], [1, -2], [-2, 1]]))
    lm /= np.sqrt((lm * lm).sum(1))[:, None]
    le /= np.sqrt((le * le).sum(1))[:, None]
    noise = np.random.default_rng(task["noise_seed"]).standard_normal((n, 12)) * 0.15
    order = np.random.default_rng(task["permutation_seed"]).permutation(n)
    result = {
        "morphology": (np.einsum("ij,kj->ik", m, lm) + noise[:, :4])[order],
        "ecology": (np.einsum("ij,kj->ik", e, le) + noise[:, 4:])[order],
        "latent_morphology": m[order],
        "latent_ecology": e[order],
        "noise": noise[order],
        "order": order,
        "ecology_order": np.random.default_rng(task["ecology_order_seed"]).permutation(n),
    }
    for key, value in (("truth", truth), ("morphology_truth", mt), ("ecology_truth", et)):
        if value is not None:
            result[key] = value[order]
    return result


def balanced_inputs(x, a, paired):
    kept = a["kept_indices"]
    np.testing.assert_allclose(a["means"], np.mean(x, axis=0), atol=1e-12)
    scales = x.std(0)
    scales[scales == 0] = 1
    np.testing.assert_allclose(a["scales"], scales, atol=1e-12)
    np.testing.assert_array_equal(kept, np.flatnonzero(x.std(0) > 0))
    # Check estimated parameters independently above, then replay the actual
    # fitted input. Re-reducing an advanced-index copy changes rounding order.
    z = (x[:, kept] - a["means"][kept]) / a["scales"][kept]
    if paired:
        counts = np.array([sum(kept < 4), sum(kept >= 4)])
        require(np.all(counts >= 3), "Both fitted views retained")
        expected = (len(kept) / (2 * counts)) ** 0.5
        np.testing.assert_array_equal(a["retained_view_counts"], counts)
        np.testing.assert_array_equal(a["selected_view_counts"], [4, 8])
        np.testing.assert_allclose(a["view_weights"], expected, atol=1e-15)
        weights = expected[(kept >= 4).astype(int)]
        np.testing.assert_allclose(a["column_weights"], weights, atol=1e-15)
        z *= a["column_weights"]
    return z


def audit_optimizer(x, a, initial):
    """Replay every Adam update using NumPy derivatives, never torch/autograd."""
    weights = {k: v.copy() for k, v in initial.items()}
    require(
        len(weights) == 12 and a["losses"].shape == (501,), "Complete ordinary training evidence"
    )
    first = {k: np.zeros_like(v) for k, v in weights.items()}
    second = {k: np.zeros_like(v) for k, v in weights.items()}
    losses = [np.mean((reconstruction(x, weights) - x) ** 2)]
    for update in range(1, 501):
        g = gradients(x, x, weights)
        for key in weights:
            gradient = g[key] + 0.0001 * weights[key]
            first[key] = 0.9 * first[key] + 0.1 * gradient
            second[key] = 0.999 * second[key] + 0.001 * gradient**2
            weights[key] -= (
                0.001
                * (first[key] / (1 - 0.9**update))
                / (np.sqrt(second[key] / (1 - 0.999**update)) + 1e-8)
            )
        losses.append(np.mean((reconstruction(x, weights) - x) ** 2))
    # Independent tanh/matmul/Adam arithmetic accumulates roundoff over 500
    # updates. Retain measured discrepancies; do not weaken weight guards.
    np.testing.assert_allclose(a["losses"], losses, rtol=1e-8, atol=1e-8)
    for key in weights:
        np.testing.assert_allclose(a[key], weights[key], rtol=1e-7, atol=1e-8)
    audit_latent(x, a)
    return {
        "updates": len(losses) - 1,
        "max_loss_absolute_difference": float(np.max(np.abs(a["losses"] - losses))),
        "max_parameter_absolute_difference": max(
            float(np.max(np.abs(a[key] - weights[key]))) for key in weights
        ),
        "loss_relative_tolerance": 1e-8,
        "loss_absolute_tolerance": 1e-8,
        "parameter_relative_tolerance": 1e-7,
        "parameter_absolute_tolerance": 1e-8,
    }


def audit_latent(x, a):
    raw = neural_forward(x, a, "encoder")
    np.testing.assert_allclose(a["latent_means"], raw.mean(0), atol=1e-11)
    np.testing.assert_allclose(a["latent_scales"], raw.std(0), atol=1e-11)
    scores = (raw - raw.mean(0)) / raw.std(0)
    np.testing.assert_allclose(a["scores"], scores, atol=1e-10)
    require(np.linalg.matrix_rank(scores) == 2, "Resolved latent space")


def audit_pca(x, a, scores):
    mean = a.get("feature_means", a.get("means"))
    scale = a.get("feature_scales", a.get("scales"))
    center = a.get("pca_mean", a.get("mean"))
    kept = a["kept_indices"]
    np.testing.assert_allclose(mean, x.mean(0), atol=1e-12)
    np.testing.assert_allclose(scale, x.std(0), atol=1e-12)
    z = (x[:, kept] - mean[kept]) / scale[kept]
    components = a["components"]
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-11)
    np.testing.assert_allclose(center, z.mean(0), atol=1e-12)
    np.testing.assert_allclose(scores, (z - center) @ components.T, atol=1e-10)
    np.testing.assert_allclose(
        a["explained_variance"], np.linalg.eigvalsh(np.cov(z, rowvar=False))[::-1][:2], atol=1e-10
    )


def audit_density(a):
    labels = a["labels"]
    strength = a["strength"]
    require(
        labels.dtype.kind in "iu" and labels.shape == (len(a["scores"]),),
        "Density label dimensions",
    )
    require(
        np.all(labels >= -1)
        and np.isfinite(strength).all()
        and np.all((strength >= 0) & (strength <= 1)),
        "Density labels/strength range",
    )
    require(np.all(strength[labels < 0] == 0), "Noise has zero density strength")
    require(all(sum(labels == j) >= 5 for j in set(labels) - {-1}), "Native density size rule")
    # This checks geometry/invariants, not a second HDBSCAN implementation.
    from scipy.spatial.distance import cdist

    distance = cdist(a["scores"], a["scores"])
    require(np.isfinite(distance).all() and np.all(distance >= 0), "Finite Euclidean density space")


def audit_integrative(xm, xe, folder):
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.io.config import read_json

    parts = []
    for i, x in enumerate((xm, xe)):
        root = folder / f"view_{i:03d}" / "parameters"
        a = arrays(root / "pca.npz")
        profiles = arrays(root / "profiles.npz")
        score = profiles["profiles"]
        audit_pca(x, a, score)
        mean = score.mean(0)
        scale = np.sqrt(((score - mean) ** 2).sum() / (len(x) - 1))
        np.testing.assert_allclose(profiles["mean"], mean, atol=1e-12)
        np.testing.assert_allclose(profiles["scale"], scale, atol=1e-12)
        np.testing.assert_allclose(profiles["normalized_weight"], 0.5)
        require(read_json(root / "summary.json")["aggregation"] == "none", "No group aggregation")
        parts.append((score - mean) * np.sqrt(0.5) / scale)
    fused = np.column_stack(parts)
    a = arrays(folder / "fusion.npz")
    np.testing.assert_allclose(a["fused"], fused, atol=1e-11)
    np.testing.assert_allclose(a["mean"], fused.mean(0), atol=1e-12)
    c = a["components"]
    np.testing.assert_allclose(c @ c.T, np.eye(2), atol=1e-11)
    np.testing.assert_allclose(
        a["explained_variance"],
        np.linalg.eigvalsh(np.cov(fused, rowvar=False))[::-1][:2],
        atol=1e-11,
    )
    return (fused - a["mean"]) @ c.T
