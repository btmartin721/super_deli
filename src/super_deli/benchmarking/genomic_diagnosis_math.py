"""Pure array evaluations of archived maps; no estimators, training or random draws."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata


def require(condition, message):
    if not condition:
        raise ValueError(message)


def forward(x, arrays, side):
    value = x
    hidden = []
    for i in range(3):
        value = value @ arrays[f"{side}_{i}_weight"].T + arrays[f"{side}_{i}_bias"]
        if i < 2:
            value = np.tanh(value)
            hidden.append(value)
    return value, hidden


def jacobian(x, arrays):
    """Exact row-wise encoder derivatives with respect to scaled inputs."""
    first = np.tanh(x @ arrays["encoder_0_weight"].T + arrays["encoder_0_bias"])
    second = np.tanh(first @ arrays["encoder_1_weight"].T + arrays["encoder_1_bias"])
    derivative = (1 - first**2)[:, :, None] * arrays["encoder_0_weight"][None]
    derivative = np.einsum("ij,njd->nid", arrays["encoder_1_weight"], derivative)
    derivative *= (1 - second**2)[:, :, None]
    return np.einsum("ij,njd->nid", arrays["encoder_2_weight"], derivative)


def check_jacobian(x, arrays):
    analytic = jacobian(x[[0, -1]], arrays)
    d = x.shape[1]
    epsilon = 1e-6
    maximum = 0.0
    for i, row in enumerate(x[[0, -1]]):
        plus = np.repeat(row[None], d, axis=0)
        minus = plus.copy()
        plus[np.arange(d), np.arange(d)] += epsilon
        minus[np.arange(d), np.arange(d)] -= epsilon
        reference = (
            forward(plus, arrays, "encoder")[0] - forward(minus, arrays, "encoder")[0]
        ).T / (2 * epsilon)
        np.testing.assert_allclose(analytic[i], reference, rtol=1e-5, atol=2e-7)
        maximum = max(maximum, float(np.max(np.abs(analytic[i] - reference))))
    return maximum


def correlation(x, y):
    x, y = rankdata(x), rankdata(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def association(values, labels):
    total = np.sum((values - values.mean()) ** 2)
    if total == 0:
        return None
    return float(
        sum(
            np.sum(labels == k) * (values[labels == k].mean() - values.mean()) ** 2
            for k in np.unique(labels)
        )
        / total
    )


def burdens(raw, x, keep):
    frequency = raw.mean(0) / 2
    minor = np.where(frequency > 0.5, 2 - raw, raw)
    rare = np.minimum(frequency, 1 - frequency) <= 0.1
    return dict(
        minor_allele_dosage=minor.sum(1),
        heterozygous_fraction=(raw == 1).mean(1),
        rare_minor_carriers=((minor > 0) & rare[None]).sum(1),
        patterson_energy=(x * x).sum(1),
    ), np.minimum(frequency[keep], 1 - frequency[keep])


def parameters(k, diagonal):
    return k - 1 + 2 * k + (2 * k if diagonal else 3)


def candidate_costs(candidates, n, diagonal, minimum):
    rows = []
    require(len({r["k"] for r in candidates}) == len(candidates), "Unique candidate K values")
    for r in candidates:
        if r["status"] == "insufficient_samples":
            require(n < r["k"] * minimum, "Candidate feasibility agrees with specimen count")
            continue
        sizes = np.asarray(r["hard_cluster_sizes"])
        require(
            len(sizes) == r["k"] and int(sizes.sum()) == n and r["converged"] is True,
            "Candidate accounting/convergence",
        )
        require(
            r["status"] == ("eligible" if sizes.min() >= minimum else "small_component"),
            "Candidate size eligibility",
        )
        rows.append(
            dict(
                k=r["k"],
                status=r["status"],
                bic=float(r["bic"]),
                n_parameters=parameters(r["k"], diagonal),
                minimum_size=int(sizes.min()),
            )
        )
    base = next(r for r in rows if r["k"] == 1)
    for r in rows:
        r["parameter_cost"] = (r["n_parameters"] - base["n_parameters"]) * np.log(n)
        r["twice_likelihood_gain"] = base["bic"] - r["bic"] + r["parameter_cost"]
        r["bic_improvement"] = base["bic"] - r["bic"]
    return rows


def selected_density(scores, arrays):
    delta = scores[:, None] - arrays["mixture_means"][None]
    if bool(arrays["diagonal"]):
        covariance = arrays["mixture_covariance"]
        require(np.all(covariance > 0), "Positive mixture variances")
        quadratic = (delta**2 / covariance[None]).sum(2)
        determinant = np.log(covariance).sum(1)
    else:
        np.testing.assert_allclose(
            arrays["mixture_covariance"], arrays["mixture_covariance"].T, rtol=1e-12, atol=1e-12
        )
        chol = np.linalg.cholesky(arrays["mixture_covariance"])
        z = np.linalg.solve(chol, delta.reshape(-1, 2).T).T.reshape(delta.shape)
        quadratic = (z * z).sum(2)
        determinant = 2 * np.log(np.diag(chol)).sum()
    logp = np.log(arrays["mixture_weights"])[None] - 0.5 * (
        2 * np.log(2 * np.pi) + determinant + quadratic
    )
    evidence = logsumexp(logp, axis=1)
    return np.exp(logp - evidence[:, None]), float(evidence.sum())


def evaluate(raw, arrays, meta, minimum):
    require(
        raw.ndim == 2 and np.isfinite(raw).all() and np.isin(raw, [0, 1, 2]).all(),
        "Complete diploid calls",
    )
    means = raw.mean(0)
    keep = np.flatnonzero(np.var(raw, axis=0) > 0)
    scales = np.sqrt(2 * (means[keep] / 2) * (1 - means[keep] / 2))
    np.testing.assert_array_equal(keep, arrays["kept_indices"])
    np.testing.assert_array_equal(means[keep], arrays["means"])
    np.testing.assert_array_equal(scales, arrays["scales"])
    require(int(arrays["source_locus_count"]) == raw.shape[1], "Original panel dimensions")
    x = (raw[:, keep] - means[keep]) / scales
    neural = meta["family"] == "categorical_ae"
    if neural:
        latent, hidden = forward(x, arrays, "encoder")
        logits = forward(latent, arrays, "decoder")[0].reshape(len(raw), len(keep), 3)
        logp = logits - logsumexp(logits, axis=2, keepdims=True)
        loss = float(-np.take_along_axis(logp, raw[:, keep, None].astype(int), axis=2).mean())
        np.testing.assert_allclose(loss, arrays["losses"][-1], rtol=1e-10, atol=1e-12)
        require(arrays["losses"].shape == (501,), "Fixed archived update history")
        sensitivity = (
            jacobian(x, arrays) / scales[None, None] / arrays["latent_scales"][None, :, None]
        )
        derivative_error = check_jacobian(x, arrays)
        saturation = [float(np.mean(np.abs(h) >= 0.95)) for h in hidden]
    else:
        require(meta["family"] in ("pca_shared", "pca_diagonal"), "Supported saved representation")
        latent = (x - arrays["pca_mean"]) @ arrays["pca_components"].T
        _, _, vectors = np.linalg.svd(x - x.mean(0), full_matrices=False)
        np.testing.assert_allclose(
            arrays["pca_components"].T @ arrays["pca_components"],
            vectors[:2].T @ vectors[:2],
            rtol=1e-8,
            atol=1e-9,
        )
        sensitivity = np.broadcast_to(
            arrays["pca_components"][None]
            / scales[None, None]
            / arrays["latent_scales"][None, :, None],
            (len(raw), 2, len(keep)),
        )
        loss = None
        saturation = []
        derivative_error = None
    require(np.all(arrays["latent_scales"] > 0), "Positive coordinate scales")
    scores = (latent - arrays["latent_means"]) / arrays["latent_scales"]
    np.testing.assert_allclose(scores, arrays["scores"], rtol=1e-10, atol=1e-10)
    p, ll = selected_density(scores, arrays)
    np.testing.assert_allclose(p, arrays["membership"], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(p.argmax(1), arrays["labels"])
    diagonal = bool(arrays["diagonal"])
    require(diagonal == (meta["family"] == "pca_diagonal"), "Declared covariance family")
    k = p.shape[1]
    costs = candidate_costs(meta["candidates"], len(raw), diagonal, minimum)
    selected = min(
        (r for r in costs if r["status"] == "eligible"), key=lambda r: (r["bic"], r["k"])
    )
    require(selected["k"] == k, "Native BIC selection")
    np.testing.assert_allclose(
        -2 * ll + parameters(k, diagonal) * np.log(len(raw)), selected["bic"], rtol=1e-10, atol=1e-8
    )
    b, maf = burdens(raw, x, keep)
    radius = np.linalg.norm(scores, axis=1)
    normalized = (scores - scores.mean(0)) / scores.std(0)
    correlations = {
        name: dict(
            radius_spearman=correlation(value, radius),
            axis_spearman=[correlation(value, scores[:, i]) for i in range(2)],
            label_variance_fraction=association(value, arrays["labels"]),
        )
        for name, value in b.items()
    }
    local = np.linalg.norm(sensitivity, axis=1).mean(0)
    total = local.sum()
    sensitivity_share = float(local[maf <= 0.1].sum() / total) if total else None
    distances = np.linalg.norm(scores[:, None] - scores[None], axis=2)
    upper = np.triu_indices(len(raw), 1)
    same = arrays["labels"][upper[0]] == arrays["labels"][upper[1]]
    within = float(np.median(distances[upper][same])) if same.any() else None
    between = float(np.median(distances[upper][~same])) if (~same).any() else None
    result = dict(
        selected_k=k,
        at_search_boundary=k == max(r["k"] for r in meta["candidates"]),
        candidate_costs=costs,
        selected_bic_improvement=selected["bic_improvement"],
        twice_likelihood_gain=selected["twice_likelihood_gain"],
        parameter_cost=selected["parameter_cost"],
        retained_loci=len(keep),
        rare_retained_fraction=float(np.mean(maf <= 0.1)),
        rare_sensitivity_share=sensitivity_share,
        hidden_saturation=saturation,
        decoder_cross_entropy=loss,
        jacobian_maximum_error=derivative_error,
        coordinate_skew=(normalized**3).mean(0).tolist(),
        coordinate_excess_kurtosis=((normalized**4).mean(0) - 3).tolist(),
        burden_associations=correlations,
        within_label_median_distance=within,
        between_label_median_distance=between,
        evidence_scope="Descriptive associations; no significance or causal mechanism test",
    )
    return result, dict(
        burden_values=np.column_stack(list(b.values())),
        raw_latent=latent,
        rare_locus_mask=maf <= 0.1,
        locus_sensitivity=local,
    )
