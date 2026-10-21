"""Descriptive recovery measures on fixed scores, with explicit BIC dimensions."""

import warnings


def axis_r2(scores, labels):
    import numpy as np

    x = np.column_stack((np.ones(len(labels)), scores))
    result = {}
    for name, y in (("a", 2 * (labels // 2) - 1), ("b", 2 * (labels % 2) - 1)):
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        result[name] = float(1 - residual @ residual / ((y - y.mean()) @ (y - y.mean())))
    return result


def bic_terms(candidates, selected_k, n, d):
    import numpy as np

    rows = []
    for c in candidates:
        p = c["k"] * (d + d * (d + 1) // 2) + c["k"] - 1
        penalty = float(p * np.log(n))
        rows.append(
            {
                **c,
                "n": n,
                "d": d,
                "parameters": p,
                "penalty": penalty,
                "log_likelihood_from_bic": None if c["bic"] is None else (penalty - c["bic"]) / 2,
            }
        )
    selected = next(c for c in rows if c["k"] == selected_k)
    for c in rows:
        c["bic_minus_selected"] = None if c["bic"] is None else c["bic"] - selected["bic"]
        c["penalty_minus_selected"] = c["penalty"] - selected["penalty"]
        c["twice_log_likelihood_gain"] = (
            None
            if c["bic"] is None
            else 2 * (c["log_likelihood_from_bic"] - selected["log_likelihood_from_bic"])
        )
    return rows


def recovery_metrics(directory, labels):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, homogeneity_score

    from super_deli.io.config import read_json

    summary = read_json(directory / "summary.json")
    with np.load(directory / "fusion.npz", allow_pickle=False) as arrays:
        fusion = {k: arrays[k] for k in arrays.files}
    with np.load(directory / "mixture.npz", allow_pickle=False) as arrays:
        mixture = {k: arrays[k] for k in arrays.files}
    predicted = mixture["labels"]
    offsets, views, start = [], {}, 0
    for i, view in enumerate(summary["views"]):
        with np.load(
            directory / f"view_{i:03d}/parameters/profiles.npz", allow_pickle=False
        ) as arrays:
            profiles = arrays["profiles"]
            views[view["kind"]] = axis_r2(profiles, labels)
            offsets.append((start, start + profiles.shape[1]))
            start += profiles.shape[1]
    loadings = {
        view["kind"]: np.sum(fusion["components"][:, lo:hi] ** 2, axis=1).tolist()
        for view, (lo, hi) in zip(summary["views"], offsets, strict=True)
    }
    return {
        "selected_k": summary["selected_k"],
        "ari": float(adjusted_rand_score(labels, predicted)),
        "axis_homogeneity": {
            name: float(homogeneity_score(axis, predicted))
            for name, axis in (("a", labels // 2), ("b", labels % 2))
        },
        "view_axis_r2": views,
        "fused_axis_r2": axis_r2(fusion["fused"], labels),
        "joint_axis_r2": axis_r2(mixture["scores"], labels),
        "joint_variance_ratio": fusion["explained_variance_ratio"].tolist(),
        "joint_squared_loading_by_view": loadings,
        "bic_terms": bic_terms(
            read_json(directory / "candidates.json"),
            summary["selected_k"],
            len(labels),
            mixture["scores"].shape[1],
        ),
    }


def truth_initialized(scores, labels, model, destination):
    """One unrestricted EM fit after a truth-based initialization; never selection."""
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    from super_deli.io.config import write_json
    from super_deli.models.mixture import FIT_LOCK

    destination.mkdir(exist_ok=False)
    means = np.stack([scores[labels == k].mean(axis=0) for k in range(4)])
    covariances = []
    for k in range(4):
        residual = scores[labels == k] - means[k]
        covariances.append(
            residual.T @ residual / len(residual) + model.reg_covar * np.eye(scores.shape[1])
        )
    covariances = np.stack(covariances)
    weights = np.bincount(labels, minlength=4) / len(labels)
    with (destination / "initialization.npz").open("xb") as handle:
        np.savez(handle, weights=weights, means=means, covariances=covariances)
    record = {
        "role": "truth-initialized forced K=4 optimization diagnostic; not unsupervised selection",
        "attempted": True,
    }
    try:
        estimator = GaussianMixture(
            n_components=4,
            covariance_type="full",
            n_init=1,
            weights_init=weights,
            means_init=means,
            precisions_init=np.linalg.inv(covariances),
            random_state=model.random_seed,
            max_iter=model.max_iter,
            tol=model.tol,
            reg_covar=model.reg_covar,
        )
        with FIT_LOCK, threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(scores)
        if not estimator.converged_:
            raise ValueError("Truth-initialized EM did not converge")
        membership = estimator.predict_proba(scores)
        predicted = membership.argmax(axis=1)
        sizes = np.bincount(predicted, minlength=4)
        bic = float(estimator.bic(scores))
        if not np.isfinite(bic) or not np.isfinite(membership).all():
            raise ValueError("Nonfinite truth-initialized EM result")
        with (destination / "mixture.npz").open("xb") as handle:
            np.savez(
                handle,
                scores=scores,
                weights=estimator.weights_,
                means=estimator.means_,
                covariances=estimator.covariances_,
                membership=membership,
                labels=predicted,
            )
        record.update(
            status="complete",
            bic=bic,
            converged=True,
            iterations=int(estimator.n_iter_),
            hard_cluster_sizes=sizes.tolist(),
            eligible=bool(sizes.min() >= model.min_cluster_size),
            ari=float(adjusted_rand_score(labels, predicted)),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, ConvergenceWarning) as exc:
        record.update(status="failed", error=str(exc))
    write_json(record, destination / "result.json")
    return record
