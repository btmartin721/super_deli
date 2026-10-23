"""Descriptive recovery measures on fixed scores, with explicit BIC dimensions."""

import warnings

from super_deli.benchmarking.recovery_metrics import axis_r2, bic_terms


def recovery_metrics(directory, labels):
    from super_deli.benchmarking.recovery_metrics import recovery_metrics as common_metrics
    from super_deli.io.config import read_json

    result = common_metrics(directory, labels)
    d = len(result["joint_variance_ratio"])
    result["bic_terms"] = bic_terms(
        read_json(directory / "candidates.json"), result["selected_k"], len(labels), d, "tied"
    )
    return {**result, **upstream_metrics(directory, labels)}


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
    residual = scores - means[labels]
    covariance = residual.T @ residual / len(scores) + model.reg_covar * np.eye(scores.shape[1])
    covariances = np.repeat(covariance[None, :, :], 4, axis=0)
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
            covariance_type="tied",
            n_init=1,
            weights_init=weights,
            means_init=means,
            precisions_init=np.linalg.inv(covariance),
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
                covariances=np.repeat(estimator.covariances_[None, :, :], 4, axis=0),
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


def upstream_metrics(directory, labels):
    """Trace source scores and group profiles; repeated labels are descriptive only."""
    import numpy as np

    from super_deli.io.config import read_json

    units = read_json(directory / "summary.json")["unit_ids"]
    order = {u: i for i, u in enumerate(units)}
    result = {}
    for j, kind in enumerate(("morphological", "ecological")):
        base = directory / f"view_{j:03d}"
        values = np.load(base / "source_values.npy", allow_pickle=False)
        rows = read_json(base / "source_rows.json")
        target = np.array([order[u] for u in rows["target_ids"]])
        counts = np.bincount(target, minlength=len(units))
        means = np.stack([values[target == i].mean(0) for i in range(len(units))])
        with np.load(base / "parameters/pca.npz", allow_pickle=False) as pca:
            transformed = (values[:, pca["kept_indices"]] - pca["feature_means"]) / pca[
                "feature_scales"
            ]
            source_scores = (transformed - pca["mean"]) @ pca["components"].T
            profiles = np.stack([source_scores[target == i].mean(0) for i in range(len(units))])
            components = pca["components"]
            variance = pca["explained_variance_ratio"]
        with np.load(base / "parameters/profiles.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(profiles, saved["profiles"], atol=1e-12)
        result[kind] = {
            "raw_group_axis_r2": axis_r2(means, labels),
            "source_axis_r2": axis_r2(source_scores, labels[target]),
            "profile_axis_r2": axis_r2(profiles, labels),
            "source_pca_components": components.tolist(),
            "source_variance_ratio": variance.tolist(),
            "counts": counts.tolist(),
            "counts_by_label": [counts[labels == k].tolist() for k in range(4)],
            "count_axis_r2": axis_r2(counts[:, None], labels)
            if np.ptp(counts)
            else {"a": 0.0, "b": 0.0},
            "raw_group_mean": means.tolist(),
        }
    return {"upstream": result}
