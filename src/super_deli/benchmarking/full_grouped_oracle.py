"""Truth-based initialization for isolated optimization diagnosis, never selection."""

import warnings


def partition(labels, name):
    if name == "four":
        return labels
    if name == "axis_a":
        return labels // 2
    if name == "axis_b":
        return labels % 2
    raise ValueError("Unknown generating partition")


def fit_oracle(scores, truth, name, model, destination):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    from super_deli.io.config import write_json
    from super_deli.models.mixture import FIT_LOCK

    destination.mkdir()
    labels = partition(truth, name)
    k, d = len(np.unique(labels)), scores.shape[1]
    means = np.stack([scores[labels == j].mean(0) for j in range(k)])
    covariances = []
    for j in range(k):
        residual = scores[labels == j] - means[j]
        covariances.append(residual.T @ residual / len(residual) + model.reg_covar * np.eye(d))
    covariances = np.stack(covariances)
    weights = np.bincount(labels, minlength=k) / len(labels)
    np.savez(
        destination / "initialization.npz", means=means, covariances=covariances, weights=weights
    )
    record = {
        "attempted": True,
        "partition": name,
        "k": k,
        "role": "Truth-initialized full-covariance optimization diagnostic; never selection",
    }
    try:
        estimator = GaussianMixture(
            n_components=k,
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
        membership = estimator.predict_proba(scores)
        predicted = membership.argmax(1)
        bic = float(estimator.bic(scores))
        if not estimator.converged_ or not np.isfinite(bic) or not np.isfinite(membership).all():
            raise ValueError("Truth-initialized fit failed finite convergence checks")
        sizes = np.bincount(predicted, minlength=k)
        np.savez(
            destination / "mixture.npz",
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
            ari=float(adjusted_rand_score(truth, predicted)),
            partition_ari=float(adjusted_rand_score(labels, predicted)),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, ConvergenceWarning) as exc:
        record.update(status="failed", error=str(exc))
    write_json(record, destination / "result.json")
    return record


def audit_oracle(directory, truth, scores, name, model, record):
    import numpy as np

    from super_deli.benchmarking.recovery_analysis import _ari, _dense_mixture

    labels = partition(truth, name)
    k, d = len(np.unique(labels)), scores.shape[1]
    if (record["partition"], record["k"], record["attempted"]) != (name, k, True):
        raise ValueError("Truth diagnostic identity mismatch")
    if record["status"] != "complete":
        if record["status"] != "failed" or not record.get("error"):
            raise ValueError("Invalid failed truth-initialized diagnostic")
        return
    with np.load(directory / "initialization.npz", allow_pickle=False) as initial:
        for j in range(k):
            rows = scores[labels == j]
            mean = rows.sum(0) / len(rows)
            residual = rows - mean
            covariance = sum(np.outer(r, r) for r in residual) / len(
                rows
            ) + model.reg_covar * np.eye(d)
            np.testing.assert_allclose(initial["means"][j], mean, atol=1e-12)
            np.testing.assert_allclose(initial["covariances"][j], covariance, atol=1e-12)
        np.testing.assert_allclose(initial["weights"], np.bincount(labels) / len(labels), atol=0)
    with np.load(directory / "mixture.npz", allow_pickle=False) as fitted:
        np.testing.assert_array_equal(fitted["scores"], scores)
        ll = _dense_mixture(fitted)
        q = k - 1 + k * d + k * d * (d + 1) // 2
        np.testing.assert_allclose(record["bic"], -2 * ll + q * np.log(len(labels)), atol=1e-7)
        sizes = np.bincount(fitted["labels"], minlength=k)
        if (
            record["hard_cluster_sizes"] != sizes.tolist()
            or record["eligible"] != bool(sizes.min() >= model.min_cluster_size)
            or record["converged"] is not True
        ):
            raise ValueError("Truth-initialized eligibility/convergence mismatch")
        np.testing.assert_allclose(record["ari"], _ari(truth, fitted["labels"]), atol=1e-12)
        np.testing.assert_allclose(
            record["partition_ari"], _ari(labels, fitted["labels"]), atol=1e-12
        )
