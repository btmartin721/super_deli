"""Isolated truth starts and exact ordinary-head replays; never production selection."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.coordinate_config import MixtureSettings
from super_deli.benchmarking.coordinate_fit import mixture
from super_deli.benchmarking.neural_analysis import require
from super_deli.io.config import write_json


def replay(scores, old, arrays):
    row, saved = mixture(scores, "tied")
    require(row["status"] == "available", "Successful exact replay required")
    require(saved.keys() == arrays.keys(), "Complete replay array coverage")
    for key in saved:
        np.testing.assert_array_equal(saved[key], arrays[key])
    require(
        {k: v for k, v in old.items() if k != "seconds"} == row, "Exact replay metadata/candidates"
    )
    return row, saved


def truth_fit(scores, truth, destination):
    destination.mkdir()
    cfg = MixtureSettings("tied")
    means = np.stack([scores[truth == k].mean(0) for k in range(4)])
    weights = np.bincount(truth, minlength=4) / len(truth)
    residual = scores - means[truth]
    covariance = residual.T @ residual / len(truth) + cfg.reg_covar * np.eye(scores.shape[1])
    np.savez_compressed(
        destination / "initialization.npz", means=means, weights=weights, covariance=covariance
    )
    row = dict(
        status="failed",
        attempted=1,
        role="Truth-initialized optimization diagnostic only",
        n_init=1,
    )
    saved = {}
    try:
        model = GaussianMixture(
            n_components=4,
            covariance_type="tied",
            n_init=1,
            weights_init=weights,
            means_init=means,
            precisions_init=np.linalg.inv(covariance),
            random_state=cfg.random_seed,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            reg_covar=cfg.reg_covar,
        )
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(scores)
        saved = {
            "k4_" + name: getattr(model, name + "_") for name in ("weights", "means", "covariances")
        }
        probabilities = model.predict_proba(scores)
        labels = probabilities.argmax(1)
        saved.update(k4_membership=probabilities, k4_labels=labels)
        row.update(
            converged=bool(model.converged_),
            iterations=int(model.n_iter_),
            warnings=[str(w.message) for w in caught],
            hard_cluster_sizes=np.bincount(labels, minlength=4).tolist(),
            eligible=bool(np.bincount(labels, minlength=4).min() >= 5),
            log_likelihood=float(model.score(scores) * len(scores)),
            bic=float(model.bic(scores)),
        )
        if model.converged_ and np.isfinite(row["bic"]) and np.isfinite(probabilities).all():
            row["status"] = "complete"
        else:
            row["error"] = (
                "Unresolved convergence or finite-density guard; diagnostic endpoint withheld"
            )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row["error"] = str(exc)
    np.savez_compressed(destination / "fit.npz", **saved)
    write_json(row, destination / "result.json")
    return row
