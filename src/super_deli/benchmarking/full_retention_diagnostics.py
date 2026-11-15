"""Truth-initialized optimization checks, excluded from unsupervised selection."""

import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.signal_loss_fit import audit_candidates, observe_candidates
from super_deli.io.config import read_json, write_json
from super_deli.models.mixture import FIT_LOCK


def truth_initialized(point, truth, output):
    point, output = Path(point), Path(output)
    output.mkdir()
    if not (point / "summary.json").exists():
        result = dict(status="unavailable", reason="Native point withheld", selected=False)
        write_json(result, output / "summary.json")
        return result
    summary = read_json(point / "summary.json")
    config = summary["provenance"]["configuration"]["model"]
    lookup = dict(zip(truth["unit_ids"], truth["labels"], strict=True))
    y = np.asarray([lookup[u] for u in summary["unit_ids"]])
    assert set(y) == {0, 1, 2, 3}
    with np.load(point / "mixture.npz", allow_pickle=False) as a:
        x = a["scores"].copy()
    means = np.stack([x[y == j].mean(0) for j in range(4)])
    weights = np.bincount(y, minlength=4) / len(y)
    residual = x - means[y]
    cov = residual.T @ residual / len(y) + config["reg_covar"] * np.eye(x.shape[1])
    seed = int(np.random.SeedSequence([summary["point_seed"], 4]).generate_state(1)[0])
    model = GaussianMixture(
        n_components=4,
        covariance_type="tied",
        n_init=1,
        max_iter=config["max_iter"],
        tol=config["tol"],
        reg_covar=config["reg_covar"],
        random_state=seed,
        means_init=means,
        weights_init=weights,
        precisions_init=np.linalg.inv(cov),
    )
    error = None
    try:
        with FIT_LOCK, threadpool_limits(limits=1), observe_candidates(output / "candidates"):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x)
    except (ValueError, ConvergenceWarning, np.linalg.LinAlgError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    checked = audit_candidates(output / "candidates")
    result = dict(
        status="failed" if error else "complete",
        error=error,
        selected=False,
        truth_informed=True,
        optimization_guarantee=False,
        audited_candidates=checked,
    )
    with (output / "initialization.npz").open("xb") as f:
        np.savez(f, means=means, weights=weights, covariance=cov, truth=y)
    if error is None:
        labels = model.predict(x)
        sizes = np.bincount(labels, minlength=4)
        native = read_json(point / "candidates.json")
        selected = next(r for r in native if r["k"] == summary["selected_k"])
        result.update(
            k=4,
            bic=float(model.bic(x)),
            log_likelihood=float(model.score(x) * len(x)),
            sizes=sizes.tolist(),
            eligible=bool(min(sizes) >= config["min_cluster_size"]),
            beats_native_selected_bic=bool(model.bic(x) < selected["bic"]),
            native_selected_k=summary["selected_k"],
            native_selected_bic=selected["bic"],
        )
    write_json(result, output / "summary.json")
    return result
