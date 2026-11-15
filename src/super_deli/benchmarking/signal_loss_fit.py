"""Unchanged native point workflows and separately labelled cached-coordinate searches."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.multimodal_validation import array_hash
from super_deli.benchmarking.signal_loss_metrics import parameters
from super_deli.io.config import write_json
from super_deli.models.mixture import FIT_LOCK, select_gaussian_mixture


@contextmanager
def observe_candidates(directory):
    directory = Path(directory)
    directory.mkdir()
    records = []
    native = GaussianMixture.fit

    def capture(model, x, error=None):
        n, d = x.shape
        k = model.n_components
        record = dict(
            k=k,
            dimension=d,
            n=n,
            covariance_type=model.covariance_type,
            seed=int(model.random_state),
            n_init=model.n_init,
            iterations=int(getattr(model, "n_iter_", 0)),
            converged=bool(getattr(model, "converged_", False)),
            scores_sha256=array_hash(x),
            fit_status="failed" if error else "complete",
            error=error,
            arrays=None,
        )
        try:
            probabilities = model.predict_proba(x)
            record.update(bic=float(model.bic(x)), log_likelihood=float(model.score(x) * n))
            key = f"fit_{len(records):04d}_k{k}.npz"
            with (directory / key).open("xb") as f:
                np.savez(
                    f,
                    scores=x.copy(),
                    weights=model.weights_.copy(),
                    means=model.means_.copy(),
                    covariances=model.covariances_.copy(),
                    membership=probabilities,
                    labels=probabilities.argmax(1),
                )
            record["arrays"] = key
        except Exception as exc:
            record["export_error"] = f"{type(exc).__name__}: {exc}"
            if error is None:
                raise
        finally:
            records.append(record)

    def observed(model, x, *args, **kwargs):
        try:
            result = native(model, x, *args, **kwargs)
        except Exception as exc:
            capture(model, x, f"{type(exc).__name__}: {exc}")
            raise
        capture(model, x)
        return result

    try:
        with FIT_LOCK, threadpool_limits(limits=1), patch.object(GaussianMixture, "fit", observed):
            yield records
    finally:
        write_json(records, directory / "index.json")


def audit_candidates(directory):
    from scipy.special import logsumexp

    from super_deli.io.config import read_json

    records = read_json(Path(directory) / "index.json")
    for r in records:
        if r["fit_status"] == "failed":
            continue
        with np.load(Path(directory) / r["arrays"], allow_pickle=False) as a:
            x, w, mean, cov = a["scores"], a["weights"], a["means"], a["covariances"]
            if array_hash(x) != r["scores_sha256"]:
                raise ValueError("Candidate score hash mismatch")
            if r["covariance_type"] == "tied":
                cov = np.repeat(cov[None], len(w), axis=0)
            elif r["covariance_type"] == "diag":
                cov = np.array([np.diag(c) for c in cov])
            terms = []
            for p, m, c in zip(w, mean, cov, strict=True):
                sign, logdet = np.linalg.slogdet(c)
                if sign != 1 or np.linalg.eigvalsh(c).min() <= 0:
                    raise ValueError("Nonpositive covariance")
                delta = x - m
                terms.append(
                    np.log(p)
                    - 0.5
                    * (
                        x.shape[1] * np.log(2 * np.pi)
                        + logdet
                        + np.einsum("ij,ji->i", delta, np.linalg.solve(c, delta.T))
                    )
                )
            terms = np.array(terms).T
            total = logsumexp(terms, axis=1)
            np.testing.assert_allclose(
                np.exp(terms - total[:, None]), a["membership"], rtol=1e-7, atol=1e-9
            )
            np.testing.assert_array_equal(a["labels"], np.argmax(terms, axis=1))
            np.testing.assert_allclose(total.sum(), r["log_likelihood"], rtol=1e-8, atol=1e-7)
            bic = -2 * total.sum() + parameters(
                r["k"], r["dimension"], r["covariance_type"]
            ) * np.log(len(x))
            np.testing.assert_allclose(bic, r["bic"], rtol=1e-8, atol=1e-7)
    return len(records)


def fit_scores(matrix, config, seed, covariance, directory):
    directory = Path(directory)
    directory.mkdir()
    if config.min_cluster_size <= matrix.shape[1] or matrix.shape[1] != config.n_pcs:
        raise ValueError("Diagnostic score dimensions must retain native size guards")
    try:
        with observe_candidates(directory / "candidates"):
            point = select_gaussian_mixture(
                matrix,
                config,
                seed,
                covariance_type=covariance,
                unit_indices=np.arange(len(matrix)),
            )
        order = list(point.component_order)
        cov = point.mixture.covariances_
        cov = np.repeat(cov[None], point.selected_k, axis=0) if covariance == "tied" else cov[order]
        with (directory / "point.npz").open("xb") as f:
            np.savez(
                f,
                scores=matrix,
                labels=point.labels,
                membership=point.membership,
                weights=point.mixture.weights_[order],
                means=point.mixture.means_[order],
                covariances=cov,
            )
        result = dict(
            status="complete",
            selected_k=point.selected_k,
            candidates=point.candidates,
            covariance_type=covariance,
            dimension=matrix.shape[1],
            minimum=config.min_cluster_size,
            configuration=config.to_dict(),
            point_seed=seed,
            scope="benchmark-only head on fixed training coordinates; no upstream refit",
        )
    except Exception as exc:
        result = dict(status="failed", selected_k=None, error=f"{type(exc).__name__}: {exc}")
    result["audited_candidates"] = audit_candidates(directory / "candidates")
    write_json(result, directory / "summary.json")
    return result


def score_matrix(context, names, dimensions):
    # All cached contrasts use the parent joint model's row order and raw PCs.
    weight = 1 / len(names)
    blocks = []
    for name in names:
        view = context["views"][name]
        # Undo its original normalized weight, then apply equal selected-view weight.
        blocks.append(view["block"] * np.sqrt(weight / view["normalized_weight"]))
    fused = np.concatenate(blocks, axis=1)
    with threadpool_limits(limits=1):
        pca = PCA(n_components=dimensions, svd_solver="full", whiten=False).fit(fused)
    if pca.singular_values_[-1] <= 1e-12 * max(pca.singular_values_[0], 1):
        raise ValueError("Rank-deficient diagnostic PCA; do not alter its dimension")
    return fused, pca.transform(fused), pca.components_
