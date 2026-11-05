"""Existing clustering fits on unchanged coordinates, with read-only candidate capture."""

import sys
from contextlib import contextmanager

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.coordinate_config import MixtureSettings, model_for
from super_deli.benchmarking.moons_hierarchy import retained_hierarchy, selection_trace
from super_deli.benchmarking.neural_analysis import require
from super_deli.models.density_scores import fit_density_scores
from super_deli.models.mixture import FIT_LOCK, select_gaussian_mixture


class AuditFailure(RuntimeError):
    """Stop dependent work after a source, replay or numerical-audit discrepancy."""


@contextmanager
def frozen_only():
    class Block:
        def find_spec(self, fullname, *args):
            if fullname.split(".")[0] in {"torch", "snpio"}:
                raise AuditFailure(
                    "Frozen diagnostic prohibits optional training/input import: " + fullname
                )

    if any(name.split(".")[0] in {"torch", "snpio"} for name in sys.modules):
        raise AuditFailure("Run the frozen diagnostic in a process without Torch/SNPio imports")
    block = Block()
    sys.meta_path.insert(0, block)
    try:
        yield
    finally:
        sys.meta_path.remove(block)


def density(point, space):
    captured = {}

    def factory(**kwargs):
        captured["estimator"] = HDBSCAN(**kwargs)
        return captured["estimator"]

    with FIT_LOCK, threadpool_limits(limits=1):
        labels, strength = fit_density_scores(
            point["scores"], model_for(space), estimator_class=factory
        )
        try:
            np.testing.assert_array_equal(labels, point["labels"])
            np.testing.assert_array_equal(strength, point["strength"])
            linkage, condensed = retained_hierarchy(captured["estimator"], len(labels), 5)
            np.testing.assert_array_equal(linkage, point["linkage"])
            np.testing.assert_array_equal(condensed, point["condensed"])
        except (AssertionError, ValueError) as exc:
            raise AuditFailure("Exact source density replay failed: " + str(exc)) from exc
    arrays = dict(labels=labels, strength=strength, linkage=linkage, condensed=condensed)
    policies = {}
    for name, method, root in (
        ("baseline", "eom", True),
        ("no_root", "eom", False),
        ("leaf", "leaf", True),
    ):
        trace, assigned, support = selection_trace(
            condensed, method=method, allow_single_cluster=root
        )
        arrays[name + "_labels"] = assigned
        arrays[name + "_strength"] = support
        policies[name] = trace
    np.testing.assert_array_equal(arrays["baseline_labels"], labels)
    np.testing.assert_allclose(arrays["baseline_strength"], strength, atol=1e-12)
    return dict(
        status="available",
        selected_k=len(set(labels) - {-1}),
        source_replay=True,
        policies=policies,
    ), arrays


@contextmanager
def capture_candidates(scores):
    """Observe the library fit after each attempted K; do not alter its optimizer."""
    original = GaussianMixture.fit
    records = []
    saved = {}

    def fitted(estimator, x, *args, **kwargs):
        k = estimator.n_components
        record = dict(
            k=k,
            status="started",
            covariance_type=estimator.covariance_type,
            random_state=int(estimator.random_state),
            n_init=estimator.n_init,
            max_iter=estimator.max_iter,
            tol=estimator.tol,
            reg_covar=estimator.reg_covar,
        )
        records.append(record)
        try:
            result = original(estimator, x, *args, **kwargs)
            record.update(
                status="complete",
                converged=bool(estimator.converged_),
                iterations=int(estimator.n_iter_),
            )
            return result
        except Exception as exc:
            record.update(
                status="failed",
                error=str(exc),
                converged=bool(getattr(estimator, "converged_", False)),
            )
            raise
        finally:
            for field in ("weights", "means", "covariances"):
                value = getattr(estimator, field + "_", None)
                if value is not None:
                    saved[f"k{k}_{field}"] = value.copy()
            if record["status"] == "complete":
                probabilities = estimator.predict_proba(scores)
                saved[f"k{k}_membership"] = probabilities.copy()
                saved[f"k{k}_labels"] = probabilities.argmax(1)
                record.update(
                    log_likelihood=float(estimator.score(scores) * len(scores)),
                    bic=float(estimator.bic(scores)),
                )

    GaussianMixture.fit = fitted
    try:
        yield records, saved
    finally:
        GaussianMixture.fit = original


def mixture(scores, covariance_type):
    cfg = MixtureSettings(covariance_type)
    record = dict(configuration=cfg.to_dict())
    with FIT_LOCK, threadpool_limits(limits=1):
        with capture_candidates(scores) as (attempts, saved):
            try:
                fitted = select_gaussian_mixture(
                    scores, cfg, cfg.random_seed, covariance_type=covariance_type
                )
                record.update(
                    status="available",
                    selected_k=fitted.selected_k,
                    candidates=list(fitted.candidates),
                    component_order=list(fitted.component_order),
                )
                saved.update(labels=fitted.labels, membership=fitted.membership)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                record.update(status="failed", selected_k=None, error=str(exc))
            finally:
                record["candidate_attempts"] = attempts
    require(
        len(attempts) <= 4 and [a["k"] for a in attempts] == list(range(1, len(attempts) + 1)),
        "No replacement or out-of-order candidate fits",
    )
    return record, saved
