"""Global GMM-search calibration against a fitted Gaussian in a frozen score space."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from super_deli.models.calibration_config import GaussianCalibrationConfig
from super_deli.models.cluster_config import ClusterConfig
from super_deli.models.mixture import FIT_LOCK, select_gaussian_mixture

SCOPE = (
    "Approximate global single-Gaussian score-space test; fitted-null parameters; "
    "fixed upstream preprocessing, aggregation, weights and PCA representation. "
    "Not a species-boundary test, cluster-specific support or probability of species. "
    "Gaussian rejection can reflect heavy tails or other density misspecification. "
    "One prespecified analysis; no adjustment across view subsets or configurations. "
    "Grouped profiles require the same iid score assumption; unequal sampling precision, "
    "spatial dependence and within-group sampling uncertainty are not calibrated."
)


def score_hash(scores):
    values = np.asarray(scores, dtype="<f8", order="C")
    return hashlib.sha256(values.tobytes()).hexdigest()


def bic_improvement(candidates):
    """The K search is part of the statistic; K=1 implies a zero lower bound."""
    eligible = [c for c in candidates if c["status"] == "eligible"]
    single = [c for c in eligible if c["k"] == 1]
    if len(single) != 1 or any(not np.isfinite(c["bic"]) for c in eligible):
        raise ValueError("Statistic requires an eligible finite K=1 BIC")
    return float(max(0.0, single[0]["bic"] - min(c["bic"] for c in eligible)))


def monte_carlo_summary(observed, records, *, alpha):
    """Keep failed draws in denominators; intervals describe numerical tail error only."""
    if not records or not np.isfinite(observed) or observed < 0:
        raise ValueError("A finite nonnegative statistic and simulation records are required")
    failed = sum(r["status"] == "failed" for r in records)
    if any(r["status"] not in {"complete", "failed"} for r in records):
        raise ValueError("Unknown null-draw status")
    statistics = [r["statistic"] for r in records if r["status"] == "complete"]
    if any(not np.isfinite(t) or t < 0 for t in statistics):
        raise ValueError("Null statistics must be finite and nonnegative")
    # Include ties (especially the atom at zero when K=1 wins) in the upper tail.
    exceedances = sum(t >= observed for t in statistics)
    requested = len(records)
    p = (1 + exceedances) / (requested + 1) if not failed else None
    interval = None
    if not failed:
        b, m = exceedances, requested
        interval = [
            0.0 if b == 0 else float(beta.ppf(0.025, b, m - b + 1)),
            1.0 if b == m else float(beta.ppf(0.975, b + 1, m - b)),
        ]
    return {
        "status": "available" if not failed else "incomplete",
        "requested": requested,
        "successful": requested - failed,
        "failed": failed,
        "exceedances": exceedances,
        "tail_rule": "T_null >= T_observed (ties included)",
        "p_value": p,
        "p_value_kind": "Approximate fitted-Gaussian parametric bootstrap with plus-one rank",
        "minimum_p_value": 1 / (requested + 1),
        "alpha": alpha,
        "resolution_reaches_alpha": 1 / (requested + 1) <= alpha,
        "reject_single_gaussian": p <= alpha if p is not None else None,
        "failed_outcome_bounds": [
            (1 + exceedances) / (requested + 1),
            (1 + exceedances + failed) / (requested + 1),
        ],
        "conditional_null_tail_interval_95": interval,
        "interval_scope": (
            "Clopper-Pearson binomial interval for the tail probability conditional on "
            "the observed statistic and fitted null, not a species confidence interval; "
            "failed-outcome bounds identify unknown null statistics and are not intervals"
        ),
    }


@dataclass(frozen=True)
class GaussianCalibration:
    summary: dict
    records: tuple[dict, ...]
    mean: np.ndarray
    covariance: np.ndarray
    scores: np.ndarray


def calibrate_gaussian_scores(
    scores,
    *,
    search: ClusterConfig,
    config: GaussianCalibrationConfig,
    point_seed=None,
    expected_point=None,
):
    """Fit the null once; simulate iid rows and repeat every GMM candidate search.

    The plus-one calculation does not provide exact size when nuisance parameters
    are estimated. Upstream pipelines are intentionally not refitted here.
    """
    search = type(search).from_dict(search.to_dict())
    config = GaussianCalibrationConfig.from_dict(config.to_dict())
    scores = np.asarray(scores)
    if (
        scores.ndim != 2
        or scores.dtype.kind not in "fiu"
        or not np.isfinite(scores).all()
        or scores.shape[1] != search.n_pcs
    ):
        raise ValueError("Scores must be a complete finite matrix matching fixed n_pcs")
    n, d = scores.shape
    if n <= d or n < 2 * search.min_cluster_size or search.max_k < 2:
        raise ValueError("Calibration requires enough independent scores and max_k >= 2")
    if 8 * (24 * n * d + 8 * d * d) + config.n_simulations * search.max_k * 2048 > (
        config.max_working_bytes
    ):
        raise ValueError("Score calibration workspace/logs exceed max_working_bytes")
    scores = np.array(scores, dtype=float, copy=True)
    mean = scores.mean(axis=0)
    centered = scores - mean
    covariance = centered.T @ centered / n
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= config.rank_tol * max(
        eigenvalues[-1], 1.0
    ):
        raise ValueError("Fitted Gaussian score covariance is rank deficient")
    factor = np.linalg.cholesky(covariance)
    seeds = np.random.SeedSequence(config.random_seed).spawn(config.n_simulations + 1)
    if point_seed is None:
        point_seed = int(seeds[0].generate_state(1)[0])
    if type(point_seed) is not int or point_seed < 0:
        raise ValueError("point_seed must be a nonnegative integer")
    records = []
    with FIT_LOCK, threadpool_limits(limits=search.n_threads):
        observed = select_gaussian_mixture(scores, search, point_seed, covariance_type="full")
        statistic = bic_improvement(observed.candidates)
        if expected_point is not None and (
            observed.selected_k != expected_point[0]
            or not np.isclose(statistic, expected_point[1], rtol=1e-9, atol=1e-9)
        ):
            raise ValueError("Point GMM search cannot reproduce the saved result in this runtime")
        for i, seed in enumerate(seeds[1:]):
            draw_seed, fit_seed = map(int, seed.generate_state(2))
            simulated = mean + np.random.default_rng(draw_seed).standard_normal((n, d)) @ factor.T
            record = {
                "replicate": i,
                "draw_seed": draw_seed,
                "fit_seed": fit_seed,
                "scores_sha256": score_hash(simulated),
            }
            try:
                fitted = select_gaussian_mixture(
                    simulated, search, fit_seed, covariance_type="full"
                )
                record.update(
                    status="complete",
                    statistic=bic_improvement(fitted.candidates),
                    selected_k=fitted.selected_k,
                    candidates=fitted.candidates,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                record.update(status="failed", error=str(exc))
            records.append(record)
    inference = monte_carlo_summary(statistic, records, alpha=config.alpha)
    return GaussianCalibration(
        {
            "schema_version": 1,
            "method": "fitted_gaussian_score_gmm_bic",
            "scope": SCOPE,
            "null_hypothesis": "Independent identically distributed rows from one full Gaussian",
            "statistic_definition": "BIC(K=1) - min eligible BIC(K=1..max_k)",
            "observed_statistic": statistic,
            "observed_selected_k": observed.selected_k,
            "observed_candidates": observed.candidates,
            "point_seed": point_seed,
            "n_units": n,
            "n_scores": d,
            "scores_sha256": score_hash(scores),
            "null_estimation": "Sample mean and full maximum-likelihood covariance (divisor N)",
            "null_covariance_eigenvalues": eigenvalues.tolist(),
            "calibration": config.to_dict(),
            "search_configuration": search.to_dict(),
            "inference": inference,
        },
        tuple(records),
        mean,
        covariance,
        scores,
    )
