"""Point-only constrained EM with retained multistart and failure evidence."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import kmeans_plusplus
from threadpoolctl import threadpool_limits

from super_deli.models.mixture import FIT_LOCK
from super_deli.models.observed_config import ObservedGMMConfig
from super_deli.models.observed_likelihood import (
    Parameters,
    constrain_covariance,
    evaluate,
    parameter_change,
    patterns,
    update,
)


def numeric_matrix(values):
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or not 2 <= raw.shape[1] <= 8:
        raise ValueError("Expected a numeric matrix with 2–8 continuous features")
    x = np.array(raw, dtype=float, copy=True)
    if np.isinf(x).any():
        raise ValueError("Only finite numbers or NaN missing cells are supported")
    return x


@dataclass(frozen=True)
class Normalization:
    means: np.ndarray
    scales: np.ndarray

    def transform(self, x):
        if (
            self.means.shape != (x.shape[1],)
            or self.scales.shape != self.means.shape
            or not np.isfinite(self.means).all()
            or not np.isfinite(self.scales).all()
            or np.any(self.scales <= 0)
        ):
            raise ValueError("Invalid frozen training normalization")
        with np.errstate(over="raise", invalid="raise"):
            return (x - self.means) / self.scales


def normalization(x):
    count = (~np.isnan(x)).sum(axis=0)
    if np.any(count == 0):
        raise ValueError("A selected feature is unobserved")
    mean = np.nansum(x, axis=0) / count
    variance = np.nansum((x - mean) ** 2, axis=0) / count
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("A selected feature is constant or has invalid variance")
    return Normalization(mean, np.sqrt(variance))


def validate_training(x, config):
    n, d = x.shape
    estimate = 8 * (
        40 * n * d * d
        + n * config.max_k * 20
        + config.n_init * config.max_k * (config.max_iter + 1)
    )
    if estimate > config.max_working_bytes:
        raise ValueError("Observed-GMM workspace exceeds max_working_bytes")
    if d != len(config.feature_names):
        raise ValueError("Matrix must match the configured feature order")
    if n < max(config.min_cluster_size, d + 2):
        raise ValueError("Too few informative independent units")
    mask = (~np.isnan(x)).astype(np.int64)
    counts = mask.T @ mask
    if np.min(counts) < config.min_coobserved:
        raise ValueError("Insufficient feature/pair co-observation information")
    if not np.isfinite(np.nanvar(x, axis=0)).all() or np.any(np.nanvar(x, axis=0) <= 0):
        raise ValueError("A selected feature is constant or has invalid variance")
    return counts


def initialize(x, k, seed, bound):
    filled = np.where(np.isnan(x), np.nanmean(x, axis=0), x)
    means, _ = kmeans_plusplus(filled, k, random_state=seed)
    centered = filled - filled.mean(axis=0)
    cov = constrain_covariance(centered.T @ centered / len(x), bound)
    return Parameters(np.full(k, 1 / k), means, np.repeat(cov[None], k, axis=0))


def fit_start(x, initial, groups, config):
    current = initial
    previous = initial
    trace = [float(evaluate(len(x), current, groups)[0].sum())]
    status, error, residual = "failed", "Iteration limit reached", None
    for _ in range(config.max_iter):
        try:
            proposed = update(len(x), current, groups, config.covariance_bound)
            value = float(evaluate(len(x), proposed, groups)[0].sum())
            decrease_tolerance = max(1e-8, 128 * np.finfo(float).eps * max(1.0, abs(trace[-1])))
            if value < trace[-1] - decrease_tolerance:
                raise FloatingPointError("Observed likelihood decreased beyond tolerance")
            gain = (value - trace[-1]) / len(x)
            change = parameter_change(current, proposed)
            previous, current = current, proposed
            trace.append(value)
            if abs(gain) <= config.tol and change <= config.fixed_point_tol:
                residual = parameter_change(
                    current, update(len(x), current, groups, config.covariance_bound)
                )
                if residual <= config.fixed_point_tol:
                    status, error = "converged", None
                    break
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            error = str(exc)
            break
    return dict(
        status=status,
        error=error,
        iterations=len(trace) - 1,
        log_likelihood=trace[-1],
        fixed_point_residual=residual,
    ), {
        "initial": initial,
        "previous": previous,
        "final": current,
        "trace": np.asarray(trace),
    }


@dataclass(frozen=True)
class ObservedFit:
    summary: dict
    arrays: dict
    normalization: Normalization
    parameters: Parameters | None
    kept_indices: np.ndarray

    def predict(self, values):
        if self.parameters is None:
            raise ValueError("A failed search cannot predict")
        x = numeric_matrix(values)
        if x.shape[1] != len(self.normalization.means):
            raise ValueError("Prediction feature order/dimension must match training")
        keep = ~np.isnan(x).all(axis=1)
        density = np.full(len(x), np.nan)
        membership = np.full((len(x), len(self.parameters.weights)), np.nan)
        if keep.any():
            scaled = self.normalization.transform(x[keep])
            local_density, local_membership = evaluate(
                len(scaled), self.parameters, patterns(scaled)
            )
            # Marginal densities in original feature units, with a row-specific Jacobian.
            density[keep] = local_density - np.sum(
                np.where(np.isnan(x[keep]), 0.0, np.log(self.normalization.scales)), axis=1
            )
            membership[keep] = local_membership
        return density, membership


def fit_observed_matrix(values, *, config: ObservedGMMConfig, fixed_normalization=None):
    """Fit point candidates; failed feasible candidates withhold the whole selection.

    fixed_normalization supports prespecified paired simulation comparisons. It
    must come from training data and is never learned from prediction values.
    """
    x = numeric_matrix(values)
    keep = np.flatnonzero(~np.isnan(x).all(axis=1))
    x = x[keep]
    counts = validate_training(x, config)
    norm = normalization(x) if fixed_normalization is None else fixed_normalization
    scaled = norm.transform(x)
    groups = patterns(scaled)
    arrays = dict(
        raw=x,
        observed=~np.isnan(x),
        kept_indices=keep,
        normalization_means=norm.means,
        scales=norm.scales,
        coobserved=counts,
    )
    candidates, accepted = [], []
    failed = False
    with FIT_LOCK, threadpool_limits(limits=1):
        for k in range(1, config.max_k + 1):
            if len(x) < k * config.min_cluster_size:
                candidates.append(dict(k=k, status="insufficient_samples", starts=[]))
                continue
            starts, results = [], []
            for j in range(config.n_init):
                seed = int(np.random.SeedSequence([config.seed, k, j]).generate_state(1)[0])
                try:
                    initial = initialize(scaled, k, seed, config.covariance_bound)
                    record, evidence = fit_start(scaled, initial, groups, config)
                    for stage in ("initial", "previous", "final"):
                        p = evidence[stage]
                        for name in ("weights", "means", "covariances"):
                            arrays[f"k{k}_s{j}_{stage}_{name}"] = getattr(p, name)
                    arrays[f"k{k}_s{j}_trace"] = evidence["trace"]
                    if record["status"] == "converged":
                        results.append((record["log_likelihood"], -j, evidence["final"]))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    record = dict(status="failed", error=str(exc), iterations=0)
                starts.append(dict(start=j, seed=seed, **record))
            if not results:
                candidates.append(dict(k=k, status="numerical_failure", starts=starts))
                failed = True
                continue
            likelihood, negative_index, parameters = max(results, key=lambda a: (a[0], a[1]))
            density, membership = evaluate(len(x), parameters, groups)
            labels = membership.argmax(axis=1)
            sizes = np.bincount(labels, minlength=k)
            status = "eligible" if sizes.min() >= config.min_cluster_size else "small_component"
            parameter_count = (
                k * scaled.shape[1] + k * scaled.shape[1] * (scaled.shape[1] + 1) // 2 + k - 1
            )
            bic = -2 * likelihood + parameter_count * np.log(len(x))
            record = dict(
                k=k,
                status=status,
                log_likelihood=likelihood,
                bic=float(bic),
                parameter_count=parameter_count,
                best_start=-negative_index,
                hard_cluster_sizes=sizes.tolist(),
                starts=starts,
                eigenvalues=np.linalg.eigvalsh(parameters.covariances).tolist(),
            )
            record["active_bound_eigenvalues"] = int(
                np.sum(
                    np.linalg.eigvalsh(parameters.covariances)
                    <= config.covariance_bound * (1 + 1e-7)
                )
            )
            candidates.append(record)
            for name in ("weights", "means", "covariances"):
                arrays[f"k{k}_{name}"] = getattr(parameters, name)
            arrays[f"k{k}_membership"] = membership
            arrays[f"k{k}_density_scaled"] = density
            if status == "eligible":
                accepted.append((bic, k, parameters, membership))
    selected = None
    if not failed and accepted:
        _, k, selected, membership = min(accepted, key=lambda a: (a[0], a[1]))
        labels = membership.argmax(axis=1)
        order = np.array(sorted(range(k), key=lambda c: int(np.flatnonzero(labels == c)[0])))
        selected = Parameters(
            selected.weights[order], selected.means[order], selected.covariances[order]
        )
        arrays["membership"] = membership[:, order]
        arrays["labels"] = arrays["membership"].argmax(axis=1)
        for name in ("weights", "means", "covariances"):
            arrays[name] = getattr(selected, name)
    summary = dict(
        schema_version=1,
        status="complete" if selected is not None else "failed",
        model="observed_coordinate_full_gmm",
        config=config.to_dict(),
        selected_k=None if selected is None else len(selected.weights),
        informative_rows=len(x),
        excluded_all_missing=int(len(values) - len(x)),
        missing_patterns=len(groups),
        coobserved=counts.tolist(),
        candidates=candidates,
        covariance_constraint=(
            "Sigma >= covariance_bound * I in fixed observed-value training scales"
        ),
        objective="Constrained observed-coordinate log likelihood; no additive ridge M-step",
        p_value=None,
        stability="not_computed",
        selection="heuristic_BIC",
        assumptions=[
            "Independent units",
            "Ignorable MCAR/MAR with distinct parameters",
            "Gaussian mixture in selected continuous features",
        ],
        limitations=[
            "No species probabilities or significance guarantee",
            "No missingness-model or parameter uncertainty integration",
            "Multistart EM does not guarantee a global optimum",
            "Pair counts do not prove mixture identifiability",
            "Fixed scaled covariance constraint can affect selection",
        ],
    )
    return ObservedFit(summary, arrays, norm, selected, keep)
