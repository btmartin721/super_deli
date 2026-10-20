"""A diploid-dosage PCA/GMM baseline with complete-pipeline block resampling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from super_deli.models.config import BaselineConfig
from super_deli.models.mixture import FIT_LOCK as _FIT_LOCK
from super_deli.models.mixture import select_gaussian_mixture


class BaselineError(ValueError):
    """The baseline cannot estimate its declared statistic on the supplied data."""


@dataclass(frozen=True)
class Preprocessing:
    kept_indices: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    exclusions: tuple[dict, ...]
    imputed_calls: int
    sample_missing_fraction: np.ndarray


@dataclass(frozen=True)
class PointFit:
    labels: np.ndarray
    membership: np.ndarray
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    candidates: tuple[dict, ...]
    preprocessing: Preprocessing
    pca: PCA
    mixture: GaussianMixture
    component_order: tuple[int, ...]

    @property
    def selected_k(self) -> int:
        return self.membership.shape[1]


@dataclass(frozen=True)
class BootstrapStability:
    status: str
    requested: int
    successful: int
    failures: tuple[dict, ...]
    block_source: str
    block_count: int
    coassignment: np.ndarray | None
    coassignment_lower: np.ndarray | None
    coassignment_upper: np.ndarray | None
    exact_recovery_counts: tuple[int, ...]
    selected_k_counts: dict[int, int]
    replicates: tuple[dict, ...]


@dataclass(frozen=True)
class BaselineFit:
    point: PointFit
    stability: BootstrapStability
    specimen_ids: tuple[str, ...]
    config: BaselineConfig


def _preprocess(x: np.ndarray, config: BaselineConfig) -> tuple[np.ndarray, Preprocessing]:
    counts = np.sum(~np.isnan(x), axis=0)
    means = np.divide(np.nansum(x, axis=0), counts, out=np.zeros(x.shape[1]), where=counts > 0)
    centered = np.where(np.isnan(x), 0.0, x - means)
    variances = np.divide(
        np.sum(centered**2, axis=0), counts, out=np.zeros(x.shape[1]), where=counts > 0
    )
    keep = (counts > 0) & (variances > 0)
    indices = np.flatnonzero(keep)
    exclusions = tuple(
        {
            "locus_index": int(i),
            "reason": "all_missing" if counts[i] == 0 else "zero_dosage_variance",
        }
        for i in np.flatnonzero(~keep)
    )
    if len(indices) < config.n_pcs:
        raise BaselineError("Too few variable, observed loci for the requested PCA dimension")
    observed = ~np.isnan(x[:, keep])
    if np.any(~observed.any(axis=1)):
        raise BaselineError("A specimen has no observed calls at retained loci")
    missing = int((~observed).sum())
    if missing and config.missing_policy == "error":
        raise BaselineError("Missing calls require an explicit mean-imputation policy")
    if config.scaling == "standard":
        scales = np.sqrt(variances[keep])
    elif config.scaling == "patterson":
        p = means[keep] / 2
        scales = np.sqrt(2 * p * (1 - p))
    else:
        scales = np.ones(len(indices))
    transformed = centered[:, keep] / scales
    return transformed, Preprocessing(
        indices, means[keep], scales, exclusions, missing, np.isnan(x).mean(axis=1)
    )


def fit_genomic_pca(x: np.ndarray, config: BaselineConfig):
    """Fit preprocessing/PCA on validated diploid dosages, without a mixture."""
    transformed, preprocessing = _preprocess(x, config)
    pca = PCA(n_components=config.n_pcs, svd_solver="full")
    try:
        scores = pca.fit_transform(transformed)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise BaselineError(f"PCA failed: {exc}") from exc
    rank_tolerance = np.finfo(float).eps * max(transformed.shape) * pca.singular_values_[0]
    if pca.singular_values_[-1] <= rank_tolerance:
        raise BaselineError("The requested PCA space is rank deficient")
    return scores, preprocessing, pca


def project_genomic_pca(x, preprocessing, pca, *, missing_policy):
    retained = np.asarray(x, dtype=float)[:, preprocessing.kept_indices]
    if np.isnan(retained).all(axis=1).any():
        raise BaselineError("A projected specimen has no observed calls at retained loci")
    if missing_policy == "error" and np.isnan(retained).any():
        raise BaselineError("Missing calls require an explicit mean-imputation policy")
    centered = np.where(np.isnan(retained), 0, retained - preprocessing.means)
    scores = pca.transform(centered / preprocessing.scales)
    if not np.isfinite(scores).all():
        raise BaselineError("Nonfinite genomic projection")
    return scores


def _fit_once(x: np.ndarray, config: BaselineConfig, seed: int) -> PointFit:
    scores, preprocessing, pca = fit_genomic_pca(x, config)
    try:
        fit = select_gaussian_mixture(scores, config, seed, covariance_type="diag")
    except ValueError as exc:
        raise BaselineError(str(exc)) from exc
    return PointFit(
        fit.labels,
        fit.membership,
        scores,
        pca.explained_variance_ratio_.copy(),
        fit.candidates,
        preprocessing,
        pca,
        fit.mixture,
        fit.component_order,
    )


def _block_groups(block_ids: Sequence[str], n_loci: int) -> tuple[np.ndarray, ...]:
    if len(block_ids) != n_loci:
        raise BaselineError("Block IDs must cover every original locus, before exclusions")
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(block_ids):
        if not isinstance(name, str) or not name or name != name.strip():
            raise BaselineError("Block IDs must be nonempty, unpadded strings")
        groups.setdefault(name, []).append(index)
    if len(groups) < 2:
        raise BaselineError("Bootstrap stability needs at least two declared independent blocks")
    return tuple(np.asarray(indices, dtype=int) for indices in groups.values())


def draw_block_indices(groups: tuple[np.ndarray, ...], rng: np.random.Generator) -> np.ndarray:
    """Resample whole blocks uniformly, keeping all loci and repeated block draws."""
    return np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])


def coassignment_counts(partitions: Sequence[np.ndarray], n_samples: int) -> np.ndarray:
    """Count pairwise co-assignment without aligning arbitrary cluster labels."""
    counts = np.zeros((n_samples, n_samples), dtype=np.uint64)
    for labels in partitions:
        counts += np.equal.outer(labels, labels)
    return counts


def _bootstrap(x, point, config, groups, block_source, seeds) -> BootstrapStability:
    n_samples = x.shape[0]
    counts = np.zeros((n_samples, n_samples), dtype=np.uint64)
    recoveries = np.zeros(point.selected_k, dtype=np.int64)
    reference_groups = [
        frozenset(np.flatnonzero(point.labels == k)) for k in range(point.selected_k)
    ]
    selected_k: Counter = Counter()
    failures, records = [], []
    for i, child in enumerate(seeds):
        draw_seed, fit_seed = map(int, child.generate_state(2))
        indices = draw_block_indices(groups, np.random.default_rng(draw_seed))
        record = {
            "replicate": i,
            "draw_seed": draw_seed,
            "fit_seed": fit_seed,
            "resampled_locus_count": len(indices),
        }
        try:
            fit = _fit_once(x[:, indices], config, fit_seed)
        except BaselineError as exc:
            record.update(status="failed", error=str(exc))
            failures.append(record)
        else:
            counts += np.equal.outer(fit.labels, fit.labels)
            observed_groups = {
                frozenset(np.flatnonzero(fit.labels == k)) for k in range(fit.selected_k)
            }
            recoveries += [group in observed_groups for group in reference_groups]
            selected_k[fit.selected_k] += 1
            record.update(
                status="complete",
                selected_k=fit.selected_k,
                labels=fit.labels.tolist(),
                retained_locus_count=len(fit.preprocessing.kept_indices),
                candidates=fit.candidates,
            )
        records.append(record)
    successful = config.n_bootstrap - len(failures)
    lower = counts.astype(float) / config.n_bootstrap
    upper = (counts.astype(float) + len(failures)) / config.n_bootstrap
    np.fill_diagonal(lower, 1.0)
    np.fill_diagonal(upper, 1.0)
    return BootstrapStability(
        "unavailable" if failures else "available",
        config.n_bootstrap,
        successful,
        tuple(failures),
        block_source,
        len(groups),
        lower if not failures else None,
        lower,
        upper,
        tuple(map(int, recoveries)),
        dict(sorted(selected_k.items())),
        tuple(records),
    )


def fit_pca_gmm(
    dosages: np.ndarray,
    specimen_ids: Sequence[str],
    *,
    config: BaselineConfig,
    block_ids: Sequence[str] | None = None,
) -> BaselineFit:
    """Estimate candidate structure in diploid 0/1/2 dosages (NaN = missing).

    No truth labels, population map, locality labels, or taxonomic labels are used
    as features. Bootstrap support requires independent genomic blocks or an
    explicit unlinked-locus assumption. These assumptions cannot be verified here.
    """
    x = np.array(dosages, dtype=float, copy=True)
    ids = tuple(specimen_ids)
    if x.ndim != 2 or min(x.shape) == 0 or len(ids) != x.shape[0]:
        raise BaselineError("Dosages must be a nonempty specimen-by-locus matrix with matching IDs")
    if len(set(ids)) != len(ids):
        raise BaselineError(
            "Repeated specimens require an explicit aggregation policy before fitting"
        )
    if any(not isinstance(i, str) or not i or i != i.strip() for i in ids):
        raise BaselineError("Specimen IDs must be nonempty, unpadded strings")
    if not np.isin(x[~np.isnan(x)], [0, 1, 2]).all():
        raise BaselineError("Only diploid 0/1/2 dosages and NaN missing calls are supported")
    if x.shape[0] <= config.n_pcs or x.shape[0] < config.min_cluster_size:
        raise BaselineError("Too few independent specimens for the requested PCA/mixture settings")
    if block_ids is not None and config.assume_unlinked:
        raise BaselineError("Supply genomic blocks or assume unlinked loci, not both")
    groups = None
    block_source = "none"
    if block_ids is not None:
        groups = _block_groups(block_ids, x.shape[1])
        block_source = "provided"
    elif config.assume_unlinked:
        groups = _block_groups(tuple(f"locus_{i}" for i in range(x.shape[1])), x.shape[1])
        block_source = "assumed_unlinked"
    if config.n_bootstrap and groups is None:
        raise BaselineError("Bootstrap requires block IDs or assume_unlinked=true")
    if config.n_bootstrap and 32 * x.shape[0] ** 2 > config.max_pairwise_bytes:
        raise BaselineError(
            "Dense pairwise stability exceeds max_pairwise_bytes; raise the explicit budget"
        )
    seeds = np.random.SeedSequence(config.random_seed).spawn(config.n_bootstrap + 1)
    # threadpoolctl changes process-wide library settings. Serialize our own
    # contexts; callers running unrelated numerical threads should use processes.
    with _FIT_LOCK, threadpool_limits(limits=config.n_threads):
        point = _fit_once(x, config, int(seeds[0].generate_state(1)[0]))
        if config.n_bootstrap:
            stability = _bootstrap(x, point, config, groups, block_source, seeds[1:])
        else:
            stability = BootstrapStability(
                "not_computed",
                0,
                0,
                (),
                block_source,
                len(groups) if groups else 0,
                None,
                None,
                None,
                (),
                {},
                (),
            )
    return BaselineFit(point, stability, ids, config)
