"""Complete-genotype point models and unchanged genomic PCA reference fits."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, xlogy
from threadpoolctl import threadpool_limits

from super_deli.models import genomic_autoencoder as categorical
from super_deli.models.genomic_neural_config import GenomicNeuralConfig
from super_deli.models.mixture import FIT_LOCK, select_gaussian_mixture
from super_deli.models.pca_gmm import _preprocess, fit_genomic_pca, fit_pca_gmm
from super_deli.models.tabular_neural import NeuralFitError, forward_numpy, latent_transform


def validate_dosages(values, config, *, minimum=10):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2:
        raise ValueError("A numeric diploid specimen-by-original-locus matrix is required")
    n, d = raw.shape
    if (
        n < minimum
        or not 2 <= d <= config.max_loci
        or not np.isfinite(raw).all()
        or not np.isin(raw, [0, 1, 2]).all()
    ):
        raise ValueError(
            "Use complete 0/1/2 diploid calls, >=10 specimens and the declared locus limit"
        )
    if 512 * n * d + 128 * n * n + 256 * d > config.max_working_bytes:
        raise ValueError("Genotype/neural workspace exceeds max_working_bytes")
    if config.n_subsamples and 112 * n * n > config.max_pairwise_bytes:
        raise ValueError("Specimen stability exceeds max_pairwise_bytes")
    return np.array(raw, dtype=float, copy=True)


def membership(scores, a):
    d = scores.shape[1]
    delta = scores[:, None] - a["mixture_means"][None]
    if (
        a["mixture_covariance"].ndim == 2
        and a["mixture_covariance"].shape == a["mixture_means"].shape
        and a.get("diagonal", np.asarray(False)).item()
    ):
        var = a["mixture_covariance"]
        logp = np.log(a["mixture_weights"])[None] - 0.5 * (
            d * np.log(2 * np.pi) + np.log(var).sum(1)[None] + (delta**2 / var[None]).sum(2)
        )
    else:
        chol = np.linalg.cholesky(a["mixture_covariance"])
        z = np.linalg.solve(chol, delta.reshape(-1, d).T).T.reshape(delta.shape)
        logp = np.log(a["mixture_weights"])[None] - 0.5 * (
            d * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum() + (z * z).sum(2)
        )
    result = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite frozen mixture membership")
    return result


@dataclass(frozen=True)
class GenomicModel:
    family: str
    arrays: dict
    locus_keys: tuple[str, ...]

    def _raw(self, values, locus_keys):
        x = np.asarray(values)
        if (
            tuple(locus_keys) != self.locus_keys
            or x.dtype.kind not in "iuf"
            or x.ndim != 2
            or x.shape[1] != len(self.locus_keys)
            or not len(x)
            or not np.isfinite(x).all()
            or not np.isin(x, [0, 1, 2]).all()
        ):
            raise ValueError(
                "Frozen projection requires complete calls in the identical declared locus "
                "order/coding"
            )
        a = self.arrays
        return (x[:, a["kept_indices"]] - a["means"]) / a["scales"]

    def encode(self, values, *, locus_keys):
        x = self._raw(values, locus_keys)
        a = self.arrays
        raw = (
            forward_numpy(x, a, "encoder")
            if self.family == "categorical_ae"
            else (x - a["pca_mean"]) @ a["pca_components"].T
        )
        return (raw - a["latent_means"]) / a["latent_scales"]

    def reconstruct_probabilities(self, values, *, locus_keys):
        if self.family != "categorical_ae":
            raise ValueError("PCA has no categorical decoder")
        return np.exp(categorical.reconstruction(self._raw(values, locus_keys), self.arrays))

    def predict(self, values, *, locus_keys):
        scores = self.encode(values, locus_keys=locus_keys)
        p = membership(scores, self.arrays)
        return dict(scores=scores, membership=p, labels=p.argmax(1))


@dataclass(frozen=True)
class GenomicPoint:
    model: GenomicModel
    candidates: tuple
    diagnostics: dict

    @property
    def arrays(self):
        return self.model.arrays

    @property
    def selected_k(self):
        return self.arrays["membership"].shape[1]


def _keys(keys, d):
    result = tuple(keys) if keys is not None else tuple(f"source_locus_{j}" for j in range(d))
    if (
        len(result) != d
        or any(not isinstance(k, str) or not k or k != k.strip() for k in result)
        or len(set(result)) != d
    ):
        raise ValueError("One unique nonempty key per original source locus required")
    return result


def _base(pre, x):
    return dict(
        kept_indices=pre.kept_indices,
        means=pre.means,
        scales=pre.scales,
        source_locus_count=np.asarray(x.shape[1], dtype=np.int64),
    )


def _mix(a, scores, config, seed):
    fitted = select_gaussian_mixture(scores, config, seed, covariance_type="tied")
    order = list(fitted.component_order)
    a.update(
        scores=scores,
        labels=fitted.labels,
        membership=fitted.membership,
        mixture_weights=fitted.mixture.weights_[order],
        mixture_means=fitted.mixture.means_[order],
        mixture_covariance=fitted.mixture.covariances_,
        diagonal=np.asarray(False),
    )
    return fitted


def fit_genomic_neural_point(values, *, config, locus_keys=None):
    config = GenomicNeuralConfig.from_dict(config.to_dict())
    x = validate_dosages(values, config)
    keys = _keys(locus_keys, x.shape[1])
    transformed, pre = _preprocess(x, config.baseline_config())
    a = _base(pre, x)
    target = x[:, pre.kept_indices].astype(np.int64)
    with FIT_LOCK, threadpool_limits(limits=1):
        try:
            weights, losses = categorical._train(transformed, target, config.random_seed)
            a.update(weights)
            raw = forward_numpy(transformed, weights, "encoder")
            scores, means, scales, singular = latent_transform(raw)
            a.update(
                losses=losses,
                latent_means=means,
                latent_scales=scales,
                latent_singular_values=singular,
            )
            fitted = _mix(a, scores, config, config.mixture_seed)
        except Exception as exc:
            raise NeuralFitError(
                str(exc),
                diagnostics=getattr(exc, "diagnostics", {}),
                arrays={**a, **getattr(exc, "arrays", {})},
            ) from exc
    observed = np.stack([(target == g).mean(0) for g in range(3)], axis=1)
    baseline = float(-xlogy(observed, observed).sum(1).mean())
    logp = categorical.reconstruction(transformed, weights)
    final = float(-np.take_along_axis(logp, target[..., None], axis=2).mean())
    if not np.isclose(final, losses[-1], rtol=1e-10, atol=1e-12):
        raise NeuralFitError("Frozen categorical loss mismatch", arrays=a)
    diagnostics = dict(
        seed=config.random_seed,
        completed_updates=500,
        initial_cross_entropy=float(losses[0]),
        final_cross_entropy=final,
        marginal_genotype_entropy=baseline,
        genotype_counts=np.bincount(target.ravel(), minlength=3).tolist(),
        retained_loci=len(pre.kept_indices),
        exclusions=list(pre.exclusions),
        parameter_count=sum(v.size for k, v in weights.items()),
        interpretation=(
            "Training reconstruction probabilities are not calibrated genotype "
            "uncertainty or species evidence"
        ),
    )
    return GenomicPoint(GenomicModel("categorical_ae", a, keys), fitted.candidates, diagnostics)


def fit_genomic_pca_shared_point(values, *, config, locus_keys=None):
    x = validate_dosages(values, config)
    keys = _keys(locus_keys, x.shape[1])
    with FIT_LOCK, threadpool_limits(limits=1):
        raw, pre, pca = fit_genomic_pca(x, config.baseline_config())
        scores, means, scales, singular = latent_transform(raw)
        a = _base(pre, x)
        a.update(
            pca_mean=pca.mean_,
            pca_components=pca.components_,
            latent_means=means,
            latent_scales=scales,
            latent_singular_values=singular,
        )
        fitted = _mix(a, scores, config, config.mixture_seed)
    return GenomicPoint(
        GenomicModel("pca_shared", a, keys),
        fitted.candidates,
        dict(retained_loci=len(pre.kept_indices), exclusions=list(pre.exclusions)),
    )


def fit_genomic_legacy_point(values, *, config, locus_keys=None):
    x = validate_dosages(values, config)
    keys = _keys(locus_keys, x.shape[1])
    fit = fit_pca_gmm(x, tuple(f"row_{i}" for i in range(len(x))), config=config.baseline_config())
    p = fit.point
    order = list(p.component_order)
    a = _base(p.preprocessing, x)
    a.update(
        pca_mean=p.pca.mean_,
        pca_components=p.pca.components_,
        latent_means=np.zeros(2),
        latent_scales=np.ones(2),
        scores=p.scores,
        labels=p.labels,
        membership=p.membership,
        mixture_weights=p.mixture.weights_[order],
        mixture_means=p.mixture.means_[order],
        mixture_covariance=p.mixture.covariances_[order],
        diagonal=np.asarray(True),
    )
    return GenomicPoint(
        GenomicModel("pca_diagonal", a, keys),
        p.candidates,
        dict(
            retained_loci=len(p.preprocessing.kept_indices),
            exclusions=list(p.preprocessing.exclusions),
            native_baseline=config.baseline_config().to_dict(),
        ),
    )


def load_genomic_model(directory):
    """Load checksummed numeric arrays only; no Python object deserialization."""
    import json
    from pathlib import Path

    from super_deli.core.provenance import fingerprint_file

    path = Path(directory)
    meta = json.loads((path / "model.json").read_text())
    if (
        meta.get("schema_version") != 1
        or type(meta.get("schema_version")) is not int
        or meta.get("family") not in {"categorical_ae", "pca_shared", "pca_diagonal"}
    ):
        raise ValueError("Unsupported genomic model schema/family")
    if fingerprint_file(path / "model.npz").sha256 != meta.get("model_sha256"):
        raise ValueError("Genomic numeric model checksum mismatch")
    with np.load(path / "model.npz", allow_pickle=False) as source:
        a = {k: source[k] for k in source.files}
    d = int(a["source_locus_count"])
    keys = _keys(meta["locus_keys"], d)
    kept = a["kept_indices"]
    if (
        kept.dtype.kind not in "iu"
        or kept.ndim != 1
        or len(kept) < 2
        or not np.array_equal(np.unique(kept), kept)
        or np.any(kept < 0)
        or np.any(kept >= d)
    ):
        raise ValueError("Invalid retained original locus indices")
    q = len(kept)
    for name, shape in (
        ("means", (q,)),
        ("scales", (q,)),
        ("latent_means", (2,)),
        ("latent_scales", (2,)),
    ):
        if a[name].shape != shape or not np.isfinite(a[name]).all():
            raise ValueError("Invalid genomic transformation arrays")
    if np.any(a["scales"] <= 0) or np.any(a["latent_scales"] <= 0):
        raise ValueError("Projection scales must be positive")
    if meta["family"] == "categorical_ae":
        for side, widths in (("encoder", (q, 16, 8, 2)), ("decoder", (2, 8, 16, 3 * q))):
            for i in range(3):
                for suffix, shape in (
                    ("weight", (widths[i + 1], widths[i])),
                    ("bias", (widths[i + 1],)),
                ):
                    value = a[f"{side}_{i}_{suffix}"]
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError("Invalid categorical network arrays")
    elif a["pca_mean"].shape != (q,) or a["pca_components"].shape != (2, q):
        raise ValueError("Invalid frozen PCA arrays")
    k = len(a["mixture_weights"])
    diagonal = meta["family"] == "pca_diagonal"
    if (
        a["diagonal"].shape != ()
        or bool(a["diagonal"]) != diagonal
        or a["mixture_means"].shape != (k, 2)
        or a["mixture_covariance"].shape != ((k, 2) if diagonal else (2, 2))
    ):
        raise ValueError("Invalid fitted mixture arrays")
    if (
        not all(np.isfinite(a[key]).all() for key in a)
        or np.any(a["mixture_weights"] <= 0)
        or not np.isclose(a["mixture_weights"].sum(), 1)
    ):
        raise ValueError("Invalid finite fitted model values")
    if diagonal:
        if np.any(a["mixture_covariance"] <= 0):
            raise ValueError("Mixture variances must be positive")
    else:
        np.linalg.cholesky(a["mixture_covariance"])
    return GenomicModel(meta["family"], a, keys)
