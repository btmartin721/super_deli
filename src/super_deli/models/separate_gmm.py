"""Unchanged native branch training followed by prespecified shared-covariance BIC."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from super_deli.models import separate_neural as separate
from super_deli.models.mixture import FIT_LOCK, select_gaussian_mixture
from super_deli.models.separate_gmm_config import SeparateGMMConfig
from super_deli.models.tabular_neural import NeuralPoint


@dataclass(frozen=True)
class SeparateGMMModel(separate.SeparateNeuralModel):
    """Numeric frozen transforms and conditional membership, without PyTorch."""

    def predict(self, tables):
        result = self.encode(tables)
        if (
            128 * len(result["scores"]) * (4 + len(self.arrays["mixture_weights"]))
            > self.config.max_working_bytes
        ):
            raise ValueError("Frozen mixture projection workspace exceeds budget")
        membership = mixture_membership(result["scores"], self.arrays)
        return {**result, "membership": membership, "labels": membership.argmax(1)}


def mixture_membership(scores, arrays):
    means, cov, weights = (
        arrays[n] for n in ("mixture_means", "mixture_covariance", "mixture_weights")
    )
    chol = np.linalg.cholesky(cov)
    delta = scores[:, None, :] - means[None, :, :]
    z = np.linalg.solve(chol, delta.reshape(-1, 4).T).T.reshape(delta.shape)
    logp = np.log(weights) - 0.5 * (
        4 * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum() + (z * z).sum(2)
    )
    membership = np.exp(logp - logsumexp(logp, axis=1)[:, None])
    if not np.isfinite(membership).all():
        raise ValueError("Nonfinite frozen mixture membership")
    return membership


def fit_shared_point(x, *, config):
    x = np.asarray(x, dtype=float)
    p = len(config.morphology_features) + len(config.ecology_features)
    if x.ndim != 2 or x.shape[1] != p or len(x) < 10 or not np.isfinite(x).all():
        raise ValueError("Complete matched rows in declared feature order required")
    if 192 * len(x) ** 2 + 160 * x.size > config.max_working_bytes:
        raise ValueError("Separate-encoder fitting workspace exceeds budget")
    split, saved, branches = len(config.morphology_features), {}, []
    role = "morphology"
    with FIT_LOCK, threadpool_limits(limits=1):
        try:
            for role, view in zip(separate.ROLES, (x[:, :split], x[:, split:]), strict=True):
                branch = separate.fit_branch(view, config.random_seed)
                branches.append(branch)
                saved.update({role + "__" + k: v for k, v in branch.items()})
            scores = np.column_stack([b["scores"] for b in branches])
            singular, rank = separate.fusion_spectrum(scores)
            saved.update(fusion_singular_values=singular, fusion_rank=np.asarray(rank))
            role = "mixture"
            fitted = select_gaussian_mixture(
                scores, config, config.mixture_seed, covariance_type="tied"
            )
            saved.update(
                membership=fitted.membership,
                mixture_weights=fitted.mixture.weights_[list(fitted.component_order)],
                mixture_means=fitted.mixture.means_[list(fitted.component_order)],
                mixture_covariance=fitted.mixture.covariances_,
            )
        except Exception as exc:
            exc.arrays = {
                **saved,
                **{role + "__failed_" + k: v for k, v in getattr(exc, "arrays", {}).items()},
            }
            exc.diagnostics = {
                **getattr(exc, "diagnostics", {}),
                "failed_role": role,
                "completed_branches": len(branches),
            }
            raise
    diagnostics = {
        "training": separate.SEPARATE_TRAINING,
        "seed": config.random_seed,
        "branch_seeds": [config.random_seed] * 2,
        "completed_updates": 1000,
        "fusion_rank": rank,
        "fusion_singular_values": singular.tolist(),
        "per_view_standardized_mse": [float(b["losses"][-1]) for b in branches],
        "branch_parameter_counts": [
            sum(v.size for k, v in b.items() if k.startswith(("encoder_", "decoder_")))
            for b in branches
        ],
        "candidates": list(fitted.candidates),
        "covariance_type": "tied",
        "mixture_seed": config.mixture_seed,
        "loss_interpretation": "mean per-view reconstruction, not a jointly optimized objective",
        "membership_interpretation": "conditional on fitted mixture, not species probability",
        "species_significance": "not_computed",
    }
    return NeuralPoint(
        SeparateGMMModel(config, saved),
        scores,
        fitted.labels,
        fitted.membership.max(1),
        0.5 * (branches[0]["losses"] + branches[1]["losses"]),
        diagnostics,
    )


def fit_separate_gmm(tables, *, config):
    from super_deli.models.paired_subsampling import fit_paired_subsamples

    config = SeparateGMMConfig.from_dict(config.to_dict())
    return fit_paired_subsamples(tables, config=config, point_function=fit_shared_point)
