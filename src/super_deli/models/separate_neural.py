"""Independent native encoders; concatenate four coordinates without joint reduction."""

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.models import tabular_neural as native
from super_deli.models.density_scores import fit_density_scores
from super_deli.models.matched_preprocessing import align_tables
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.neural_config import TRAINING
from super_deli.models.separate_config import SeparateNeuralConfig

ROLES = ("morphology", "ecology")
SEPARATE_TRAINING = {
    "per_view": TRAINING,
    "branches": 2,
    "updates_per_view": 500,
    "total_updates": 1000,
    "fusion": (
        "concatenate two standardized coordinates per view; "
        "equal view influence; no joint reduction"
    ),
}


def unpack_branch(arrays, role):
    prefix = role + "__"
    return {
        key.removeprefix(prefix): value for key, value in arrays.items() if key.startswith(prefix)
    }


def fusion_spectrum(scores):
    if (
        scores.ndim != 2
        or scores.shape[1] != 4
        or not np.isfinite(scores).all()
        or np.any(scores.std(0) <= 0)
    ):
        raise ValueError("Four finite nonconstant fused coordinates required")
    singular = np.linalg.svd(scores, compute_uv=False)
    rank = int(sum(singular > singular[0] * 1e-12))
    if rank < 2:
        raise ValueError("Fused numerical rank below two")
    return singular, rank


def prepare_branch(x):
    """C-order matches the standalone reader's declared row/feature layout."""
    x = np.array(x, dtype=float, order="C", copy=True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        means, scales = x.mean(axis=0), x.std(axis=0)
        kept = np.flatnonzero(scales > 0)
        if not 3 <= len(kept) <= 32:
            raise ValueError("Each encoder requires 3–32 nonconstant traits")
        scales[scales == 0] = 1.0
        z = (x[:, kept] - means[kept]) / scales[kept]
    return z, {"means": means, "scales": scales, "kept_indices": kept}


def fit_branch(x, seed):
    """Use the exact standalone preprocessing and unchanged native optimizer."""
    z, preprocessing = prepare_branch(x)
    weights, losses = native._train(z, seed)
    try:
        raw = native.forward_numpy(z, weights, "encoder")
        scores, center, scale, singular = native.latent_transform(raw)
    except Exception as exc:
        exc.arrays = {**weights, "losses": losses}
        raise
    return {
        **weights,
        **preprocessing,
        "latent_means": center,
        "latent_scales": scale,
        "scores": scores,
        "losses": losses,
        "latent_singular_values": singular,
    }


@dataclass(frozen=True)
class SeparateNeuralModel:
    config: SeparateNeuralConfig
    arrays: dict

    def _views(self, tables):
        x, alignment = align_tables(tables, self.config, minimum=1)
        split = len(self.config.morphology_features)
        return (x[:, :split], x[:, split:]), alignment

    def encode(self, tables):
        views, alignment = self._views(tables)
        scores = [
            native.NeuralModel(names, unpack_branch(self.arrays, role)).encode(x)
            for role, names, x in zip(
                ROLES,
                (self.config.morphology_features, self.config.ecology_features),
                views,
                strict=True,
            )
        ]
        return {
            "unit_ids": alignment["unit_ids"],
            "scores": np.column_stack(scores),
            "views": dict(zip((self.config.morphology, self.config.ecology), scores, strict=True)),
        }

    def reconstruct(self, tables):
        views, alignment = self._views(tables)
        result = [
            native.NeuralModel(names, unpack_branch(self.arrays, role)).reconstruct(x)
            for role, names, x in zip(
                ROLES,
                (self.config.morphology_features, self.config.ecology_features),
                views,
                strict=True,
            )
        ]
        return {
            "unit_ids": alignment["unit_ids"],
            "views": dict(zip((self.config.morphology, self.config.ecology), result, strict=True)),
        }


def fit_separate_point(x, *, config):
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
            for role, view in zip(ROLES, (x[:, :split], x[:, split:]), strict=True):
                branch = fit_branch(view, config.random_seed)
                branches.append(branch)
                saved.update({role + "__" + key: value for key, value in branch.items()})
            scores = np.column_stack([b["scores"] for b in branches])
            singular, rank = fusion_spectrum(scores)
            saved.update(fusion_singular_values=singular, fusion_rank=np.asarray(rank))
            labels, strength = fit_density_scores(scores, config.density_config())
        except Exception as exc:
            partial = {role + "__failed_" + k: v for k, v in getattr(exc, "arrays", {}).items()}
            exc.arrays = {**saved, **partial}
            exc.diagnostics = {
                **getattr(exc, "diagnostics", {}),
                "failed_role": role,
                "completed_branches": len(branches),
            }
            raise
    losses = 0.5 * (branches[0]["losses"] + branches[1]["losses"])
    diagnostics = {
        "training": SEPARATE_TRAINING,
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
        "loss_interpretation": "mean per-view reconstruction, not a jointly optimized objective",
        "species_significance": "not_computed",
    }
    return native.NeuralPoint(
        SeparateNeuralModel(config, saved), scores, labels, strength, losses, diagnostics
    )


def fit_separate_neural(tables, *, config):
    from super_deli.models.paired_subsampling import fit_paired_subsamples

    config = SeparateNeuralConfig.from_dict(config.to_dict())
    return fit_paired_subsamples(tables, config=config, point_function=fit_separate_point)
