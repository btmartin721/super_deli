"""Benchmark-only four-coordinate controls; no public model promotion."""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.models import tabular_neural as native
from super_deli.models.density_scores import fit_density_scores
from super_deli.models.matched_neural import MatchedNeuralModel
from super_deli.models.matched_neural_config import MatchedNeuralConfig
from super_deli.models.matched_preprocessing import align_tables, prepare_views
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.neural_config import TRAINING
from super_deli.models.separate_config import SeparateNeuralConfig
from super_deli.models.separate_neural import ROLES, fusion_spectrum, prepare_branch, unpack_branch

_train = native._train
JOINT4 = "benchmark_joint_four_autoencoder_hdbscan"
PCA4 = "benchmark_per_view_pca_hdbscan"


@dataclass(frozen=True)
class ControlConfig(SeparateNeuralConfig):
    model_family: str = JOINT4

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.model_family not in (JOINT4, PCA4)
        ):
            raise ValueError("Unsupported benchmark control")
        if self.model_family == PCA4 and self.n_subsamples:
            raise ValueError("PCA control is point-only in this study")
        MatchedNeuralConfig(
            **{**self.to_dict(), "model_family": "matched_morphology_ecology_autoencoder_hdbscan"}
        )


def network_four(d):
    torch = native._torch()

    def layers(widths):
        result = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            result.append(torch.nn.Linear(a, b, dtype=torch.float64, device="cpu"))
            if i < 2:
                result.append(torch.nn.Tanh())
        return torch.nn.Sequential(*result)

    return torch.nn.ModuleDict({"encoder": layers([d, 16, 8, 4]), "decoder": layers([4, 8, 16, d])})


@contextmanager
def four_coordinate_factory():
    with FIT_LOCK:
        old = native._network
        native._network = network_four
        try:
            yield
        finally:
            native._network = old


def normalize_four(raw):
    means, scales = raw.mean(0), raw.std(0)
    if not np.isfinite(raw).all() or np.any(scales <= 0):
        raise ValueError("Collapsed/nonfinite four-coordinate representation")
    scores = (raw - means) / scales
    singular, rank = fusion_spectrum(scores)
    return scores, means, scales, singular, rank


def fit_joint_four(x, *, config):
    z, prep = prepare_views(x, len(config.morphology_features))
    with FIT_LOCK, threadpool_limits(limits=1):
        weights, losses = _train(z, config.random_seed)
        try:
            scores, mean, scale, singular, rank = normalize_four(
                native.forward_numpy(z, weights, "encoder")
            )
            labels, strength = fit_density_scores(scores, config.density_config())
        except Exception as exc:
            exc.arrays = {**prep, **weights, "losses": losses}
            raise
    a = {
        **prep,
        **weights,
        "latent_means": mean,
        "latent_scales": scale,
        "fusion_singular_values": singular,
        "fusion_rank": np.asarray(rank),
    }
    residual = (
        native.forward_numpy(native.forward_numpy(z, weights, "encoder"), weights, "decoder") - z
    ) / a["column_weights"]
    split = int(a["retained_view_counts"][0])
    errors = [float(np.mean(residual[:, :split] ** 2)), float(np.mean(residual[:, split:] ** 2))]
    diagnostics = {
        "benchmark_control": JOINT4,
        "fusion_rank": rank,
        "completed_updates": 500,
        "per_view_standardized_mse": errors,
        "branch_parameter_counts": [sum(v.size for v in weights.values())],
    }
    return native.NeuralPoint(
        MatchedNeuralModel(config, a), scores, labels, strength, losses, diagnostics
    )


@dataclass(frozen=True)
class PCAViewsModel:
    config: ControlConfig
    arrays: dict

    def encode(self, tables):
        x, alignment = align_tables(tables, self.config, minimum=1)
        split = len(self.config.morphology_features)
        scores = []
        for role, view in zip(ROLES, (x[:, :split], x[:, split:]), strict=True):
            a = unpack_branch(self.arrays, role)
            kept = a["kept_indices"]
            z = (view[:, kept] - a["means"][kept]) / a["scales"][kept]
            raw = (z - a["pca_mean"]) @ a["components"].T
            scores.append((raw - a["latent_means"]) / a["latent_scales"])
        return {"unit_ids": alignment["unit_ids"], "scores": np.column_stack(scores)}


def fit_pca_four(x, *, config, clusterer=None):
    split = len(config.morphology_features)
    saved = {}
    parts = []
    errors = []
    for role, view in zip(ROLES, (x[:, :split], x[:, split:]), strict=True):
        z, a = prepare_branch(view)
        center = z.mean(0)
        _, singular, vt = np.linalg.svd(z - center, full_matrices=False)
        if singular[1] <= 1e-12 * singular[0]:
            raise ValueError("PCA control lacks two resolved axes")
        components = vt[:2].copy()
        signs = np.sign(components[np.arange(2), np.argmax(abs(components), axis=1)])
        components *= signs[:, None]
        raw = (z - center) @ components.T
        mean, scale = raw.mean(0), raw.std(0)
        if np.any(scale <= 0):
            raise ValueError("PCA control has a constant coordinate")
        scores = (raw - mean) / scale
        a.update(
            pca_mean=center,
            components=components,
            singular_values=singular,
            latent_means=mean,
            latent_scales=scale,
            scores=scores,
        )
        saved.update({role + "__" + k: v for k, v in a.items()})
        parts.append(scores)
        errors.append(float(np.mean((raw @ components + center - z) ** 2)))
    scores = np.column_stack(parts)
    singular, rank = fusion_spectrum(scores)
    saved.update(fusion_singular_values=singular, fusion_rank=np.asarray(rank))
    labels, strength = (clusterer or fit_density_scores)(scores, config.density_config())
    diagnostics = {
        "benchmark_control": PCA4,
        "fusion_rank": rank,
        "completed_updates": 0,
        "per_view_standardized_mse": errors,
        "branch_parameter_counts": [0, 0],
    }
    return native.NeuralPoint(
        PCAViewsModel(config, saved),
        scores,
        labels,
        strength,
        np.repeat(np.mean(errors), 2),
        diagnostics,
    )


def render_control(out, summary, point, units, **kwargs):
    from super_deli.workflows._neural_common import _report

    summary.update(
        n_coordinates=4,
        benchmark_control=point.diagnostics["benchmark_control"],
        fusion_rank=point.diagnostics["fusion_rank"],
        branch_parameter_counts=point.diagnostics["branch_parameter_counts"],
    )
    proxy = native.NeuralPoint(
        point.model,
        point.scores[:, :2],
        point.labels,
        point.strength,
        point.losses,
        point.diagnostics,
    )
    _report(
        out,
        summary,
        proxy,
        units,
        title="Benchmark-only four-coordinate control",
        view_details=(
            "<p>Clustering uses all four coordinates. This display shows the first two "
            "only. This control is not a public model adapter.</p>"
        ),
    )


def run_control(dataset, *, model_config, base_dir, output_dir):
    from super_deli.models.paired_subsampling import fit_paired_subsamples
    from super_deli.workflows._paired_neural_common import run_paired_neural

    config = ControlConfig.from_dict(model_config.to_dict())
    joint = config.model_family == JOINT4
    training = (
        {
            **TRAINING,
            "encoder_widths": [16, 8, 4],
            "latent_dimensions": 4,
            "benchmark_control": JOINT4,
        }
        if joint
        else {
            "method": (
                "per-view standardized covariance PCA; two PCs/view; concatenation; no joint PCA"
            ),
            "updates": 0,
            "benchmark_control": PCA4,
        }
    )

    def fit(tables, *, config):
        return fit_paired_subsamples(
            tables, config=config, point_function=fit_joint_four if joint else fit_pca_four
        )

    return run_paired_neural(
        dataset,
        model_config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        config_type=ControlConfig,
        fit_function=fit,
        training=training,
        render=render_control,
    )
