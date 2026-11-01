"""Fixed CPU autoencoder with pickle-free frozen transforms and density groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.models.density_scores import fit_density_scores
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.neural_config import TRAINING, NeuralConfig
from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples
from super_deli.models.tabular_hdbscan import prepare_density_table


class NeuralFitError(ValueError):
    """Retain partial training evidence instead of replacing a failed fit."""

    def __init__(self, message, *, diagnostics=None, arrays=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}
        self.arrays = arrays or {}


def _torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError("Neural clustering requires: pip install 'super-deli[neural]'") from exc
    return torch


def _network(d):
    torch = _torch()

    def layers(widths):
        result = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            result.append(torch.nn.Linear(a, b, dtype=torch.float64, device="cpu"))
            if i < len(widths) - 2:
                result.append(torch.nn.Tanh())
        return torch.nn.Sequential(*result)

    return torch.nn.ModuleDict({"encoder": layers([d, 16, 8, 2]), "decoder": layers([2, 8, 16, d])})


def _weights(network):
    return {
        f"{side}_{i}_{kind}": getattr(network[side][2 * i], kind).detach().numpy().copy()
        for side in ("encoder", "decoder")
        for i in range(3)
        for kind in ("weight", "bias")
    }


def forward_numpy(x, weights, side):
    """Evaluate the saved network without requiring PyTorch."""
    y = np.asarray(x, dtype=float)
    for i in range(3):
        y = y @ weights[f"{side}_{i}_weight"].T + weights[f"{side}_{i}_bias"]
        if i < 2:
            y = np.tanh(y)
    return y


def _train(x, seed):
    torch = _torch()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    losses, network = [], None
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            # Seed only the CPU generator; do not modify CUDA generators.
            torch.random.default_generator.manual_seed(seed)
            network = _network(x.shape[1])
            tensor = torch.tensor(x, dtype=torch.float64, device="cpu")
            optimizer = torch.optim.Adam(
                network.parameters(), lr=0.001, weight_decay=0.0001, foreach=False, fused=False
            )

            def loss():
                return torch.mean((network["decoder"](network["encoder"](tensor)) - tensor) ** 2)

            with torch.no_grad():
                losses.append(float(loss()))
            for _ in range(500):
                optimizer.zero_grad(set_to_none=True)
                objective = loss()
                if not bool(torch.isfinite(objective)):
                    raise NeuralFitError("Nonfinite neural training loss")
                objective.backward()
                if any(not bool(torch.isfinite(p.grad).all()) for p in network.parameters()):
                    raise NeuralFitError("Nonfinite neural gradient")
                optimizer.step()
                with torch.no_grad():
                    value = float(loss())
                if not np.isfinite(value):
                    raise NeuralFitError("Nonfinite neural update")
                losses.append(value)
            arrays = _weights(network)
    except Exception as exc:
        arrays = _weights(network) if network is not None else {}
        raise NeuralFitError(
            str(exc),
            diagnostics={
                "seed": seed,
                "completed_updates": max(0, len(losses) - 1),
                "losses": losses,
            },
            arrays=arrays,
        ) from exc
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn)
    return arrays, np.asarray(losses)


def latent_transform(z):
    if not np.isfinite(z).all():
        raise NeuralFitError("Nonfinite neural embedding")
    means, scales = z.mean(axis=0), z.std(axis=0)
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise NeuralFitError("Collapsed neural embedding")
    scores = (z - means) / scales
    singular = np.linalg.svd(scores, compute_uv=False)
    if singular[-1] <= 1e-12 * singular[0]:
        raise NeuralFitError("Rank-deficient neural embedding")
    return scores, means, scales, singular


@dataclass(frozen=True)
class NeuralModel:
    feature_names: tuple[str, ...]
    arrays: dict[str, np.ndarray]

    def _raw(self, values):
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError("Projection requires complete rows in the declared feature order")
        a = self.arrays
        ids = a["kept_indices"]
        return (x[:, ids] - a["means"][ids]) / a["scales"][ids]

    def encode(self, values):
        a = self.arrays
        result = (forward_numpy(self._raw(values), a, "encoder") - a["latent_means"]) / a[
            "latent_scales"
        ]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite frozen encoding")
        return result

    def reconstruct(self, values):
        raw = self._raw(values)
        a = self.arrays
        y = forward_numpy(forward_numpy(raw, a, "encoder"), a, "decoder")
        result = np.broadcast_to(a["means"], (len(raw), len(self.feature_names))).copy()
        ids = a["kept_indices"]
        result[:, ids] = y * a["scales"][ids] + a["means"][ids]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite frozen reconstruction")
        return result


@dataclass(frozen=True)
class NeuralPoint:
    model: NeuralModel
    scores: np.ndarray
    labels: np.ndarray
    strength: np.ndarray
    losses: np.ndarray
    diagnostics: dict

    @property
    def selected_k(self):
        return len(set(self.labels) - {-1})


def fit_neural_point(x, names, *, config):
    """Fit exactly one seed. Truth, labels and seed searches are not inputs."""
    x = np.asarray(x, dtype=float)
    if (
        x.ndim != 2
        or len(x) < 10
        or x.shape[1] != len(names)
        or tuple(names) != config.feature_names
        or not np.isfinite(x).all()
    ):
        raise ValueError("Neural fitting requires >=10 complete rows in declared feature order")
    if 64 * len(x) ** 2 + 160 * x.size > config.max_working_bytes:
        raise ValueError("Neural distance/training workspace exceeds budget")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        means, scales = x.mean(axis=0), x.std(axis=0)
        kept = np.flatnonzero(scales > 0)
        if not 3 <= len(kept) <= 64:
            raise ValueError("Neural fitting requires 3–64 nonconstant features")
        scales[scales == 0] = 1.0
        standardized = (x[:, kept] - means[kept]) / scales[kept]
    with FIT_LOCK, threadpool_limits(limits=1):
        weights, losses = _train(standardized, config.random_seed)
        try:
            raw = forward_numpy(standardized, weights, "encoder")
            scores, latent_means, latent_scales, singular = latent_transform(raw)
            labels, strength = fit_density_scores(scores, config.density_config())
        except Exception as exc:
            raise NeuralFitError(
                str(exc),
                diagnostics={
                    "losses": losses.tolist(),
                    "seed": config.random_seed,
                    "completed_updates": 500,
                },
                arrays=weights,
            ) from exc
    arrays = {
        **weights,
        "means": means,
        "scales": scales,
        "kept_indices": kept,
        "latent_means": latent_means,
        "latent_scales": latent_scales,
    }
    return NeuralPoint(
        NeuralModel(tuple(names), arrays),
        scores,
        labels,
        strength,
        losses,
        {
            "training": TRAINING,
            "seed": config.random_seed,
            "latent_singular_values": singular.tolist(),
            "excluded_constant_features": [names[i] for i in range(len(names)) if i not in kept],
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "species_significance": "not_computed",
        },
    )


def point_arrays(point):
    return {
        **point.model.arrays,
        "scores": point.scores,
        "labels": point.labels,
        "strength": point.strength,
        "losses": point.losses,
    }


def fit_tabular_neural(table, *, config):
    config = NeuralConfig.from_dict(config.to_dict())
    x, names, _ = prepare_density_table(table, config.density_config())
    point = fit_neural_point(x, names, config=config)
    records, arrays = [], {}
    for i, ids in enumerate(
        subsample_indices(len(x), config.n_subsamples, 0.8, config.subsampling_seed)
        if config.n_subsamples
        else ()
    ):
        record = {
            "draw": i,
            "training_indices": ids.tolist(),
            "unit_ids": [table.unit_ids[j] for j in ids],
            "seed": config.random_seed,
        }
        try:
            draw = fit_neural_point(x[ids], names, config=config)
            arrays.update({f"draw_{i}_{k}": v for k, v in point_arrays(draw).items()})
            record.update(
                status="complete",
                labels=draw.labels.tolist(),
                selected_k=draw.selected_k,
                diagnostics=draw.diagnostics,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            record.update(
                status="failed", error=str(exc), diagnostics=getattr(exc, "diagnostics", {})
            )
            arrays.update(
                {f"draw_{i}_failed_{k}": v for k, v in getattr(exc, "arrays", {}).items()}
            )
        records.append(record)
    stability = summarize_subsamples(len(x), records)
    stability.arrays.update(arrays)
    return point, stability
