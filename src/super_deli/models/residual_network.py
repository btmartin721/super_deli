"""Fixed supervised CPU residual network and pickle-free NumPy reference prediction."""

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.io.reference_labels import validate_reference_labels
from super_deli.models.matched_preprocessing import align_tables, prepare_views
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.residual_config import TRAINING, ResidualConfig
from super_deli.models.tabular_neural import NeuralFitError, _torch

LAYERS = ("input", "block_0_first", "block_0_second", "block_1_first", "block_1_second", "output")


def initial_weights(dimensions, classes, seed):
    rng, arrays = np.random.default_rng(seed), {}
    for name in LAYERS:
        incoming, outgoing = (
            (dimensions, 32) if name == "input" else (32, classes) if name == "output" else (32, 32)
        )
        scale = np.sqrt((1 if name == "output" else 2) / incoming)
        arrays[name + "_weight"] = rng.normal(0, scale, (outgoing, incoming))
        arrays[name + "_bias"] = np.zeros(outgoing)
    return arrays


def forward_numpy(x, arrays):
    def linear(z, name):
        return z @ arrays[name + "_weight"].T + arrays[name + "_bias"]

    h = np.maximum(linear(x, "input"), 0)
    for block in range(2):
        inner = np.maximum(linear(h, f"block_{block}_first"), 0)
        h = np.maximum(h + linear(inner, f"block_{block}_second"), 0)
    return linear(h, "output")


def probabilities(logits):
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("Finite reference logits required")
    shifted = logits - logits.max(1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(1, keepdims=True)


def named_parameters(network):
    return [
        (name + "_" + kind, getattr(network.layers[name], kind))
        for name in LAYERS
        for kind in ("weight", "bias")
    ]


def network_arrays(network):
    return {name: p.detach().numpy().copy() for name, p in named_parameters(network)}


def _network(initial):
    torch = _torch()

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleDict()
            for name in LAYERS:
                shape = initial[name + "_weight"].shape
                layer = torch.nn.Linear(shape[1], shape[0], dtype=torch.float64, device="cpu")
                with torch.no_grad():
                    layer.weight.copy_(torch.from_numpy(initial[name + "_weight"]))
                    layer.bias.copy_(torch.from_numpy(initial[name + "_bias"]))
                self.layers[name] = layer

        def forward(self, x):
            h = torch.relu(self.layers["input"](x))
            for block in range(2):
                h = torch.relu(
                    h
                    + self.layers[f"block_{block}_second"](
                        torch.relu(self.layers[f"block_{block}_first"](h))
                    )
                )
            return self.layers["output"](h)

    return Network()


def train_residual(x, y, seed, *, observer=None):
    torch = _torch()
    previous = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    initial = initial_weights(x.shape[1], int(y.max()) + 1, seed)
    network, losses, penalties = None, [], []
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            network = _network(initial)
            tensor = torch.tensor(x, dtype=torch.float64, device="cpu")
            target = torch.tensor(y, dtype=torch.long, device="cpu")
            optimizer = torch.optim.Adam(
                network.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0001,
                foreach=False,
                fused=False,
            )

            def record():
                with torch.no_grad():
                    losses.append(float(torch.nn.functional.cross_entropy(network(tensor), target)))
                    penalties.append(
                        float(0.00005 * sum(torch.sum(p * p) for p in network.parameters()))
                    )
                if not np.isfinite([losses[-1], penalties[-1]]).all():
                    raise NeuralFitError("Nonfinite reference training objective")

            record()
            for _ in range(500):
                optimizer.zero_grad(set_to_none=True)
                objective = torch.nn.functional.cross_entropy(network(tensor), target)
                objective.backward()
                if any(
                    p.grad is None or not bool(torch.isfinite(p.grad).all())
                    for p in network.parameters()
                ):
                    raise NeuralFitError("Nonfinite reference gradient")
                if observer is not None:
                    observer.pre(network, optimizer, x, y)
                optimizer.step()
                if observer is not None:
                    observer.post(network, optimizer, x, y)
                record()
            arrays = network_arrays(network)
    except Exception as exc:
        raise NeuralFitError(
            str(exc),
            arrays=network_arrays(network) if network is not None else {},
            diagnostics={"completed_updates": max(0, len(losses) - 1), "losses": losses},
        ) from exc
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])
    return arrays, initial, np.array(losses), np.array(penalties)


@dataclass(frozen=True)
class ResidualReferenceModel:
    config: ResidualConfig
    classes: tuple[str, ...]
    arrays: dict

    def predict_matrix(self, x):
        a = self.arrays
        if x.ndim != 2 or x.shape[1] != len(a["means"]) or not len(x) or not np.isfinite(x).all():
            raise ValueError("Complete finite prediction rows in fitted feature order required")
        if 160 * x.size + 2048 * len(x) > self.config.max_working_bytes:
            raise ValueError("Reference prediction workspace exceeds declared budget")
        kept = a["kept_indices"]
        weighted = (x[:, kept] - a["means"][kept]) / a["scales"][kept] * a["column_weights"]
        with threadpool_limits(limits=1):
            logits = forward_numpy(weighted, a)
            p = probabilities(logits)
        return {
            "logits": logits,
            "probabilities": p,
            "predicted_reference_labels": [self.classes[j] for j in p.argmax(1)],
            "scope": TRAINING["scope"],
        }

    def predict(self, tables):
        x, alignment = align_tables(tables, self.config, minimum=1)
        return {"unit_ids": alignment["unit_ids"], **self.predict_matrix(x)}


def fit_residual_point(x, y, *, config, observer=None):
    config = ResidualConfig.from_dict(config.to_dict())
    y = np.asarray(y)
    if (
        x.ndim != 2
        or len(x) < 10
        or x.shape[1] != len(config.morphology_features) + len(config.ecology_features)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Complete matched reference training matrix required")
    if (
        y.shape != (len(x),)
        or y.dtype.kind not in "iu"
        or not np.array_equal(np.unique(y), np.arange(int(y.max()) + 1))
        or not 2 <= len(np.unique(y)) <= 32
        or np.bincount(y).min() < 2
    ):
        raise ValueError("Contiguous encoded classes with two observations per class required")
    if 160 * x.size + 32768 * len(x) > config.max_working_bytes:
        raise ValueError("Reference training workspace exceeds declared budget")
    weighted, prep = prepare_views(x, len(config.morphology_features))
    with FIT_LOCK, threadpool_limits(limits=1):
        fitted, initial, losses, penalties = train_residual(
            weighted, y, config.random_seed, observer=observer
        )
    arrays = {
        **prep,
        **fitted,
        **{"initial_" + k: v for k, v in initial.items()},
        "losses": losses,
        "regularization": penalties,
    }
    diagnostics = {
        "completed_updates": 500,
        "training": TRAINING,
        "parameter_count": sum(v.size for v in fitted.values()),
        "final_data_loss": float(losses[-1]),
        "final_regularization": float(penalties[-1]),
        "convergence": "Fixed schedule completed; convergence not established",
        "novelty": "unassessed",
        "species_significance": "not_computed",
    }
    return arrays, diagnostics


def fit_residual_reference(tables, labels, *, config, observer=None):
    x, alignment = align_tables(tables, config)
    classes, y, counts = validate_reference_labels(alignment["unit_ids"], labels)
    arrays, diagnostics = fit_residual_point(x, np.asarray(y), config=config, observer=observer)
    diagnostics["reference_class_counts"] = counts
    return ResidualReferenceModel(config, classes, arrays), diagnostics, alignment
