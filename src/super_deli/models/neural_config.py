"""Dependency-free, versioned contract for the first tabular neural workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class NeuralConfig:
    feature_names: tuple[str, ...]
    assume_independent_units: bool
    schema_version: int = 1
    model_family: str = "tabular_autoencoder_hdbscan"
    random_seed: int = 0
    subsampling_seed: int = 0
    n_subsamples: int = 0
    max_working_bytes: int = 268435456
    max_pairwise_bytes: int = 268435456

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported neural schema version")
        if self.model_family != "tabular_autoencoder_hdbscan":
            raise ValueError("Unsupported neural model family")
        if self.assume_independent_units is not True:
            raise ValueError("Explicitly assume independent biological units")
        if (
            not isinstance(self.feature_names, tuple)
            or not 3 <= len(self.feature_names) <= 64
            or any(not isinstance(n, str) or not n or n != n.strip() for n in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("Select 3–64 unique analytical feature names")
        for name in ("random_seed", "subsampling_seed", "n_subsamples"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be a nonnegative 32-bit integer")
        for name in ("max_working_bytes", "max_pairwise_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    def density_config(self):
        from super_deli.models.hdbscan_config import HDBSCANConfig

        return HDBSCANConfig(
            assume_independent_units=True,
            feature_names=self.feature_names,
            n_subsamples=self.n_subsamples,
            random_seed=self.subsampling_seed,
            max_working_bytes=self.max_working_bytes,
            max_pairwise_bytes=self.max_pairwise_bytes,
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        required = {"schema_version", "feature_names", "assume_independent_units"}
        if (
            not isinstance(value, dict)
            or required - value.keys()
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Missing or unknown neural configuration fields")
        value = dict(value)
        if not isinstance(value["feature_names"], (tuple, list)):
            raise ValueError("feature_names must be an array")
        value["feature_names"] = tuple(value["feature_names"])
        return cls(**value)


TRAINING = {
    "encoder_widths": [16, 8, 2],
    "decoder_hidden_widths": [8, 16],
    "hidden_activation": "tanh",
    "bottleneck_and_output": "linear",
    "dtype": "float64",
    "device": "cpu",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "updates": 500,
    "batch": "all fit units",
    "latent_scaling": "fit-unit mean and population SD",
    "fit_scope": "transductive unsupervised; no held-out performance claim",
}
