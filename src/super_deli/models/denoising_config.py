"""Dependency-free contract for fixed-corruption tabular denoising."""

from dataclasses import asdict, dataclass, fields

from super_deli.models.neural_config import TRAINING, NeuralConfig

DENOISING_TRAINING = {
    **TRAINING,
    "objective": "reconstruct observed standardized targets from corrupted inputs",
    "corruption": "fresh independent additive Gaussian per update",
    "corruption_sd": 0.15,
    "corruption_units": "fit-feature population SD",
    "inference_corruption": False,
    "adam_betas": [0.9, 0.999],
    "adam_epsilon": 1e-8,
    "foreach": False,
    "fused": False,
}


@dataclass(frozen=True)
class DenoisingConfig:
    feature_names: tuple[str, ...]
    assume_independent_units: bool
    schema_version: int = 1
    model_family: str = "tabular_denoising_autoencoder_hdbscan"
    random_seed: int = 0
    corruption_seed: int = 0
    subsampling_seed: int = 0
    n_subsamples: int = 0
    max_working_bytes: int = 268435456
    max_pairwise_bytes: int = 268435456

    def __post_init__(self):
        if self.model_family != "tabular_denoising_autoencoder_hdbscan":
            raise ValueError("Unsupported denoising model family")
        if type(self.corruption_seed) is not int or not 0 <= self.corruption_seed < 2**32:
            raise ValueError("corruption_seed must be a nonnegative 32-bit integer")
        self.ordinary_config()

    def ordinary_config(self):
        return NeuralConfig(
            **{
                name: getattr(self, name)
                for name in (
                    "feature_names",
                    "assume_independent_units",
                    "schema_version",
                    "random_seed",
                    "subsampling_seed",
                    "n_subsamples",
                    "max_working_bytes",
                    "max_pairwise_bytes",
                )
            }
        )

    def density_config(self):
        return self.ordinary_config().density_config()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        required = {"schema_version", "feature_names", "assume_independent_units", "model_family"}
        if (
            not isinstance(value, dict)
            or required - value.keys()
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Missing or unknown denoising configuration fields")
        value = dict(value)
        if not isinstance(value["feature_names"], (list, tuple)):
            raise ValueError("feature_names must be an array")
        value["feature_names"] = tuple(value["feature_names"])
        return cls(**value)
