"""Dependency-free contract for two encoders and fixed equal-view fusion."""

from dataclasses import asdict, dataclass, fields

from super_deli.models.matched_neural_config import MatchedNeuralConfig

FAMILY = "matched_separate_autoencoders_hdbscan"


@dataclass(frozen=True)
class SeparateNeuralConfig:
    morphology: str
    ecology: str
    morphology_features: tuple[str, ...]
    ecology_features: tuple[str, ...]
    assume_independent_units: bool
    ecology_measurement_scope: str
    schema_version: int = 1
    model_family: str = FAMILY
    random_seed: int = 0
    subsampling_seed: int = 0
    n_subsamples: int = 0
    max_working_bytes: int = 268435456
    max_pairwise_bytes: int = 268435456

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.model_family != FAMILY
        ):
            raise ValueError("Unsupported separate-encoder schema/family")
        # Identity, trait, independence, seed and resource rules have one owner.
        MatchedNeuralConfig(
            **{**self.to_dict(), "model_family": "matched_morphology_ecology_autoencoder_hdbscan"}
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        required = {
            "schema_version",
            "morphology",
            "ecology",
            "morphology_features",
            "ecology_features",
            "assume_independent_units",
            "ecology_measurement_scope",
        }
        if (
            not isinstance(value, dict)
            or required - value.keys()
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Missing or unknown separate-encoder fields")
        options = dict(value)
        for name in ("morphology_features", "ecology_features"):
            if not isinstance(options[name], (list, tuple)):
                raise ValueError("Feature lists must be arrays")
            options[name] = tuple(options[name])
        return cls(**options)

    def density_config(self):
        from super_deli.models.hdbscan_config import HDBSCANConfig

        return HDBSCANConfig(
            True,
            feature_names=("morphology_1", "morphology_2", "ecology_1", "ecology_2"),
            n_subsamples=self.n_subsamples,
            random_seed=self.subsampling_seed,
            max_working_bytes=self.max_working_bytes,
            max_pairwise_bytes=self.max_pairwise_bytes,
        )
