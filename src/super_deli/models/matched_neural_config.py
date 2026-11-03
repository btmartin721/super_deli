"""Separate, dependency-free contract for paired individual neural integration."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class MatchedNeuralConfig:
    morphology: str
    ecology: str
    morphology_features: tuple[str, ...]
    ecology_features: tuple[str, ...]
    assume_independent_units: bool
    ecology_measurement_scope: str
    schema_version: int = 1
    model_family: str = "matched_morphology_ecology_autoencoder_hdbscan"
    random_seed: int = 0
    subsampling_seed: int = 0
    n_subsamples: int = 0
    max_working_bytes: int = 268435456
    max_pairwise_bytes: int = 268435456

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported matched neural schema")
        if self.model_family != "matched_morphology_ecology_autoencoder_hdbscan":
            raise ValueError("Unsupported matched neural model family")
        if (
            self.assume_independent_units is not True
            or self.ecology_measurement_scope != "individual"
        ):
            raise ValueError("Declare independent individuals and individually measured ecology")
        for name in ("morphology", "ecology"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("Declare nonempty, unpadded modality names")
        if self.morphology == self.ecology:
            raise ValueError("Two distinct selected modalities required")
        for name in ("morphology_features", "ecology_features"):
            names = getattr(self, name)
            if (
                not isinstance(names, tuple)
                or not 3 <= len(names) <= 32
                or any(not isinstance(n, str) or not n or n != n.strip() for n in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError("Declare 3–32 unique, unpadded feature names per view")
        for name in ("random_seed", "subsampling_seed", "n_subsamples"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**32:
                raise ValueError(f"{name} must be a nonnegative 32-bit integer")
        for name in ("max_working_bytes", "max_pairwise_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

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
            raise ValueError("Missing or unknown matched neural configuration fields")
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
            feature_names=("latent_1", "latent_2"),
            n_subsamples=self.n_subsamples,
            random_seed=self.subsampling_seed,
            max_working_bytes=self.max_working_bytes,
            max_pairwise_bytes=self.max_pairwise_bytes,
        )
