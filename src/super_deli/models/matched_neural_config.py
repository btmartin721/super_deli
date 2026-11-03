"""Separate, dependency-free contract for paired individual neural integration."""

import math
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
    view_loss_weights: tuple[float, float] | None = None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ValueError("Unsupported matched neural schema")
        if self.schema_version == 1:
            if self.view_loss_weights is not None:
                raise ValueError("Explicit view weights require matched neural schema 2")
        elif (
            not isinstance(self.view_loss_weights, tuple)
            or len(self.view_loss_weights) != 2
            or any(
                type(w) not in (int, float) or not math.isfinite(w) or not 0 < w < 1
                for w in self.view_loss_weights
            )
            or not math.isclose(math.fsum(self.view_loss_weights), 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("Declare two positive finite view-loss fractions summing to one")
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
        result = asdict(self)
        if self.schema_version == 1:
            result.pop("view_loss_weights")
        return result

    def loss_fractions(self):
        if self.schema_version == 1:
            return (0.5, 0.5)
        total = math.fsum(self.view_loss_weights)
        return tuple(w / total for w in self.view_loss_weights)

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
        if options["schema_version"] == 1 and "view_loss_weights" in options:
            raise ValueError("Schema-1 JSON has no view-weight field")
        if options["schema_version"] == 2:
            if not isinstance(options.get("view_loss_weights"), (list, tuple)):
                raise ValueError("Schema 2 requires explicit view-loss fractions")
            options["view_loss_weights"] = tuple(options["view_loss_weights"])
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
