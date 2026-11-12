"""Dependency-free, separately versioned supervised reference-assignment contract."""

from dataclasses import asdict, dataclass, fields

from super_deli.models.matched_neural_config import MatchedNeuralConfig

FAMILY = "matched_residual_reference_classifier"
TRAINING = {
    "width": 32,
    "residual_blocks": 2,
    "activation": "relu",
    "updates": 500,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "coupled_weight_decay": 0.0001,
    "loss": "mean_multiclass_cross_entropy",
    "dtype": "float64",
    "device": "cpu",
    "scope": "Reference classes only; novelty is unassessed",
}


@dataclass(frozen=True)
class ResidualConfig:
    morphology: str
    ecology: str
    morphology_features: tuple[str, ...]
    ecology_features: tuple[str, ...]
    assume_independent_units: bool
    ecology_measurement_scope: str
    schema_version: int = 1
    model_family: str = FAMILY
    random_seed: int = 0
    max_working_bytes: int = 268435456

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.model_family != FAMILY
        ):
            raise ValueError("Unsupported residual reference schema/family")
        options = self.to_dict()
        options["model_family"] = "matched_morphology_ecology_autoencoder_hdbscan"
        MatchedNeuralConfig(**options)
        if {"reference_label", "predicted_reference_label"} & set(
            self.morphology_features + self.ecology_features
        ):
            raise ValueError("Reference labels cannot be analytical features")

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
            raise ValueError("Missing or unknown residual configuration fields")
        options = dict(value)
        for name in ("morphology_features", "ecology_features"):
            if not isinstance(options[name], (list, tuple)):
                raise ValueError("Feature lists must be arrays")
            options[name] = tuple(options[name])
        return cls(**options)
