"""Opt-in cross-validated screening; deliberately separate from legacy model configs."""

from dataclasses import asdict, dataclass, fields

from super_deli.models.integrative_config import IntegrativeConfig


@dataclass(frozen=True)
class PredictiveScreenConfig:
    n_folds: int = 5
    random_seed: int = 71324
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported predictive-screen schema")
        if type(self.n_folds) is not int or not 2 <= self.n_folds <= 10:
            raise ValueError("Use two to ten whole-unit folds")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**64:
            raise ValueError("Use an unsigned 64-bit split seed")

    @staticmethod
    def validate_model(model):
        if not isinstance(model, IntegrativeConfig):
            raise ValueError("An explicit integrative model is required")
        if (
            model.selection_criterion != "aic"
            or model.covariance_type not in {"full", "tied"}
            or model.n_bootstrap
            or model.n_threads != 1
            or model.missing_views != "error"
        ):
            raise ValueError(
                "Screening requires single-thread grouped AIC/full or tied covariance "
                "without stability"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Only declared predictive-screen fields are supported")
        if "schema_version" not in value:
            raise ValueError("Predictive screening requires schema_version")
        return cls(**value)
