"""Explicit resampling settings outside the immutable point-model configuration."""

from dataclasses import asdict, dataclass

from super_deli.models.empirical_config import _options
from super_deli.models.predictive_config import PredictiveScreenConfig


@dataclass(frozen=True)
class PredictiveBootstrapConfig:
    n_bootstrap: int
    random_seed: int
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported predictive-bootstrap schema")
        if type(self.n_bootstrap) is not int or self.n_bootstrap < 1:
            raise ValueError("Explicit positive bootstrap count required")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**64:
            raise ValueError("Unsigned 64-bit bootstrap seed required")

    @staticmethod
    def validate_model(model, screening):
        PredictiveScreenConfig.validate_model(model)
        if (
            model.covariance_type != "tied"
            or model.target_level != "locality"
            or screening.n_folds != 5
            or {v.kind for v in model.views} != {"morphological", "ecological"}
            or len(model.views) != 2
            or any(v.missing_policy != "error" for v in model.views)
            or any(v.pca_fit != "target_means" for v in model.views if v.kind == "morphological")
        ):
            raise ValueError(
                "Bootstrap requires shared covariance, five fixed locality folds, "
                "complete morphology/ecology and target-mean morphology PCA"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**_options(cls, value, {"schema_version", "n_bootstrap", "random_seed"}))

    def draw_seeds(self):
        from super_deli.benchmarking.config import derived_seed

        return [
            derived_seed(self.random_seed, "predictive_locality_bootstrap_v1", i, "draw")
            for i in range(self.n_bootstrap)
        ]
