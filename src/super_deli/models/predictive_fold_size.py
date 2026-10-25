"""Explicit fold exposure policy; independent of features, candidate labels and outcomes."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PredictiveFoldSizePolicy:
    policy: str = "proportional_distinct"
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported fold-size policy schema")
        if self.policy != "proportional_distinct":
            raise ValueError("Prespecify the proportional-distinct fold-size policy")

    def validate_model(self, model, screening):
        from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig

        PredictiveBootstrapConfig.validate_model(model, screening)
        if model.min_cluster_size < 2:
            raise ValueError("Scaled fold sizes require a full-data minimum of at least two")

    def evidence(self, minimum, full_ids, training_ids):
        full, training = set(full_ids), set(training_ids)
        if type(minimum) is not int or minimum < 2 or not training or not training < full:
            raise ValueError("Nonempty proper training support and full minimum >=2 required")
        u, t = len(full), len(training)
        return dict(
            schema_version=1,
            policy=self.policy,
            full_distinct=u,
            training_distinct=t,
            full_minimum=minimum,
            training_minimum=max(2, (minimum * t + u - 1) // u),
            conditioning="Sampling IDs only; copies do not add independent localities",
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"schema_version", "policy"}:
            raise ValueError("Explicit versioned fold-size policy required")
        return cls(**value)


@dataclass(frozen=True)
class PredictiveFoldMixture:
    """Effective shared-mixture settings, separate from the legacy model schema."""

    max_k: int
    min_cluster_size: int
    n_init: int
    max_iter: int
    tol: float
    reg_covar: float
    selection_criterion: str
    max_working_bytes: int

    @classmethod
    def from_model(cls, model, minimum):
        from super_deli.models.predictive_config import PredictiveScreenConfig

        PredictiveFoldSizePolicy().validate_model(model, PredictiveScreenConfig())
        if type(minimum) is not int or not 2 <= minimum <= model.min_cluster_size:
            raise ValueError("Effective fold minimum must be between two and the full minimum")
        names = (
            "max_k",
            "n_init",
            "max_iter",
            "tol",
            "reg_covar",
            "selection_criterion",
            "max_working_bytes",
        )
        return cls(min_cluster_size=minimum, **{name: getattr(model, name) for name in names})

    def to_dict(self):
        return dict(schema_version=1, workflow="predictive_fold_mixture_v1", **asdict(self))
