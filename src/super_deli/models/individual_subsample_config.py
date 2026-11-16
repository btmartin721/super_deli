"""Separate opt-in sampling settings; legacy model serialization stays unchanged."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class IndividualSubsampleConfig:
    n_subsamples: int
    random_seed: int
    fraction: float = 0.8
    schema_version: int = 1
    policy: str = "whole_individual_80_v1"

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Use subsampling schema1")
        if self.policy != "whole_individual_80_v1":
            raise ValueError("Unsupported individual subsampling policy")
        if type(self.fraction) is not float or self.fraction != 0.8:
            raise ValueError("This workflow requires exactly 80% without replacement")
        if type(self.n_subsamples) is not int or not 1 <= self.n_subsamples <= 1000:
            raise ValueError("n_subsamples must be an integer from 1 to 1000")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("Require an explicit uint32 random seed")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Supply the exact subsampling configuration fields")
        return cls(**value)


def require_supported_point(model):
    if (
        model.fusion_policy != "retain_all_v1"
        or model.target_level != "specimen"
        or model.covariance_type != "tied"
        or model.selection_criterion != "bic"
        or model.initialization_policy != "legacy"
        or model.n_init != 100
        or model.n_threads != 1
    ):
        raise ValueError(
            "Require a matched full-retention/shared-BIC point run "
            "with ordinary 100 starts and one thread"
        )
