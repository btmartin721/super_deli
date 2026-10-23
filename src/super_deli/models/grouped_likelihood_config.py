"""Versioned opt-in likelihood calibration; old grouped declarations stay unchanged."""

from dataclasses import dataclass, fields, replace

from super_deli.models.empirical_config import _options
from super_deli.models.grouped_config import (
    BOUNDARY_CASES,
    BoundarySolverConfig,
    GroupedCalibrationConfig,
    GroupedStudyConfig,
)
from super_deli.models.grouped_config import (
    CASES as GAUSSIAN_CASES,
)
from super_deli.models.integrative_config import IntegrativeConfig

CASES = GAUSSIAN_CASES + BOUNDARY_CASES + ("unequal_variance", "cline", "complementary_256")


@dataclass(frozen=True)
class GroupedLikelihoodConfig(GroupedCalibrationConfig):
    statistic: str = "max_eligible_likelihood_gain_v1"

    def __post_init__(self):
        if not isinstance(self.model, IntegrativeConfig):
            raise ValueError("A valid integrative model is required")
        # Validate the identical raw-null policy through the existing validator.
        options = {f.name: getattr(self, f.name) for f in fields(GroupedCalibrationConfig)}
        options["model"] = replace(self.model, selection_criterion="bic")
        GroupedCalibrationConfig(**options)
        if (
            self.statistic != "max_eligible_likelihood_gain_v1"
            or self.schema_version != 2
            or self.model.selection_criterion != "aic"
            or self.model.covariance_type != "full"
            or not any(v.pca_fit == "target_means" for v in self.model.views)
        ):
            raise ValueError("Use schema-2 full-covariance AIC/group-mean likelihood calibration")

    @classmethod
    def from_dict(cls, value):
        options = _options(
            cls,
            value,
            {
                "schema_version",
                "model",
                "statistic",
                "assume_gaussian_independent_groups",
                "assume_noninformative_counts",
            },
        )
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        options["solver"] = BoundarySolverConfig.from_dict(options["solver"])
        return cls(**options)


@dataclass(frozen=True)
class GroupedLikelihoodStudyConfig(GroupedStudyConfig):
    large_n: int = 256
    random_seed: int = 20261114

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.calibration, GroupedLikelihoodConfig):
            raise ValueError("A grouped likelihood calibration policy is required")
        if type(self.large_n) is not int or self.large_n <= self.n_units or self.large_n % 4:
            raise ValueError("large_n must exceed n_units and be divisible by four")
        if 512 * self.large_n * 64 > self.calibration.model.max_working_bytes:
            raise ValueError("Large cohort exceeds workspace budget")

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "calibration"})
        options["calibration"] = GroupedLikelihoodConfig.from_dict(options["calibration"])
        return cls(**options)
