"""Dependency-free opt-in reference-support contract; no species inference."""

from dataclasses import asdict, dataclass, fields

POLICIES = ("distance", "probability")
DECISIONS = (
    "assigned",
    "withheld_outside_reference_support",
    "withheld_ambiguous_reference_support",
    "withheld_classifier_support_conflict",
)
SCOPE = "Reference compatibility only; abstention does not establish novelty or species"


@dataclass(frozen=True)
class ReferenceAbstentionConfig:
    assume_independent_calibration: bool
    policy: str = "distance"
    schema_version: int = 1
    exclusion_level: float = 0.05
    covariance_shrinkage: float = 0.1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported reference-abstention schema")
        if (
            type(self.assume_independent_calibration) is not bool
            or not self.assume_independent_calibration
        ):
            raise ValueError("Independent calibration specimens must be explicitly acknowledged")
        if self.policy not in POLICIES:
            raise ValueError("Unsupported reference-support policy")
        if (
            type(self.exclusion_level) not in (int, float)
            or self.exclusion_level != 0.05
            or type(self.covariance_shrinkage) not in (int, float)
            or self.covariance_shrinkage != 0.1
        ):
            raise ValueError("This version fixes exclusion .05 and covariance shrinkage .1")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or "assume_independent_calibration" not in value
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Missing or unknown reference-abstention configuration fields")
        return cls(**value)
