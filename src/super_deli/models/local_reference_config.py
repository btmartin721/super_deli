"""Separately versioned class-local support; old abstention schemas stay unchanged."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class LocalReferenceSupportConfig:
    assume_independent_calibration: bool
    schema_version: int = 1
    neighbors: int = 5
    exclusion_level: float = 0.05
    covariance_shrinkage: float = 0.1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported class-local support schema")
        if (
            type(self.assume_independent_calibration) is not bool
            or not self.assume_independent_calibration
        ):
            raise ValueError("Independent calibration specimens must be explicitly acknowledged")
        if type(self.neighbors) is not int or self.neighbors != 5:
            raise ValueError("This version fixes five original training neighbors per class")
        for name, expected in (("exclusion_level", 0.05), ("covariance_shrinkage", 0.1)):
            if type(getattr(self, name)) not in (int, float) or getattr(self, name) != expected:
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
            raise ValueError("Missing or unknown class-local support fields")
        return cls(**value)
