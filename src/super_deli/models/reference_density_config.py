"""Fixed class-specific support contract; predecessor policies remain unchanged."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ReferenceDensityConfig:
    assume_independent_calibration: bool
    schema_version: int = 1
    exclusion_level: float = 0.05
    covariance_shrinkage: float = 0.1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported reference-density schema")
        if self.assume_independent_calibration is not True:
            raise ValueError("Independent known calibration must be explicitly acknowledged")
        for name, expected in (("exclusion_level", 0.05), ("covariance_shrinkage", 0.1)):
            if type(getattr(self, name)) not in (int, float) or getattr(self, name) != expected:
                raise ValueError("This version fixes exclusion .05 and shrinkage .1")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or "assume_independent_calibration" not in value
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Missing or unknown reference-density fields")
        return cls(**value)
