"""Dependency-free contract for an archival diagnosis, never a production model."""

from dataclasses import asdict, dataclass, fields

REPRESENTATIONS = (
    "source_standard",
    "traits_standard",
    "pca_native",
    "neural_0_raw",
    "neural_101_raw",
    "neural_202_raw",
    "neural_0_standard",
    "neural_101_standard",
    "neural_202_standard",
)
POLICIES = {
    "baseline": ("eom", True),
    "no_root": ("eom", False),
    "leaf": ("leaf", True),
}
DENSITY_ARMS = ("density", "neural_0", "neural_101", "neural_202")


@dataclass(frozen=True)
class NeuralMoonsConfig:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 3600
    max_workers: int = 2
    max_working_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported neural-moons diagnostic schema")
        if type(self.fixture) is not bool:
            raise ValueError("fixture must be an explicit boolean")
        for field in fields(self):
            if field.name not in {"schema_version", "fixture"}:
                value = getattr(self, field.name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"{field.name} must be a positive integer")
        if self.max_workers > 2:
            raise ValueError("At most two diagnostic workers are supported")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or "schema_version" not in value
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Malformed neural-moons diagnostic configuration")
        return cls(**value)
