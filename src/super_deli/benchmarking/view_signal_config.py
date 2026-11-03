"""Frozen archival diagnosis; no production model or parameter search."""

from dataclasses import asdict, dataclass, fields

CASES = ("morphology_only", "four_complementary", "four_redundant")
SEEDS = (0, 101, 202)
REPRESENTATIONS = (
    (
        "morphology_standard",
        "ecology_standard",
        "balanced_traits",
        "morphology_view_pcs",
        "ecology_view_pcs",
        "fused_profiles",
        "joint_pcs",
        "morphology_density_pcs",
        "ecology_density_pcs",
    )
    + tuple(f"paired_{s}_{scale}" for s in SEEDS for scale in ("raw", "standard"))
    + tuple(f"{view}_{s}_standard" for view in ("morphology", "ecology") for s in SEEDS)
)


@dataclass(frozen=True)
class ViewSignalConfig:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 512 * 1024**2
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 3600
    max_workers: int = 2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported view-signal diagnostic schema")
        if type(self.fixture) is not bool:
            raise ValueError("Explicit fixture flag required")
        for name, cap in (
            ("artifact_cap_bytes", 512 * 1024**2),
            ("max_projected_seconds", 3600),
            ("max_workers", 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError("Diagnostic resource limit exceeds frozen scope")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("At least eight GiB free required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown diagnostic configuration fields")
        if "schema_version" not in value:
            raise ValueError("Explicit diagnostic schema required")
        return cls(**value)
