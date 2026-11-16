"""Fixed paired validation roster and resource gates for the opt-in workflow."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class IndividualSubsampleStudyConfig:
    source_completion_sha256: str
    source_manifest_sha256: str
    fixture: bool = False
    schema_version: int = 1
    max_file_bytes: int = 64 * 1024**2
    artifact_cap_bytes: int = 8 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_preflight_seconds: int = 1200
    max_main_seconds: int = 5400

    def __post_init__(self):
        if (
            type(self.fixture) is not bool
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("Require explicit fixture flag and study schema1")
        for value in (self.source_completion_sha256, self.source_manifest_sha256):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Require lower-case source SHA256 seals")
        for key, cap in (
            ("max_file_bytes", 64 * 1024**2),
            ("artifact_cap_bytes", 8 * 1024**3),
            ("max_preflight_seconds", 1200),
            ("max_main_seconds", 5400),
        ):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= cap:
                raise ValueError("Resource cap: " + key)
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight GiB free required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Supply exact bounded study fields")
        return cls(**value)


def tasks(phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Known phase required")
    repeats, base = (1, 17900) if phase == "preflight" else (8, 18000)
    return [
        dict(index=c * repeats + j, cell=c, n=n, law=law, seed=base + c * repeats + j)
        for c, (n, law) in enumerate(
            (n, law)
            for n in (64, 256)
            for law in ("continuous_control", "shape_balanced", "shape_unequal")
        )
        for j in range(repeats)
    ]
