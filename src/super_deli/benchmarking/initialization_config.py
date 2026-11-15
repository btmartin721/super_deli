"""Frozen independent observations and bounded paired initialization validation."""

from dataclasses import asdict, dataclass, fields

LAWS = ("shape_only", "genomic_only", "complementary", "continuous_control", "conflicting")
ARMS = ("legacy10", "legacy100", "diverse100")


@dataclass(frozen=True)
class InitializationStudyConfig:
    source_completion_sha256: str
    source_manifest_sha256: str
    fixture: bool = False
    schema_version: int = 1
    max_file_bytes: int = 64 * 1024**2
    artifact_cap_bytes: int = 8 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_preflight_seconds: int = 1200
    max_main_seconds: int = 3600

    def __post_init__(self):
        if (
            type(self.fixture) is not bool
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("Use initialization schema 1 and a Boolean fixture flag")
        for value in (self.source_completion_sha256, self.source_manifest_sha256):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Require lowercase SHA-256 source seals")
        for name, cap in (
            ("max_file_bytes", 64 * 1024**2),
            ("artifact_cap_bytes", 8 * 1024**3),
            ("max_preflight_seconds", 1200),
            ("max_main_seconds", 3600),
        ):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= cap:
                raise ValueError("Resource limit exceeded: " + name)
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Require eight GiB free")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Supply exactly the initialization study fields")
        return cls(**value)


def tasks(phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Known study phase required")
    repeats, seed = (1, 8000) if phase == "preflight" else (20, 8100)
    return [
        dict(
            index=cell * repeats + r,
            cell=cell,
            repeat=r,
            n=n,
            law=law,
            seed=seed + cell * repeats + r,
            stability=phase == "preflight" or r < 3,
        )
        for cell, (n, law) in enumerate((n, law) for n in (64, 256) for law in LAWS)
        for r in range(repeats)
    ]
