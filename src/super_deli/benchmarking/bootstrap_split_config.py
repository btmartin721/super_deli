"""Frozen diagnostic roster and resource limits; no production tuning."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class BootstrapSplitConfig:
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
            raise ValueError("Use diagnostic schema1 and Boolean fixture flag")
        for x in (self.source_completion_sha256, self.source_manifest_sha256):
            if (
                not isinstance(x, str)
                or len(x) != 64
                or any(c not in "0123456789abcdef" for c in x)
            ):
                raise ValueError("Require lowercase SHA256 source seals")
        for key, cap in (
            ("max_file_bytes", 64 * 1024**2),
            ("artifact_cap_bytes", 8 * 1024**3),
            ("max_preflight_seconds", 1200),
            ("max_main_seconds", 5400),
        ):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= cap:
                raise ValueError("Resource cap exceeded: " + key)
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Require eight GiB free")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, x):
        if not isinstance(x, dict) or set(x) != {f.name for f in fields(cls)}:
            raise ValueError("Supply exact diagnostic fields")
        return cls(**x)


def tasks(phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Known phase required")
    repeats, base = (1, 9100) if phase == "preflight" else (6, 9200)
    return [
        dict(index=cell * repeats + r, cell=cell, n=n, law=law, seed=base + cell * repeats + r)
        for cell, (n, law) in enumerate(
            (n, law) for n in (64, 256) for law in ("continuous_control", "shape_only")
        )
        for r in range(repeats)
    ]
