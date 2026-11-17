"""Frozen abundance/total-size design, source seals and bounded resources."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class HeldoutStudyConfig:
    source_completion_sha256: str
    source_manifest_sha256: str
    fixture: bool = False
    schema_version: int = 1
    max_file_bytes: int = 64 * 1024**2
    artifact_cap_bytes: int = 10 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_preflight_seconds: int = 1200
    max_main_seconds: int = 7200

    def __post_init__(self):
        if (
            type(self.fixture) is not bool
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("Explicit study schema1 and fixture flag required")
        for x in (self.source_completion_sha256, self.source_manifest_sha256):
            if (
                not isinstance(x, str)
                or len(x) != 64
                or any(c not in "0123456789abcdef" for c in x)
            ):
                raise ValueError("Source SHA256 seals required")
        for key, limit in (
            ("max_file_bytes", 64 * 1024**2),
            ("artifact_cap_bytes", 12 * 1024**3),
            ("max_preflight_seconds", 1200),
            ("max_main_seconds", 7200),
        ):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= limit:
                raise ValueError("Resource cap: " + key)
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight GiB free required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Exact bounded study declaration required")
        return cls(**value)


CELLS = (
    (64, "control", (64,)),
    (128, "control", (128,)),
    (256, "control", (256,)),
    (64, "balanced", (16, 16, 16, 16)),
    (64, "proportional", (25, 16, 13, 10)),
    (64, "rare_six", (29, 16, 13, 6)),
    (128, "proportional", (50, 32, 26, 20)),
    (256, "proportional", (100, 64, 52, 40)),
    (128, "fixed_ten", (55, 35, 28, 10)),
    (256, "fixed_ten", (114, 73, 59, 10)),
)


def tasks(phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Known phase required")
    repeats, base = (1, 20900) if phase == "preflight" else (20, 21000)
    return [
        dict(
            index=c * repeats + j,
            cell=c,
            n=n,
            design=name,
            counts=list(counts),
            seed=base + c * repeats + j,
        )
        for c, (n, name, counts) in enumerate(CELLS)
        for j in range(repeats)
    ]
