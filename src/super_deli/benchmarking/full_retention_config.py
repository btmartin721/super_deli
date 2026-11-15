"""Bounded point/stability validation; scientific settings are fixed before generation."""

from dataclasses import asdict, dataclass, fields

SOURCE_HEAD = "1f880e0873c2999a51243997d78685586c302c72"
SOURCE_VERSION = "0.82.0.dev0"
LAWS = ("shape_only", "genomic_only", "complementary", "continuous_control")
REPLAYS = (0, 2, 4, 5, 6, 8, 10, 11, 12, 14, 22, 23)
MISSES = (11, 23)


@dataclass(frozen=True)
class FullRetentionStudyConfig:
    source_completion_sha256: str
    source_manifest_sha256: str
    fixture: bool = False
    schema_version: int = 1
    max_file_bytes: int = 64 * 1024**2
    artifact_cap_bytes: int = 4 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_preflight_seconds: int = 600
    max_main_seconds: int = 1800

    def __post_init__(self):
        if (
            type(self.fixture) is not bool
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError(
                "Use full-retention study schema 1 with an explicit Boolean fixture flag"
            )
        for x in (self.source_completion_sha256, self.source_manifest_sha256):
            if (
                not isinstance(x, str)
                or len(x) != 64
                or any(c not in "0123456789abcdef" for c in x)
            ):
                raise ValueError("Require lowercase SHA-256 source seals")
        for name, cap in (
            ("max_file_bytes", 64 * 1024**2),
            ("artifact_cap_bytes", 6 * 1024**3),
            ("max_preflight_seconds", 600),
            ("max_main_seconds", 1800),
        ):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= cap:
                raise ValueError("Approved limit exceeded: " + name)
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Require at least eight GiB free")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict) or set(d) != {f.name for f in fields(cls)}:
            raise ValueError("Supply exactly the full-retention study fields")
        return cls(**d)


def tasks(phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Known full-retention phase required")
    repeats, seed = (1, 7100) if phase == "preflight" else (20, 7200)
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
