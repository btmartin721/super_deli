"""Frozen point-only multimodal diagnostic contract; no production model defaults."""

from dataclasses import asdict, dataclass, fields

SOURCE_HEAD = "5936eab934772e08e40d275d3d492389b9696e0f"
SOURCE_VERSION = "0.81.0.dev0"
LAWS = (
    "continuous_control",
    "redundant",
    "complementary",
    "conflicting",
    "genomic_only",
    "shape_only",
)
PAIRS = (
    ("snp", "outline"),
    ("traits", "outline"),
    ("habitat", "outline"),
    ("snp", "traits"),
    ("snp", "habitat"),
)


@dataclass(frozen=True)
class SignalLossConfig:
    source_completion_sha256: str
    source_manifest_sha256: str
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 2 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 1800
    max_file_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Use signal-loss diagnostic schema 1")
        if type(self.fixture) is not bool:
            raise ValueError("fixture must be Boolean")
        for value in (self.source_completion_sha256, self.source_manifest_sha256):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Require lowercase SHA-256 source seals")
        for key, ceiling in (
            ("artifact_cap_bytes", 2 * 1024**3),
            ("max_projected_seconds", 1800),
            ("max_file_bytes", 64 * 1024**2),
        ):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= ceiling:
                raise ValueError(key + " exceeds approved bounds")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight-GiB free reserve required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Supply exactly the signal-loss contract fields")
        return cls(**value)


def fresh_tasks(phase):
    if phase == "preflight":
        return [
            dict(index=i, seed=6100 + i, n=n, law=law)
            for i, (n, law) in enumerate(
                (
                    (64, "continuous_control"),
                    (64, "shape_only"),
                    (256, "continuous_control"),
                    (256, "shape_only"),
                )
            )
        ]
    if phase != "main":
        raise ValueError("Known scientific phase required")
    return [
        dict(index=i, seed=6200 + i, n=64 if i < 24 else 256, law=LAWS[i % 6]) for i in range(48)
    ]
