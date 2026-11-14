"""Dependency-free, separately versioned read-only genomic diagnosis contract."""

import re
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class GenomicDiagnosisConfig:
    source_completion_sha256: str
    source_manifest_sha256: str
    source_commit: str
    schema_version: int = 1
    source_version: str = "0.78.0.dev0"
    source_role: str = "completed_study"
    expected_observations: int = 48
    expected_point_models: int = 240
    expected_subsample_models: int = 360
    max_output_bytes: int = 67108864
    max_file_bytes: int = 33554432

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported genomic diagnosis schema")
        for name, length in (
            ("source_completion_sha256", 64),
            ("source_manifest_sha256", 64),
            ("source_commit", 40),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
                raise ValueError("Source SHA declarations must be lowercase hexadecimal")
        if self.source_role not in ("completed_study", "invented_fixture"):
            raise ValueError("Declare completed_study or invented_fixture")
        if self.source_version != (
            "0.78.0.dev0" if self.source_role == "completed_study" else "invented-fixture"
        ):
            raise ValueError("Unsupported source version/role")
        for name in (
            "expected_observations",
            "expected_point_models",
            "max_output_bytes",
            "max_file_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive expected counts and resource limits required")
        if type(self.expected_subsample_models) is not int or self.expected_subsample_models < 0:
            raise ValueError("Nonnegative expected subsample count required")
        if self.source_role == "completed_study" and (
            self.expected_observations,
            self.expected_point_models,
            self.expected_subsample_models,
        ) != (48, 240, 360):
            raise ValueError(
                "Completed schema-0.78 study has 48 observations and 600 archived maps"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        required = {
            "schema_version",
            "source_completion_sha256",
            "source_manifest_sha256",
            "source_commit",
        }
        if (
            not isinstance(value, dict)
            or required - value.keys()
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Missing or unknown genomic diagnosis fields")
        return cls(**value)
