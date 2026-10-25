"""Fit-free declarations for matched continuous empirical benchmark curation."""

import re
from dataclasses import asdict, dataclass, fields
from urllib.parse import urlsplit

HEX = re.compile(r"[0-9a-f]+\Z")
NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def checked_fields(cls, value):
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
        raise ValueError(f"Exactly the declared {cls.__name__} fields are required")
    return dict(value)


def text(value):
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise ValueError("Nonempty unpadded text without NUL required")
    return value


def strings(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Nonempty string sequence required")
    result = tuple(text(v) for v in value)
    if len(set(result)) != len(result):
        raise ValueError("Duplicate declarations are not allowed")
    return result


@dataclass(frozen=True)
class LockedSource:
    url: str
    sha256: str
    size_bytes: int
    git_blob_sha1: str
    license_id: str
    source_role: str
    direct_upstream_verified: bool

    def __post_init__(self):
        parsed = urlsplit(text(self.url))
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("An explicit HTTPS source URL without credentials is required")
        for v, n in ((self.sha256, 64), (self.git_blob_sha1, 40)):
            if not isinstance(v, str) or len(v) != n or not HEX.fullmatch(v):
                raise ValueError("Exact SHA-256 and Git blob SHA-1 fingerprints required")
        if type(self.size_bytes) is not int or not 1 <= self.size_bytes <= 8 * 1024**2:
            raise ValueError("Source byte count must be in 1..8 MiB")
        if self.license_id != "CC0-1.0" or self.source_role != "author_maintained_mirror":
            raise ValueError("This curation schema requires an explicitly licensed CC0 mirror")
        if type(self.direct_upstream_verified) is not bool or self.direct_upstream_verified:
            raise ValueError("Direct upstream verification is unavailable in this schema")

    @classmethod
    def from_dict(cls, value):
        return cls(**checked_fields(cls, value))


@dataclass(frozen=True)
class CuratedFeature:
    source_column: str
    name: str
    kind: str
    unit: str
    positive: bool

    def __post_init__(self):
        text(self.source_column)
        if (
            not isinstance(self.name, str)
            or not NAME.fullmatch(self.name)
            or self.name in {"observation_id", "specimen_id", "unit_id"}
        ):
            raise ValueError("Feature names must be neutral snake_case")
        if (
            not isinstance(self.kind, str)
            or self.kind not in {"morphological", "ecological"}
            or type(self.positive) is not bool
        ):
            raise ValueError("Explicit continuous morphology/ecology feature roles required")
        text(self.unit)

    @classmethod
    def from_dict(cls, value):
        return cls(**checked_fields(cls, value))


@dataclass(frozen=True)
class EmpiricalProfile:
    schema_version: int
    dataset_id: str
    source: LockedSource
    source_columns: tuple[str, ...]
    identity_columns: tuple[str, ...]
    reference_column: str
    metadata_columns: tuple[str, ...]
    ignored_columns: tuple[str, ...]
    missing_values: tuple[str, ...]
    features: tuple[CuratedFeature, ...]
    identity_grain: str
    reference_role: str

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported empirical curation schema")
        if not isinstance(self.dataset_id, str) or not NAME.fullmatch(self.dataset_id):
            raise ValueError("Neutral dataset identifier required")
        if not isinstance(self.source, LockedSource):
            raise ValueError("Explicit source lock required")
        for v in (self.source_columns, self.identity_columns, self.metadata_columns):
            if not isinstance(v, tuple) or strings(v) != v:
                raise ValueError("Immutable unique column declarations required")
        if not isinstance(self.ignored_columns, tuple):
            raise ValueError("Ignored column declarations must be a tuple")
        if self.ignored_columns:
            strings(self.ignored_columns)
        text(self.reference_column)
        if (
            not isinstance(self.features, tuple)
            or not self.features
            or any(not isinstance(f, CuratedFeature) for f in self.features)
        ):
            raise ValueError("Explicit immutable continuous features required")
        if {f.kind for f in self.features} != {"morphological", "ecological"}:
            raise ValueError("Both matched continuous morphology and ecology are required")
        assigned = (
            *self.identity_columns,
            self.reference_column,
            *self.metadata_columns,
            *self.ignored_columns,
            *(f.source_column for f in self.features),
        )
        if len(set(assigned)) != len(assigned) or set(assigned) != set(self.source_columns):
            raise ValueError("Each source column needs exactly one disjoint role")
        if len({f.name for f in self.features}) != len(self.features):
            raise ValueError("Duplicate analytical feature names")
        if (
            not isinstance(self.missing_values, tuple)
            or not self.missing_values
            or any(
                not isinstance(v, str) or v != v.strip() or "\x00" in v for v in self.missing_values
            )
            or len(set(self.missing_values)) != len(self.missing_values)
        ):
            raise ValueError("Exact unique missing-value tokens required")
        if (
            self.identity_grain != "capture_record"
            or self.reference_role != "development_reference"
        ):
            raise ValueError("Only descriptive capture-record development curation is supported")

    def to_dict(self):
        import json

        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        v = checked_fields(cls, value)
        v["source"] = LockedSource.from_dict(v["source"])
        if not isinstance(v["features"], list):
            raise ValueError("Features must be an array")
        v["features"] = tuple(CuratedFeature.from_dict(f) for f in v["features"])
        for n in (
            "source_columns",
            "identity_columns",
            "metadata_columns",
            "ignored_columns",
            "missing_values",
        ):
            if not isinstance(v[n], list):
                raise ValueError(f"{n} must be an array")
            v[n] = tuple(v[n])
        return cls(**v)
