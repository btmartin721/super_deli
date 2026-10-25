"""Source-bound correspondence declarations, separate from analytical readiness."""

import json
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from super_deli.core.empirical import HEX, NAME, checked_fields, text

ROLES = frozenset(
    {
        "haul_catalogue",
        "field_dictionary",
        "search_register",
        "voucher_crosswalk",
        "haul_crosswalk",
        "archive_metadata",
        "archive_inventory",
        "molecular_records",
        "molecular_mirror",
        "primary_publication",
        "service_documentation",
        "survey_context",
        "figure",
    }
)


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or not HEX.fullmatch(value):
        raise ValueError("Exact SHA-256 binding required")


@dataclass(frozen=True)
class CorrespondenceSource:
    source_id: str
    role: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    provenance_group: str
    license_id: str
    review_locator: str
    limitation: str

    def __post_init__(self):
        if any(
            not isinstance(x, str) or not NAME.fullmatch(x)
            for x in (self.source_id, self.provenance_group)
        ):
            raise ValueError("Neutral source and provenance identifiers required")
        if self.role not in ROLES:
            raise ValueError("Unsupported correspondence source role")
        if (
            not isinstance(self.filename, str)
            or not self.filename
            or self.filename != self.filename.strip()
            or any(s in self.filename for s in ("/", "\\", "..", "\x00"))
        ):
            raise ValueError("Flat explicit evidence filename required")
        u = urlsplit(text(self.url))
        if u.scheme != "https" or not u.hostname or u.username or u.password:
            raise ValueError("Explicit HTTPS source without credentials required")
        digest(self.sha256)
        if type(self.size_bytes) is not int or not 0 < self.size_bytes <= 8 * 1024**2:
            raise ValueError("Evidence snapshot must be within 1..8 MiB")
        for value in (self.license_id, self.review_locator, self.limitation):
            text(value)

    @classmethod
    def from_dict(cls, value):
        return cls(**checked_fields(cls, value))


@dataclass(frozen=True)
class CorrespondencePlan:
    schema_version: int
    assessment_id: str
    phase: str
    prior_specification_sha256: str
    prior_checksums_sha256: str
    sources: tuple[CorrespondenceSource, ...]
    identity_policy: str = "literal_or_source_documented_only"
    contact_policy: str = "no_contact"

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported correspondence schema")
        if not isinstance(self.assessment_id, str) or not NAME.fullmatch(self.assessment_id):
            raise ValueError("Neutral assessment ID required")
        if self.phase != "correspondence_audit_only":
            raise ValueError("Correspondence review cannot authorize fitting")
        for value in (self.prior_specification_sha256, self.prior_checksums_sha256):
            digest(value)
        if (
            not isinstance(self.sources, tuple)
            or not self.sources
            or any(not isinstance(s, CorrespondenceSource) for s in self.sources)
        ):
            raise ValueError("Explicit immutable evidence sources required")
        for key in ("source_id", "filename"):
            if len({getattr(s, key) for s in self.sources}) != len(self.sources):
                raise ValueError("Duplicate source identity or filename")
        for role in ("haul_catalogue", "field_dictionary", "search_register"):
            if sum(s.role == role for s in self.sources) != 1:
                raise ValueError("Exactly one " + role + " source required")
        if sum(s.size_bytes for s in self.sources) > 16 * 1024**2:
            raise ValueError("Reviewed evidence exceeds 16 MiB")
        if (
            self.identity_policy != "literal_or_source_documented_only"
            or self.contact_policy != "no_contact"
        ):
            raise ValueError("No inferred normalization or contact is supported")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        v = checked_fields(cls, value)
        if not isinstance(v["sources"], list):
            raise ValueError("Evidence declarations must be arrays")
        v["sources"] = tuple(CorrespondenceSource.from_dict(s) for s in v["sources"])
        return cls(**v)
