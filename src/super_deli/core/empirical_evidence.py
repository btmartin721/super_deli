"""Bound reviewed evidence, without certifying biological independence.

Schema 1 assesses the approved capture-record development case only. Source
hashes verify the reviewed snapshots, not the truth of prose or an analyst's
interpretation. New sampling designs require a separate protocol version.
"""

import json
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from super_deli.core.empirical import HEX, NAME, checked_fields, strings, text
from super_deli.core.provenance import configuration_hash

# A bounded review cannot turn missing evidence into an independence assertion.
CLAIMS = {
    "persistent_identity": "not_established",
    "nest_dependence": "documented",
    "nest_crosswalk": "not_established",
    "cross_season_identity": "not_established",
    "reference_provenance": "documented",
    "taxonomic_context": "documented",
    "specimen_reference_independence": "not_established",
    "upstream_equivalence": "not_established",
}


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or not HEX.fullmatch(value):
        raise ValueError("Exact SHA-256 binding required")


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    role: str
    url: str
    access: str
    sha256: str | None
    size_bytes: int
    failure_reason: str

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not NAME.fullmatch(self.source_id):
            raise ValueError("Neutral evidence source ID required")
        if self.role not in (
            "primary_publication",
            "author_documentation",
            "issue_report",
            "archive_metadata",
        ):
            raise ValueError("Explicit evidence source role required")
        u = urlsplit(text(self.url))
        if u.scheme != "https" or not u.hostname or u.username or u.password:
            raise ValueError("Evidence URL must use HTTPS without credentials")
        if self.access not in ("usable", "unavailable"):
            raise ValueError("Explicit content availability required")
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= 8 * 1024**2:
            raise ValueError("Evidence snapshot must be at most 8 MiB")
        if self.size_bytes:
            digest(self.sha256)
        elif self.sha256 is not None:
            raise ValueError("An absent snapshot cannot have a digest")
        if self.access == "usable":
            if not self.size_bytes or self.failure_reason != "":
                raise ValueError("Usable evidence requires bytes and no access failure")
        else:
            text(self.failure_reason)

    @classmethod
    def from_dict(cls, value):
        return cls(**checked_fields(cls, value))


@dataclass(frozen=True)
class ReviewedClaim:
    claim_id: str
    status: str
    source_ids: tuple[str, ...]
    locator: str
    interpretation: str
    limitation: str
    required_evidence: str

    def __post_init__(self):
        if not isinstance(self.claim_id, str) or CLAIMS.get(self.claim_id) != self.status:
            raise ValueError("Unsupported claim or promotion beyond the bounded review")
        if not isinstance(self.source_ids, tuple) or strings(self.source_ids) != self.source_ids:
            raise ValueError("Explicit immutable evidence references required")
        for v in (self.locator, self.interpretation, self.limitation, self.required_evidence):
            text(v)

    @classmethod
    def from_dict(cls, value):
        v = checked_fields(cls, value)
        if not isinstance(v["source_ids"], list):
            raise ValueError("Evidence references must be arrays")
        v["source_ids"] = tuple(v["source_ids"])
        return cls(**v)


@dataclass(frozen=True)
class EmpiricalEvidence:
    schema_version: int
    phase: str
    dataset_id: str
    source_sha256: str
    profile_sha256: str
    protocol_sha256: str
    sources: tuple[EvidenceSource, ...]
    claims: tuple[ReviewedClaim, ...]
    contradictions: tuple[str, ...]
    assumptions: tuple[str, ...]

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported empirical evidence schema")
        if self.phase != "readiness_audit_only":
            raise ValueError("Evidence review cannot authorize fitting")
        if not isinstance(self.dataset_id, str) or not NAME.fullmatch(self.dataset_id):
            raise ValueError("Neutral dataset ID required")
        for v in (self.source_sha256, self.profile_sha256, self.protocol_sha256):
            digest(v)
        if (
            not isinstance(self.sources, tuple)
            or not self.sources
            or any(not isinstance(s, EvidenceSource) for s in self.sources)
        ):
            raise ValueError("Immutable evidence sources required")
        sources = {s.source_id: s for s in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("Duplicate evidence source")
        if (
            not isinstance(self.claims, tuple)
            or any(not isinstance(c, ReviewedClaim) for c in self.claims)
            or {c.claim_id for c in self.claims} != set(CLAIMS)
            or len(self.claims) != len(CLAIMS)
        ):
            raise ValueError("Every bounded claim must be assessed exactly once")
        for c in self.claims:
            if not set(c.source_ids) <= sources.keys():
                raise ValueError("Unbound evidence reference")
            if c.status == "documented" and not any(
                sources[n].access == "usable"
                and sources[n].role in ("primary_publication", "author_documentation")
                for n in c.source_ids
            ):
                raise ValueError("Documented claims require usable primary/author evidence")
        if (
            not isinstance(self.contradictions, tuple)
            or strings(self.contradictions) != self.contradictions
        ):
            raise ValueError("Record source contradictions explicitly")
        if self.assumptions != () or not isinstance(self.assumptions, tuple):
            raise ValueError("This checkpoint adopts no sampling-independence assumptions")

    def check_bindings(self, profile, protocol):
        protocol.check_profile(profile)
        if (
            self.dataset_id != profile.dataset_id
            or self.source_sha256 != profile.source.sha256
            or self.profile_sha256 != configuration_hash(profile.to_dict())
            or self.protocol_sha256 != configuration_hash(protocol.to_dict())
        ):
            raise ValueError("Evidence must bind the exact source, profile and protocol")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        v = checked_fields(cls, value)
        for name, constructor in (
            ("sources", EvidenceSource.from_dict),
            ("claims", ReviewedClaim.from_dict),
        ):
            if not isinstance(v[name], list):
                raise ValueError("Evidence declarations must be arrays")
            v[name] = tuple(constructor(x) for x in v[name])
        for name in ("contradictions", "assumptions"):
            if not isinstance(v[name], list):
                raise ValueError("Evidence declarations must be arrays")
            v[name] = tuple(v[name])
        return cls(**v)
