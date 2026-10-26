"""Immutable, purpose-specific trait evidence review; never a fitting permit."""

import json
from dataclasses import asdict, dataclass

from super_deli.core.correspondence import CorrespondenceSource, digest
from super_deli.core.empirical import NAME, checked_fields, text

CLAIMS = {
    "trait_selection": "documented_targeted",
    "source_assignments": "documented_mixed_basis",
    "molecular_methods": "documented_sequence_based",
    "accepted_partitions": "documented_integrative_review",
    "catalogue_access": "not_public",
    "sampling_independence": "not_established",
    "reference_species_truth": "not_established",
    "imputation_provenance": "documented_reference_informed",
}
TABLES = {
    "reference01": ("Sheet1", 1),
    "reference02": ("MOTU_tree", 1),
    "reference03": ("Sheet1", 1),
    "reference05": ("Syntopy", 3),
    "reference07": ("Sheet1", 3),
}


@dataclass(frozen=True)
class TraitEvidenceClaim:
    claim_id: str
    status: str
    source_id: str
    locator: str
    anchor: str
    interpretation: str
    limitation: str
    required_evidence: str

    def __post_init__(self):
        if not isinstance(self.claim_id, str) or CLAIMS.get(self.claim_id) != self.status:
            raise ValueError("Unsupported claim or promotion beyond the reviewed profile")
        for value in (
            self.source_id,
            self.locator,
            self.anchor,
            self.interpretation,
            self.limitation,
            self.required_evidence,
        ):
            text(value)
        if not NAME.fullmatch(self.source_id):
            raise ValueError("Neutral claim source ID required")
        if len(self.anchor) > 160:
            raise ValueError("Use a short exact source locator, not a reproduced passage")

    @classmethod
    def from_dict(cls, value):
        return cls(**checked_fields(cls, value))


@dataclass(frozen=True)
class TraitEvidencePlan:
    schema_version: int
    assessment_id: str
    phase: str
    prior_specification_sha256: str
    prior_checksums_sha256: str
    sources: tuple[CorrespondenceSource, ...]
    claims: tuple[TraitEvidenceClaim, ...]
    identity_policy: str = "literal_documentary_keys_only"
    reference_policy: str = "retain_partitions_without_species_truth_promotion"
    contact_policy: str = "no_contact"

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported trait evidence schema")
        if not isinstance(self.assessment_id, str) or not NAME.fullmatch(self.assessment_id):
            raise ValueError("Neutral assessment ID required")
        if self.phase != "trait_evidence_audit_only":
            raise ValueError("Evidence assessment cannot authorize fitting")
        digest(self.prior_specification_sha256)
        digest(self.prior_checksums_sha256)
        if (
            not isinstance(self.sources, tuple)
            or not self.sources
            or any(not isinstance(s, CorrespondenceSource) for s in self.sources)
        ):
            raise ValueError("Immutable source snapshots required")
        for key in ("source_id", "filename"):
            if len({getattr(s, key) for s in self.sources}) != len(self.sources):
                raise ValueError("Duplicate source identity or filename")
        if sum(s.size_bytes for s in self.sources) > 16 * 1024**2:
            raise ValueError("New source evidence exceeds 16 MiB")
        if (
            not isinstance(self.claims, tuple)
            or len(self.claims) != len(CLAIMS)
            or any(not isinstance(c, TraitEvidenceClaim) for c in self.claims)
            or {c.claim_id for c in self.claims} != set(CLAIMS)
        ):
            raise ValueError("Assess every bounded claim exactly once")
        sources = {s.source_id: s for s in self.sources}
        for claim in self.claims:
            if claim.source_id not in sources:
                raise ValueError("Unbound reviewed source")
            if not sources[claim.source_id].filename.endswith(".txt"):
                raise ValueError("Claims need a separately locked reviewed text snapshot")
        if (
            self.identity_policy != "literal_documentary_keys_only"
            or self.reference_policy != "retain_partitions_without_species_truth_promotion"
            or self.contact_policy != "no_contact"
        ):
            raise ValueError("No inferred identity, label promotion or contact is supported")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        v = checked_fields(cls, value)
        for name, constructor in (
            ("sources", CorrespondenceSource.from_dict),
            ("claims", TraitEvidenceClaim.from_dict),
        ):
            if not isinstance(v[name], list):
                raise ValueError("Immutable declarations must be JSON arrays")
            v[name] = tuple(constructor(x) for x in v[name])
        return cls(**v)
