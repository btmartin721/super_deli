"""Immutable source locks for a bounded, fit-free specimen audit."""

import json
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from super_deli.core.empirical import HEX, NAME, checked_fields, strings, text

ROLES = (
    "metadata",
    "colour",
    "full_disc",
    "anterior",
    "rostrum",
    "author_code",
    "readme",
    "references",
    "publication",
    "supplement",
    "archive_metadata",
    "archive_files",
)


@dataclass(frozen=True)
class AuditSource:
    role: str
    filename: str
    url: str
    sha256: str
    size_bytes: int

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError("Unknown audit source role")
        if (
            not isinstance(self.filename, str)
            or not self.filename
            or any(s in self.filename for s in ("/", "\\", "..", "\x00"))
            or self.filename != self.filename.strip()
        ):
            raise ValueError("Flat explicit source filenames required")
        u = urlsplit(text(self.url))
        if u.scheme != "https" or not u.hostname or u.username or u.password:
            raise ValueError("Explicit HTTPS source URL without credentials required")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or not HEX.fullmatch(self.sha256)
        ):
            raise ValueError("Exact source SHA-256 required")
        if type(self.size_bytes) is not int or not 0 < self.size_bytes <= 8 * 1024**2:
            raise ValueError("Each source must be within 1..8 MiB")

    @classmethod
    def from_dict(cls, value):
        return cls(**checked_fields(cls, value))


@dataclass(frozen=True)
class ReplacementAuditPlan:
    schema_version: int
    audit_id: str
    phase: str
    archive_license_id: str
    sources: tuple[AuditSource, ...]
    voucher_prefixes: tuple[str, ...]
    taxon_names: tuple[tuple[str, str], ...]
    reference_policy: str
    sampling_policy: str
    scale_policy: str
    review_limitations: tuple[str, ...]

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported replacement audit schema")
        if not isinstance(self.audit_id, str) or not NAME.fullmatch(self.audit_id):
            raise ValueError("Neutral audit ID required")
        if self.phase != "source_audit_only" or self.archive_license_id != "CC0-1.0":
            raise ValueError("Only the approved fit-free CC0 source audit is supported")
        if (
            not isinstance(self.sources, tuple)
            or any(not isinstance(s, AuditSource) for s in self.sources)
            or len(self.sources) != len(ROLES)
            or {s.role for s in self.sources} != set(ROLES)
        ):
            raise ValueError("Exactly one locked source per audit role required")
        if len({s.filename for s in self.sources}) != len(self.sources):
            raise ValueError("Source paths must be disjoint")
        if (
            not isinstance(self.voucher_prefixes, tuple)
            or strings(self.voucher_prefixes) != self.voucher_prefixes
        ):
            raise ValueError("Explicit immutable candidate voucher prefixes required")
        if any(
            a.startswith(b) for a in self.voucher_prefixes for b in self.voucher_prefixes if a != b
        ):
            raise ValueError("Overlapping candidate namespaces are ambiguous")
        if (
            not isinstance(self.taxon_names, tuple)
            or len(self.taxon_names) != 2
            or any(
                not isinstance(pair, tuple) or len(pair) != 2 or any(text(x) != x for x in pair)
                for pair in self.taxon_names
            )
            or len({p[0] for p in self.taxon_names}) != 2
            or len({p[1] for p in self.taxon_names}) != 2
        ):
            raise ValueError("Two explicit, disjoint source-to-GenBank taxon names required")
        if (
            self.reference_policy != "candidate_namespaces_are_not_authoritative"
            or self.sampling_policy != "documented_design_without_independence_assertion"
            or self.scale_policy != "retain_factor_without_physical_unit_assertion"
        ):
            raise ValueError("Audit declarations cannot grant biological or fitting readiness")
        if (
            not isinstance(self.review_limitations, tuple)
            or strings(self.review_limitations) != self.review_limitations
        ):
            raise ValueError("Scientific review limitations must be explicit")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        v = checked_fields(cls, value)
        for name in ("sources", "voucher_prefixes", "taxon_names", "review_limitations"):
            if not isinstance(v[name], list):
                raise ValueError("Audit collections must be arrays")
        v["sources"] = tuple(AuditSource.from_dict(s) for s in v["sources"])
        v["taxon_names"] = tuple(tuple(p) if isinstance(p, list) else p for p in v["taxon_names"])
        for name in ("voucher_prefixes", "review_limitations"):
            v[name] = tuple(v[name])
        return cls(**v)
