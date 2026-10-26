"""Immutable declarations for a bounded, fit-free individual trait audit."""

import json
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from super_deli.core.correspondence import digest
from super_deli.core.empirical import NAME, checked_fields, text

METADATA = (
    "voucher",
    "individual_number",
    "storage",
    "species",
    "MOTU",
    "family",
    "locality_name",
    "latitude",
    "longitude",
    "locality_type",
    "dinaric",
    "locality_LT",
    "survey_date",
    "legators",
    "measurer",
    "sex",
)
TRAITS = (
    "body_l",
    "antenna_I",
    "antenna_II",
    "gnI_carp",
    "gnI_pl",
    "gnI_pw",
    "gnI_pd",
    "cx_II",
    "gnII_carp",
    "gnII_pl",
    "gnII_pw",
    "gnII_pd",
    "cx_III",
    "ppV_bw",
    "ppV_l",
    "ppVI_bw",
    "ppVI_l",
    "ppVII_bw",
    "ppVII_l",
    "upI_basis",
    "upI_exopodite",
    "upI_endopodite",
    "upIII_exopodite_prox",
    "upIII_exopodite_dist",
    "egg_diam_avg",
)
COLUMNS = (*METADATA, *TRAITS, "eggs_no", "spines", "COI_sequence", "remarks")
ROLES = (
    "workbook",
    "archive",
    "collection",
    "publication",
    "references",
    "reference_text",
    "screen",
)


@dataclass(frozen=True)
class TraitSnapshot:
    role: str
    filename: str
    url: str
    access: str
    sha256: str | None
    size_bytes: int
    license_id: str
    limitation: str

    def __post_init__(self):
        if not isinstance(self.role, str) or self.role not in ROLES:
            raise ValueError("Unsupported trait audit source role")
        if (
            not isinstance(self.filename, str)
            or not self.filename
            or self.filename != self.filename.strip()
            or any(x in self.filename for x in ("/", "\\", "..", "\x00"))
        ):
            raise ValueError("Flat explicit source filename required")
        u = urlsplit(text(self.url))
        if u.scheme != "https" or not u.hostname or u.username or u.password:
            raise ValueError("Explicit HTTPS source without credentials required")
        if self.access not in ("usable", "unavailable"):
            raise ValueError("Explicit evidence access status required")
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= 8 * 1024**2:
            raise ValueError("Source exceeds the 8 MiB snapshot bound")
        if self.size_bytes:
            digest(self.sha256)
        elif self.sha256 is not None:
            raise ValueError("Absent snapshots cannot have a digest")
        if self.access == "usable" and not self.size_bytes:
            raise ValueError("Usable sources require retained bytes")
        if self.role not in ("references", "reference_text") and self.access != "usable":
            raise ValueError("Required documentary sources must be available")
        text(self.license_id)
        text(self.limitation)

    @classmethod
    def from_dict(cls, value):
        return cls(**checked_fields(cls, value))


@dataclass(frozen=True)
class TraitAuditPlan:
    schema_version: int
    audit_id: str
    phase: str
    profile_id: str
    sources: tuple[TraitSnapshot, ...]
    published_counts: tuple[tuple[str, int], ...]
    identity_policy: str = "literal_vouchers_no_alias_inference"
    reference_policy: str = "availability_is_not_taxonomic_validation"
    contact_policy: str = "no_contact"

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported trait audit schema")
        if not isinstance(self.audit_id, str) or not NAME.fullmatch(self.audit_id):
            raise ValueError("Neutral audit ID required")
        if self.phase != "trait_source_audit_only" or self.profile_id != "groundwater_amphipods_v1":
            raise ValueError("Only the bounded fit-free trait source profile is supported")
        if (
            not isinstance(self.sources, tuple)
            or len(self.sources) != len(ROLES)
            or any(not isinstance(s, TraitSnapshot) for s in self.sources)
            or {s.role for s in self.sources} != set(ROLES)
            or len({s.filename for s in self.sources}) != len(self.sources)
        ):
            raise ValueError("Exactly one immutable source per role required")
        if sum(s.size_bytes for s in self.sources) > 16 * 1024**2:
            raise ValueError("Trait source bundle exceeds 16 MiB")
        counts = self.published_counts
        if (
            not isinstance(counts, tuple)
            or len(counts) != 3
            or any(
                not isinstance(p, tuple)
                or len(p) != 2
                or not isinstance(p[0], str)
                or type(p[1]) is not int
                or p[1] < 0
                for p in counts
            )
            or {p[0] for p in counts} != {"rows", "motu_labels", "locality_names"}
        ):
            raise ValueError("Explicit reviewed published counts required")
        if (
            self.identity_policy != "literal_vouchers_no_alias_inference"
            or self.reference_policy != "availability_is_not_taxonomic_validation"
            or self.contact_policy != "no_contact"
        ):
            raise ValueError("No inferred identity, reference promotion or contact is supported")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        v = checked_fields(cls, value)
        if not isinstance(v["sources"], list) or not isinstance(v["published_counts"], list):
            raise ValueError("Audit collections must be arrays")
        v["sources"] = tuple(TraitSnapshot.from_dict(s) for s in v["sources"])
        v["published_counts"] = tuple(
            tuple(x) if isinstance(x, list) else x for x in v["published_counts"]
        )
        return cls(**v)
