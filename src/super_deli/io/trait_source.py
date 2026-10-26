"""Source-role reconstruction and descriptive accounting without estimators."""

import hashlib
import re
import statistics
from collections import Counter, defaultdict
from io import BytesIO

from super_deli.core.provenance import configuration_hash
from super_deli.core.trait_audit import COLUMNS, TRAITS
from super_deli.io.audit_workbook import xml_bytes
from super_deli.io.delimited import identifier
from super_deli.io.numeric import finite_float
from super_deli.io.specimen_audit import genbank_references

MISSING = frozenset(("", "NA", "unknown"))
ACCESSION = re.compile(r"[A-Z]{1,2}[0-9]{5,8}(?:\.[0-9]+)?\Z")
CAUTIONS = ("possibly swapped", "conflicting motu delimitation")


def source_rows(book):
    data = book["sheets"]["Data"]
    if not data or data[0]["row"] != 1:
        raise ValueError("Explicit first-row data headers required")
    header = {col: cell["value"] for col, cell in data[0]["cells"].items()}
    if len(header) != len(COLUMNS) or set(header.values()) != set(COLUMNS):
        raise ValueError("Exactly the declared 45 source columns required")
    result = []
    for source in data[1:]:
        if any(c not in header for c in source["cells"]):
            raise ValueError("Undeclared source data column")
        values = {
            name: source["cells"].get(col, {}).get("value", "") for col, name in header.items()
        }
        cells = {header[col]: cell for col, cell in source["cells"].items()}
        result.append(dict(source_row=source["row"], values=values, cells=cells))
    if not result:
        raise ValueError("Empty source data table")
    dictionary = {}
    for row in book["sheets"]["Descriptions"]:
        cells = row["cells"]
        name = cells.get("A", {}).get("value", "")
        if name not in COLUMNS:
            continue
        if name in dictionary:
            raise ValueError("Duplicate data dictionary field")
        dictionary[name] = {c: cells.get(c, {}).get("value", "") for c in ("A", "B", "C")}
    if set(dictionary) != set(COLUMNS):
        raise ValueError("Every source column requires its explicit dictionary entry")
    if any(
        not dictionary[name]["B"].startswith("Continuous")
        or "millimeters" not in dictionary[name]["C"].casefold()
        for name in TRAITS
    ):
        raise ValueError("This profile requires explicitly documented continuous millimetre traits")
    return result, dictionary, header


def numeric_cell(cell):
    value = cell.get("value", "")
    if cell.get("formula") is not None and not cell.get("cache_present", False):
        return None, "formula_without_cache"
    if value in MISSING:
        return None, None
    if cell.get("type") in ("e", "b", "d"):
        return None, "nonmeasurement_cell_type"
    try:
        if value != value.strip():
            raise ValueError("Padded measurement")
        return finite_float(value, "source measurement"), None
    except ValueError:
        return None, "invalid_or_nonfinite_numeric_token"


def inspect_rows(records):
    voucher_counts = Counter(
        r["values"]["voucher"] for r in records if r["values"]["voucher"] not in MISSING
    )
    for row in records:
        values, issues = row["values"], {}
        voucher = values["voucher"]
        try:
            if voucher in MISSING:
                status = "missing"
            else:
                identifier(voucher, "source voucher")
                status = "duplicate" if voucher_counts[voucher] != 1 else "unique"
        except ValueError:
            status = "invalid"
        row["identity_status"] = status
        row["unit_id"] = (
            "v_" + hashlib.sha256(voucher.encode()).hexdigest()[:20] if status == "unique" else None
        )
        row["numeric_values"] = {}
        row["nonpositive_traits"] = []
        for name in TRAITS:
            value, issue = numeric_cell(row["cells"].get(name, {}))
            row["numeric_values"][name] = value
            if issue:
                issues[name] = issue
            if value is not None and value <= 0:
                row["nonpositive_traits"].append(name)
        row["numeric_issues"] = issues
        remarks = values["remarks"].casefold()
        row["reference_cautions"] = [c for c in CAUTIONS if c in remarks]
        row["representation_eligible"] = status == "unique" and not issues
    ids = [r["unit_id"] for r in records if r["unit_id"] is not None]
    if len(ids) != len(set(ids)):
        raise ValueError("Anonymous source identity collision")


def molecular_records(path):
    raw = path.read_bytes()
    # NCBI declares an external DTD; preserve the source and never retrieve it.
    declaration = (
        b'<!DOCTYPE GBSet PUBLIC "-//NCBI//NCBI GBSeq/EN" '
        b'"https://www.ncbi.nlm.nih.gov/dtd/NCBI_GBSeq.dtd">'
    )
    if raw.count(declaration) == 1:
        prefix, suffix = raw.split(declaration)
        if not re.fullmatch(rb"\s*<\?xml[^?]*\?>\s*", prefix):
            raise ValueError("NCBI declaration must precede the root element")
        raw = prefix + suffix
    root = xml_bytes(raw)
    refs = genbank_references(BytesIO(raw))
    genes = {}
    for seq in root:
        qualifiers = seq.findall(".//GBQualifier")
        g = [
            q.findtext("GBQualifier_value", "")
            for q in qualifiers
            if q.findtext("GBQualifier_name") == "gene"
        ]
        p = [
            q.findtext("GBQualifier_value", "")
            for q in qualifiers
            if q.findtext("GBQualifier_name") == "product"
        ]
        source = [
            f
            for f in seq.findall("GBSeq_feature-table/GBFeature")
            if f.findtext("GBFeature_key") == "source"
        ]
        source_qualifiers = [
            (q.findtext("GBQualifier_name"), q.findtext("GBQualifier_value"))
            for f in source
            for q in f.findall("GBFeature_quals/GBQualifier")
        ]
        genes[seq.findtext("GBSeq_accession-version")] = dict(
            source_qualifiers=source_qualifiers,
            genes=sorted(set(g)),
            products=sorted(set(p)),
            coi_marker=any(x.casefold() in {"coi", "co1", "cox1", "coxi"} for x in g)
            or any(
                re.search(r"cytochrome(?: c)? oxidase subunit (?:i|1)(?:\b|$)", x, re.I) for x in p
            ),
        )
    return [dict(**r, **genes[r["version"]]) for r in refs]


def reference_accounting(rows, refs, available):
    lookup = defaultdict(list)
    for ref in refs:
        lookup[ref["accession"]].append(ref)
    tokens = Counter(
        r["values"]["COI_sequence"].split(".")[0]
        for r in rows
        if ACCESSION.fullmatch(r["values"]["COI_sequence"])
    )
    result = []
    for row in rows:
        value = row["values"]["COI_sequence"]
        matches = lookup.get(value.split(".")[0], [])
        record = matches[0] if len(matches) == 1 else None
        if value in MISSING:
            status = "not_stated"
        elif value == "check":
            status = "placeholder"
        elif not ACCESSION.fullmatch(value):
            status = "malformed_accession"
        elif tokens[value.split(".")[0]] != 1:
            status = "ambiguous_accession_usage"
        elif not available:
            status = "retrieval_unavailable"
        elif not matches:
            status = "not_returned"
        elif record is None:
            status = "ambiguous_record"
        elif "." in value and value != record["version"]:
            status = "version_disagreement"
        elif row["identity_status"] != "unique":
            status = "source_voucher_unresolved"
        elif not record["voucher"]:
            status = "source_documented_record_voucher_absent"
        elif record["voucher"] != row["values"]["voucher"]:
            status = "voucher_disagreement"
        elif not record["coi_marker"]:
            status = "marker_not_coi"
        else:
            status = "literal_voucher_coi_match"
        clean = status == "literal_voucher_coi_match" and not row["reference_cautions"]
        result.append(
            dict(
                source_row=row["source_row"],
                voucher=row["values"]["voucher"],
                stated_accession=value,
                status=status,
                returned_record=record,
                source_motu=row["values"]["MOTU"],
                source_taxon=row["values"]["species"],
                source_cautions=row["reference_cautions"],
                unflagged_literal_link=clean,
                reference_use="withheld_source_caution"
                if row["reference_cautions"]
                else "traceability_only",
                taxonomic_reference_independence="not_established_by_accession_availability",
                organism_label_equal=None
                if record is None
                else record["organism"] == row["values"]["species"],
            )
        )
    return result


def locality_accounting(rows, book):
    links, contexts = [], defaultdict(list)
    names_to_coordinates, coordinates_to_names = defaultdict(set), defaultdict(set)
    for row in rows:
        v = row["values"]
        triple = tuple(v[n] for n in ("locality_name", "latitude", "longitude"))
        issues = []
        if any(x in MISSING for x in triple):
            issues.append("missing_locality_context")
        else:
            try:
                lat, lon = (
                    finite_float(v["latitude"], "latitude"),
                    finite_float(v["longitude"], "longitude"),
                )
                if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                    raise ValueError("Coordinate outside geographic range")
            except ValueError:
                issues.append("invalid_coordinate")
        key = (
            None
            if issues
            else "context_"
            + configuration_hash(dict(zip(("name", "latitude", "longitude"), triple, strict=True)))[
                :20
            ]
        )
        if key is not None:
            contexts[key].append(row)
            names_to_coordinates[triple[0]].add(triple[1:])
            coordinates_to_names[triple[1:]].add(triple[0])
        date = v["survey_date"]
        date_cell = row["cells"].get("survey_date", {})
        date_kind = (
            "missing"
            if date in MISSING
            else "stored_numeric_not_converted"
            if date_cell.get("type") == "n"
            else "year"
            if re.fullmatch(r"[0-9]{4}", date)
            else "year_range"
            if re.fullmatch(r"[0-9]{4}-[0-9]{4}", date)
            else "month_year"
            if re.fullmatch(r"[0-9]{1,2}\.[0-9]{4}", date)
            else "day_month_year"
            if re.fullmatch(r"[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}", date)
            else "text_preserved_uninterpreted"
        )
        links.append(
            dict(
                source_row=row["source_row"],
                voucher=v["voucher"],
                locality_name=triple[0],
                latitude=triple[1],
                longitude=triple[2],
                context_descriptor=key,
                context_issues=issues,
                survey_date=date,
                date_kind=date_kind,
                date_cell_type=date_cell.get("type"),
                date_style=date_cell.get("style"),
                date1904=book["date1904"],
                statistical_sampling_group_asserted=False,
            )
        )
    groups = [
        dict(
            context_descriptor=k,
            source_rows=[r["source_row"] for r in rs],
            source_motu_counts=dict(
                sorted(
                    Counter(
                        r["values"]["MOTU"] for r in rs if r["values"]["MOTU"] not in MISSING
                    ).items()
                )
            ),
        )
        for k, rs in sorted(contexts.items())
    ]
    summary = dict(
        complete_context_rows=sum(link["context_descriptor"] is not None for link in links),
        exact_context_descriptors=len(contexts),
        names_with_multiple_coordinate_tokens=sum(
            len(v) > 1 for v in names_to_coordinates.values()
        ),
        coordinate_tokens_with_multiple_names=sum(
            len(v) > 1 for v in coordinates_to_names.values()
        ),
        source_mixed_motu_contexts=sum(len(g["source_motu_counts"]) > 1 for g in groups),
        date_kinds=dict(sorted(Counter(link["date_kind"] for link in links).items())),
        context_descriptors_are_independence_assertions=False,
    )
    return links, groups, summary


def trait_profiles(rows):
    profiles = {}
    for name in TRAITS:
        values = [r["numeric_values"][name] for r in rows if r["numeric_values"][name] is not None]
        strata = defaultdict(lambda: dict(rows=0, missing_or_invalid=0))
        for row in rows:
            for kind in ("sex", "family"):
                key = kind + ":" + row["values"][kind]
                strata[key]["rows"] += 1
                strata[key]["missing_or_invalid"] += row["numeric_values"][name] is None
        profiles[name] = dict(
            rows=len(rows),
            finite_values=len(values),
            missing=sum(r["values"][name] in MISSING for r in rows),
            invalid=sum(name in r["numeric_issues"] for r in rows),
            zero=sum(v == 0 for v in values),
            negative=sum(v < 0 for v in values),
            minimum=min(values) if values else None,
            maximum=max(values) if values else None,
            mean=statistics.mean(values) if values else None,
            median=statistics.median(values) if values else None,
            population_sd=statistics.pstdev(values) if values else None,
            strata=dict(sorted(strata.items())),
        )
    return profiles
