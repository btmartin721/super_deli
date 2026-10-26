"""Literal reference-table reconstruction and purpose-specific evidence accounting."""

import hashlib
import re
from collections import Counter, defaultdict

from super_deli.core.trait_evidence import TABLES
from super_deli.io.config import read_json
from super_deli.io.evidence_workbook import inspect_reference_workbook
from super_deli.io.trait_source import ACCESSION, MISSING

REFERENCE_MISSING = frozenset((*MISSING, "#N/A", "-"))
HEADERS = {
    "reference01": (
        "SequenceName",
        "ChosenHaplotype",
        "Raw sequence",
        "GenBank accesion number",
        "Sequence origin",
        "Sequence Laboratory origin",
        "PTP species delimitation",
        "ASAP species delimitation",
        "N. rhenorhodanensis species name",
        "N. rhenorhodanensis clade number",
        "Locality name",
        "Country of origin",
        "Latitude",
        "Longitude",
        "Legator",
        "Date of Sampling",
        "MOTU_SubBioLab",
        "Swiss ID",
    ),
    "reference02": (
        "voucher",
        "chimera",
        "MOTU",
        "DNA_source",
        "28S right (lev3des5)",
        "COI",
        "28S left (lev2des2)",
        "HSP70",
        "H3",
        "EPRS",
        "ArgK",
        "18S",
        "locality/locality-chimera",
        "Country",
        "WGS84 - X",
        "WGS84 - Y",
    ),
    "reference03": (
        "SequenceNAME",
        "ChosenHaplotype",
        "GenBank accesion number",
        "PTP species",
        "ASAP species",
        "N. rhenorhodanensis_name",
        "N. rhenorhodanensis_clade",
        "locality name",
        "country",
        "latitude",
        "longitude",
        "MOTU",
        "GenBank",
    ),
    "reference05": (
        "Syntopy No.",
        "SequenceNAME",
        "GenBank acc. No.",
        "locality name",
        "country",
        "latitude",
        "longitude",
        "MOTU",
    ),
    "reference07": ("Delimitation PTP", "Delimitation ASAP", "Accepted MOTU for analyses", "Notes"),
}


ACCESSION_FIELDS = {
    "reference01": "GenBank accesion number",
    "reference02": "COI",
    "reference03": "GenBank accesion number",
    "reference05": "GenBank acc. No.",
}


def verify_source_inventories(paths):
    if "reference_archive" in paths:
        metadata = read_json(paths["reference_archive"])
        files = metadata.get("files") if isinstance(metadata, dict) else None
        if (
            not isinstance(files, list)
            or any(
                not isinstance(f, dict) or not isinstance(f.get("key"), str) or not f["key"]
                for f in files
            )
            or len({f.get("key") for f in files}) != len(files)
        ):
            raise ValueError("Unique published molecular archive inventory required")
        for source_id, path in paths.items():
            if not re.fullmatch(r"reference[0-9]{2}", source_id):
                continue
            key = "SupportingInformation_" + source_id[-2:] + path.suffix
            matches = [f for f in files if f.get("key") == key]
            raw = path.read_bytes()
            if (
                len(matches) != 1
                or matches[0].get("size") != len(raw)
                or matches[0].get("checksum")
                != "md5:" + hashlib.md5(raw, usedforsecurity=False).hexdigest()
            ):
                raise ValueError("Published molecular inventory differs: " + source_id)
    for source_id in ("tree", "regional", "summary", "usage"):
        if source_id in paths and source_id + "_meta" in paths:
            metadata = read_json(paths[source_id + "_meta"])
            files = metadata.get("files") if isinstance(metadata, dict) else None
            raw = paths[source_id].read_bytes()
            if (
                not isinstance(files, list)
                or len(files) != 1
                or not isinstance(files[0], dict)
                or files[0].get("size") != len(raw)
                or files[0].get("computed_md5")
                != hashlib.md5(raw, usedforsecurity=False).hexdigest()
            ):
                raise ValueError("Published contextual inventory differs: " + source_id)


def reference_tables(paths):
    books, tables = {}, {}
    for source_id, (sheet, header_row) in TABLES.items():
        if source_id not in paths:
            tables[source_id] = []
            continue
        book = inspect_reference_workbook(paths[source_id])
        books[source_id] = book
        rows = book["sheets"].get(sheet)
        if rows is None:
            raise ValueError("Missing declared reference sheet: " + source_id)
        header = next((r for r in rows if r["row"] == header_row), None)
        if header is None:
            raise ValueError("Missing explicit reference header")
        columns = {c: v["value"] for c, v in header["cells"].items() if v["value"]}
        if tuple(columns.values()) != HEADERS[source_id]:
            raise ValueError(
                "Reference columns differ from the reviewed source profile: " + source_id
            )
        if any(c["formula"] is not None for c in header["cells"].values()):
            raise ValueError("Reference headers cannot be formula derived")
        records = []
        for row in rows:
            if row["row"] <= header_row:
                continue
            if any(
                c not in columns and (v["value"] or v["formula"] is not None)
                for c, v in row["cells"].items()
            ):
                raise ValueError("Undeclared populated reference column")
            values = {name: row["cells"].get(c, {}).get("value", "") for c, name in columns.items()}
            cells = {name: row["cells"].get(c, {}) for c, name in columns.items()}
            records.append(
                dict(
                    source_id=source_id,
                    sheet=sheet,
                    source_row=row["row"],
                    record_id=f"{source_id}:{sheet}!{row['row']}",
                    values=values,
                    cells=cells,
                    row_attributes=row["attributes"],
                )
            )
        tables[source_id] = records
    return books, tables


def accession_base(value):
    return value.split(".")[0] if ACCESSION.fullmatch(value) else None


def _index(records, field, *, accession=False):
    result = defaultdict(list)
    for row in records:
        token = row["values"][field]
        key = (
            accession_base(token)
            if accession
            else token
            if token not in REFERENCE_MISSING
            else None
        )
        if key is not None:
            result[key].append(row)
    return result


def _key_fields_clean(record, names):
    return all(
        record["cells"].get(n, {}).get("formula") is None
        and record["cells"].get(n, {}).get("type") not in {"b", "d", "e"}
        for n in names
    )


def partition_context(records):
    forward, reverse = defaultdict(set), defaultdict(set)
    for row in records:
        a, b = (row["values"][n] for n in ("PTP species delimitation", "ASAP species delimitation"))
        if a not in REFERENCE_MISSING and b not in REFERENCE_MISSING:
            forward[a].add(b)
            reverse[b].add(a)
    return forward, reverse


def nexus_context(path, regional):
    if path is None:
        return dict(status="unavailable", tips=[], regional_labels=[])
    text = path.read_text(encoding="utf-8-sig")
    matches = re.findall(r"\bTaxlabels\s+(.*?);", text, re.I | re.S)
    size = re.findall(r"\bntax\s*=\s*([0-9]+)", text, re.I)
    if len(matches) != 1 or not size or len(set(size)) != 1:
        raise ValueError("Unique declared NEXUS taxa context required")
    tips = matches[0].split()
    if (
        len(tips) != int(size[0])
        or len(tips) != len(set(tips))
        or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in tips)
    ):
        raise ValueError("Unsupported or duplicate NEXUS context label")
    labels = []
    if regional is not None:
        lines = regional.read_text(encoding="utf-8-sig").splitlines()
        if not lines or lines[0] != "MOTU":
            raise ValueError("Explicit regional MOTU header required")
        labels = lines[1:]
        if (
            not labels
            or len(labels) != len(set(labels))
            or any(x != x.strip() or not x for x in labels)
        ):
            raise ValueError("Invalid regional context labels")
    return dict(
        status="label_context_only",
        tips=tips,
        regional_labels=labels,
        tree_inferred=False,
        specimen_crosswalk=False,
    )


def source_accounting(tables):
    result = {}
    for name, rows in tables.items():
        field = ACCESSION_FIELDS.get(name)
        counts = Counter(accession_base(r["values"][field]) for r in rows) if field else Counter()
        result[name] = dict(
            records=len(rows),
            valid_accession_rows=sum(n for key, n in counts.items() if key is not None),
            duplicate_accession_keys=sum(n > 1 for key, n in counts.items() if key is not None),
            hidden_rows=sum(r["row_attributes"].get("hidden") == "1" for r in rows),
            formula_cells=sum(
                c.get("formula") is not None for r in rows for c in r["cells"].values()
            ),
        )
    result["reference02"]["composite_rows"] = sum(
        r["values"]["chimera"] not in REFERENCE_MISSING for r in tables["reference02"]
    )
    return result


def assess_records(rows, old_links, locations, tables, context):
    """A sequence-name token alone never certifies a specimen crosswalk."""
    by_table = {
        key: _index(tables[key], field, accession=True) for key, field in ACCESSION_FIELDS.items()
    }
    by_voucher = _index(tables["reference02"], "voucher")
    by_name = _index(tables["reference01"], "SequenceName")
    partitions = partition_context(tables["reference01"])
    links = {r["source_row"]: r for r in old_links}
    localities = {r["source_row"]: r for r in locations}
    tips, regional = set(context["tips"]), set(context["regional_labels"])
    output = []
    for row in rows:
        values = row["values"]
        voucher, token = values["voucher"], values["COI_sequence"]
        accession = accession_base(token)
        hits = {
            key: index.get(accession, []) if accession else [] for key, index in by_table.items()
        }
        explicit = hits["reference02"]
        version_issues = [
            r["record_id"]
            for key, records in hits.items()
            for r in records
            if "." in token
            and "." in r["values"][ACCESSION_FIELDS[key]]
            and token != r["values"][ACCESSION_FIELDS[key]]
        ]
        contradictory = [
            r["record_id"]
            for r in explicit
            if r["values"]["voucher"] not in REFERENCE_MISSING and r["values"]["voucher"] != voucher
        ]
        pair = len(explicit) == 1 and explicit[0]["values"]["voucher"] == voucher
        composite = [
            r["record_id"] for r in explicit if r["values"]["chimera"] not in REFERENCE_MISSING
        ]
        clean_pair = (
            pair and not composite and _key_fields_clean(explicit[0], ("voucher", "COI", "chimera"))
        )
        ambiguous = [
            key for key in ("reference01", "reference02", "reference03") if len(hits[key]) > 1
        ]
        old = links[row["source_row"]]
        genbank_conflict = old["status"] in {
            "voucher_disagreement",
            "version_disagreement",
            "marker_not_coi",
            "ambiguous_record",
        }
        if row["identity_status"] != "unique":
            status = "unresolved_source_identity"
        elif row["reference_cautions"]:
            status = "withheld_source_caution"
        elif not accession:
            status = "no_valid_stated_accession"
        elif version_issues or contradictory or genbank_conflict:
            status = "conflicting_documentary_evidence"
        elif ambiguous:
            status = "ambiguous_documentary_accession"
        elif old["status"] == "literal_voucher_coi_match":
            status = "corroborated_genbank_voucher"
        elif clean_pair:
            status = "corroborated_explicit_table_pair"
        else:
            status = "source_asserted_accession_only"
        verified = status in {"corroborated_genbank_voucher", "corroborated_explicit_table_pair"}
        main = hits["reference01"][0] if len(hits["reference01"]) == 1 else None
        raw_partitions = (
            None
            if main is None
            else {
                "ptp": main["values"]["PTP species delimitation"],
                "asap": main["values"]["ASAP species delimitation"],
            }
        )
        partition_relation = "unavailable"
        if raw_partitions is not None and all(
            x not in REFERENCE_MISSING for x in raw_partitions.values()
        ):
            a, b = raw_partitions["ptp"], raw_partitions["asap"]
            partition_relation = (
                "one_to_one"
                if len(partitions[0][a]) == len(partitions[1][b]) == 1
                else "split_or_merge_context"
            )
        partition_link = (
            verified
            and main is not None
            and partition_relation != "unavailable"
            and _key_fields_clean(
                main,
                (
                    "GenBank accesion number",
                    "PTP species delimitation",
                    "ASAP species delimitation",
                ),
            )
        )
        basis = []
        remarks = values["remarks"].casefold()
        if "type locality" in remarks and "not by dna" in remarks:
            basis.append("explicit_type_locality_assignment")
        if "detailed morphological study" in remarks:
            basis.append("explicit_morphological_assignment")
        if accession:
            basis.append("source_stated_individual_accession")
        if not basis:
            basis.append(
                "no_historical_motu_assignment_stated"
                if values["MOTU"] in MISSING
                else "specimen_assignment_basis_not_documented"
            )
        comparisons = []
        for ref in hits["reference01"]:
            v = ref["values"]
            comparisons.append(
                dict(
                    record_id=ref["record_id"],
                    sequence_name_token_equals_voucher=v["SequenceName"] == voucher,
                    token_agreement={
                        n: None
                        if values[n] in MISSING or v[field] in REFERENCE_MISSING
                        else values[n] == v[field]
                        for n, field in {
                            "locality_name": "Locality name",
                            "latitude": "Latitude",
                            "longitude": "Longitude",
                            "survey_date": "Date of Sampling",
                        }.items()
                    },
                    locality_identity_asserted=False,
                )
            )
        output.append(
            dict(
                source_row=row["source_row"],
                voucher=voucher,
                unit_id=row["unit_id"],
                stated_accession=token,
                previous_reference_status=old["status"],
                correspondence_status=status,
                documented_specimen_link=verified,
                matched_records={key: [r["record_id"] for r in rs] for key, rs in hits.items()},
                voucher_only_candidates=[r["record_id"] for r in by_voucher.get(voucher, [])]
                if voucher not in MISSING
                else [],
                sequence_name_only_candidates=[r["record_id"] for r in by_name.get(voucher, [])]
                if voucher not in MISSING
                else [],
                genbank_conflict=genbank_conflict,
                contradictory_voucher_records=contradictory,
                version_conflicts=version_issues,
                ambiguous_tables=ambiguous,
                composite_multilocus_records=composite,
                source_basis=basis,
                source_cautions=row["reference_cautions"],
                historical_motu=values["MOTU"],
                historical_taxon=values["species"],
                raw_molecular_partitions=raw_partitions,
                molecular_partition_link=partition_link,
                partition_relation=partition_relation,
                source_context_comparisons=comparisons,
                locality_context=localities[row["source_row"]],
                tree_motu_label_present=values["MOTU"] in tips,
                regional_motu_label_present=values["MOTU"] in regional,
                independent_sampling_unit_asserted=False,
                reference_species_truth_asserted=False,
                new_genbank_voucher_qualifier_asserted=False,
                required_evidence=(
                    "Source-issued adjudication of conflicting explicit voucher/accession claims"
                    if status == "conflicting_documentary_evidence"
                    else "An unambiguous source-issued accession/voucher crosswalk"
                    if not verified
                    else "Independent sampling design and specimen-linked reference adjudication"
                ),
            )
        )
    return output
