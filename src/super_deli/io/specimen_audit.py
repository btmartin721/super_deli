"""Conservative source inspection, not a new production landmark/GenBank reader.

TPS curve sections are counted and retained, never treated as fixed landmarks.
The fixed-landmark adapter later validates derivatives with the unchanged reader.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from super_deli.io.delimited import delimited_rows, identifier
from super_deli.io.numeric import finite_float

COLOUR = ("area_body", "area_band", "per_band", "mean_grey_body", "mean_grey_band", "per_dif_grey")


def number(token, location):
    if token != token.strip():
        raise ValueError("Padded numeric token: " + location)
    return finite_float(token, location)


def source_tables(metadata, colour):
    rows = {}
    for line, row in delimited_rows(
        Path(metadata),
        delimiter=",",
        required={"ID", "length", "sex", "species"},
        context="metadata",
    ):
        unit = identifier(row["ID"], str(line))
        if unit in rows or number(row["length"], str(line)) <= 0:
            raise ValueError("Duplicate specimen ID or invalid source length")
        rows[unit] = row
    colours = {}
    for line, row in delimited_rows(
        Path(colour), delimiter=",", required={"Id", *COLOUR}, context="colour"
    ):
        unit = identifier(row["Id"], str(line))
        if unit in colours:
            raise ValueError("Duplicate colour specimen ID")
        for name in COLOUR:
            if row[name] != "-":
                v = number(row[name], name)
                if (name == "area_body" and v <= 0) or (name == "area_band" and v < 0):
                    raise ValueError("Invalid pixel area")
                if name.startswith("mean_grey") and not 0 <= v <= 255:
                    raise ValueError("Greyscale outside 0..255")
        colours[unit] = row
    if not rows or not colours:
        raise ValueError("Nonempty source tables required")
    return rows, colours


def inspect_tps(path, *, landmarks, curves):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    result, i = [], 0

    def directive(key):
        nonlocal i
        if i >= len(lines) or not lines[i].startswith(key + "="):
            raise ValueError("Expected TPS directive " + key)
        value = lines[i].partition("=")[2]
        i += 1
        return value

    def points(n):
        nonlocal i
        values = []
        for _ in range(n):
            if i >= len(lines):
                raise ValueError("Truncated TPS coordinate block")
            tokens = lines[i].split()
            if len(tokens) != 2:
                raise ValueError("Exactly two TPS coordinate tokens required")
            for v in tokens:
                number(v, "TPS coordinate")
            values.append(tuple(tokens))
            i += 1
        return values

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if directive("LM") != str(landmarks):
            raise ValueError("Unexpected fixed-landmark count")
        fixed = points(landmarks)
        sections = []
        if curves:
            if directive("CURVES") != str(curves):
                raise ValueError("Unexpected curve count")
            for _ in range(curves):
                if directive("POINTS") != "8":
                    raise ValueError("This audit requires eight points per curve")
                sections.append(points(8))
        fields = {}
        while i < len(lines) and not lines[i].startswith("LM="):
            if not lines[i].strip():
                i += 1
                continue
            if "=" not in lines[i]:
                raise ValueError("Unexpected TPS payload after geometry")
            key, _, value = lines[i].partition("=")
            i += 1
            if key not in ("IMAGE", "ID", "VARIABLES", "SCALE") or key in fields or "\x00" in value:
                raise ValueError("Unsupported or duplicate TPS metadata")
            fields[key] = value
        if "IMAGE" not in fields:
            raise ValueError("Explicit IMAGE identity required")
        identifier(fields["IMAGE"], "TPS image identity")
        if "SCALE" in fields and number(fields["SCALE"], "TPS scale") <= 0:
            raise ValueError("TPS scale must be positive")
        result.append(dict(image=fields["IMAGE"], fixed=fixed, curves=sections, fields=fields))
    if not result or len({r["image"] for r in result}) != len(result):
        raise ValueError("Empty TPS source or duplicate image identity")
    return result


def genbank_references(path):
    root = ET.parse(path).getroot()
    if root.tag != "GBSet":
        raise ValueError("Expected NCBI GBSet XML")
    result, versions = [], set()
    for record in root:
        version = record.findtext("GBSeq_accession-version", "")
        accession = record.findtext("GBSeq_primary-accession", "")
        if (
            not re.fullmatch(r"[A-Z]+[0-9]+\.[0-9]+", version)
            or not version.startswith(accession + ".")
            or accession in versions
        ):
            raise ValueError("Missing or duplicate versioned reference accession")
        versions.add(accession)
        source = [
            f
            for f in record.findall("GBSeq_feature-table/GBFeature")
            if f.findtext("GBFeature_key") == "source"
        ]
        if len(source) != 1:
            raise ValueError("Exactly one GenBank source feature required")
        qualifiers = {}
        for q in source[0].findall("GBFeature_quals/GBQualifier"):
            name, value = q.findtext("GBQualifier_name"), q.findtext("GBQualifier_value", "")
            if name in ("organism", "specimen_voucher", "geo_loc_name", "collection_date"):
                if name in qualifiers:
                    raise ValueError("Ambiguous GenBank source qualifier")
                qualifiers[name] = value
        organism = record.findtext("GBSeq_organism", "")
        if not organism or organism != qualifiers.get("organism"):
            raise ValueError("Conflicting or absent GenBank organism")
        sequence = record.findtext("GBSeq_sequence", "")
        if (
            not sequence
            or not re.fullmatch(r"[acgtryswkmbdhvnACGTRYSWKMBDHVN]+", sequence)
            or str(len(sequence)) != record.findtext("GBSeq_length")
        ):
            raise ValueError("Invalid reference sequence/length")
        result.append(
            dict(
                accession=accession,
                version=version,
                organism=organism,
                voucher=qualifiers.get("specimen_voucher", ""),
                location=qualifiers.get("geo_loc_name", ""),
                collection_date=qualifiers.get("collection_date", ""),
                sequence_length=len(sequence),
            )
        )
    if not result:
        raise ValueError("No reference records")
    return result


def supplement_membership(path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as z:
        if z.namelist().count("word/document.xml") != 1:
            raise ValueError("Unique supplement document XML required")
        info = z.getinfo("word/document.xml")
        if info.file_size > 4 * 1024**2:
            raise ValueError("Supplement XML too large")
        root = ET.fromstring(z.read(info))
    matches = []
    for table in root.findall(".//w:tbl", ns):
        rows = [
            ["".join(c.itertext()) for c in row.findall("w:tc", ns)]
            for row in table.findall("w:tr", ns)
        ]
        if rows and rows[0] == ["ID", "Species", "Sex", "TL", "FD", "ADM", "HR", "VC"]:
            matches.append(rows[1:])
    if len(matches) != 1 or any(len(row) != 8 for row in matches[0]):
        raise ValueError("Unique source sample-membership table required")
    records = matches[0]
    if len({row[0] for row in records}) != len(records):
        raise ValueError("Duplicate supplement specimen")
    return records
