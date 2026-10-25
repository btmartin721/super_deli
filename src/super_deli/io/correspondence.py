"""Narrow documentary crosswalk and official haul inspection adapters."""

import xml.etree.ElementTree as ET
from pathlib import Path

from super_deli.io.delimited import delimited_rows, identifier

HAUL_KEY = ("Survey", "Year", "Quarter", "Country", "Ship", "Gear", "SweepLngt", "StNo", "HaulNo")
VOUCHER_COLUMNS = ("specimen_id", "photo_id", "voucher", "accession_version")
HAUL_COLUMNS = ("specimen_id", "photo_id", "fish_id", "cruise_id", *HAUL_KEY)
NS = "{ices.dk.local/DATRAS}"


def xml_root(path):
    raw = Path(path).read_bytes()
    if len(raw) > 8 * 1024**2 or b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("Bounded XML without DTD/entity declarations required")
    return ET.fromstring(raw)


def haul_key(row):
    # Official fixed-width character fields can be padded. Preserve the complete
    # token: stripping would create an undeclared identity transformation.
    values = tuple(row[n] for n in HAUL_KEY)
    if any(not isinstance(v, str) or not v.strip() or "\x00" in v for v in values):
        raise ValueError("Complete literal composite haul tokens required")
    return values


def haul_catalogue(path):
    root = xml_root(path)
    if root.tag != NS + "ArrayOfCls_DatrasExchange_HH":
        raise ValueError("Expected official DATRAS HH exchange XML")
    rows = {}
    for element in root:
        if element.tag != NS + "Cls_DatrasExchange_HH":
            raise ValueError("Unexpected haul record")
        names = [x.tag.removeprefix(NS) for x in element]
        if len(names) != len(set(names)) or any(not x.tag.startswith(NS) for x in element):
            raise ValueError("Duplicate or foreign haul field")
        row = {n: x.text or "" for n, x in zip(names, element, strict=True)}
        if not set(HAUL_KEY) <= row.keys() or row.get("RecordType") != "HH":
            raise ValueError("Incomplete haul identity")
        key = haul_key(row)
        if row["Survey"] != "IE-IGFS" or row["Year"] != "2021" or row["Quarter"] != "4":
            raise ValueError("Outside the declared IE-IGFS 2021 Q4 survey scope")
        if key in rows:
            raise ValueError("Duplicate composite haul identity")
        rows[key] = row
    if not rows:
        raise ValueError("An empty catalogue cannot verify haul correspondence")
    return rows


def field_dictionary(path):
    root = xml_root(path)
    if root.tag != NS + "ArrayOfCls_Datras_FieldList":
        raise ValueError("Expected official DATRAS field dictionary")
    rows = []
    for element in root:
        row = {x.tag.removeprefix(NS): x.text or "" for x in element}
        if (
            element.tag != NS + "Cls_Datras_FieldList"
            or len(row) != len(element)
            or set(row)
            != {"RecordHeader", "FieldName", "FieldNameOld", "DataFormat", "Description"}
        ):
            raise ValueError("Invalid field-dictionary record")
        rows.append(row)
    if not rows:
        raise ValueError("Empty field dictionary")
    return rows


def crosswalk_rows(path, role):
    if role not in ("voucher_crosswalk", "haul_crosswalk"):
        raise ValueError("Unsupported documentary crosswalk role")
    columns = VOUCHER_COLUMNS if role == "voucher_crosswalk" else HAUL_COLUMNS
    result = []
    for line, row in delimited_rows(Path(path), delimiter=",", required=set(columns), context=role):
        identifier(row["specimen_id"], "crosswalk specimen")
        for name, token in row.items():
            if token and name not in HAUL_KEY:
                identifier(token, "crosswalk " + name)
        required = ("voucher", "accession_version") if role == "voucher_crosswalk" else HAUL_KEY
        if any(not row[x] for x in required):
            raise ValueError("Required correspondence target is missing")
        if role == "haul_crosswalk":
            haul_key(row)
        result.append(dict(row=row, line=line))
    return result
