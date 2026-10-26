"""Bounded OOXML inspection; no formulas, macros, links or external code execute."""

import posixpath
import re
import stat
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
P = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def xml_bytes(raw):
    if len(raw) > 16 * 1024**2:
        raise ValueError("XML exceeds the 16 MiB bound")
    try:
        text = raw.decode("utf-8-sig")
        if "\x00" in text or "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            raise ValueError("UTF-8 XML without DTD/entity declarations required")
        return ET.fromstring(text)
    except (UnicodeError, ET.ParseError) as exc:
        raise ValueError("Malformed UTF-8 XML") from exc


def _text(element):
    if element is None:
        return ""
    if element.findall(".//" + S + "rPh"):
        raise ValueError("Phonetic string annotations require a separate source profile")
    return "".join(t.text or "" for t in element.iter(S + "t"))


def _sheet(raw, strings):
    root = xml_bytes(raw)
    if root.tag != S + "worksheet":
        raise ValueError("Expected OOXML worksheet")
    rows, seen = [], set()
    for node in root.findall(S + "sheetData/" + S + "row"):
        number = node.get("r", "")
        if not re.fullmatch(r"[1-9][0-9]*", number) or int(number) > 10000 or int(number) in seen:
            raise ValueError("Invalid, duplicate or out-of-bound worksheet row")
        seen.add(int(number))
        cells = {}
        for c in node:
            address = c.get("r", "")
            match = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]*)", address)
            if (
                c.tag != S + "c"
                or match is None
                or match[2] != number
                or match[1] in cells
                or len(c.findall(S + "v")) > 1
                or len(c.findall(S + "f")) > 1
            ):
                raise ValueError("Malformed, duplicate or mismatched worksheet cell")
            kind = c.get("t", "n")
            if kind not in ("n", "s", "inlineStr", "b", "d", "e", "str"):
                raise ValueError("Unsupported cell type")
            value = c.find(S + "v")
            scalar = value.text if value is not None else None
            decoded = scalar or ""
            if kind == "s":
                if (
                    scalar is None
                    or not scalar.isascii()
                    or not scalar.isdigit()
                    or int(scalar) >= len(strings)
                ):
                    raise ValueError("Invalid shared-string index")
                decoded = strings[int(scalar)]
            elif kind == "inlineStr":
                decoded = _text(c.find(S + "is"))
            elif kind == "b" and scalar not in ("0", "1", None):
                raise ValueError("Invalid stored boolean")
            f = c.find(S + "f")
            cells[match[1]] = dict(
                address=address,
                type=kind,
                scalar=scalar,
                value=decoded,
                style=c.get("s", "0"),
                formula=None if f is None else dict(text=f.text, attributes=dict(f.attrib)),
                cache_present=value is not None and scalar is not None,
            )
        rows.append(dict(row=int(number), cells=cells))
    if rows != sorted(rows, key=lambda r: r["row"]):
        raise ValueError("Worksheet rows must follow their explicit coordinates")
    return rows


def _inspect_workbook(path):
    with ZipFile(path) as z:
        members = z.infolist()
        if (
            len(members) > 4096
            or len({i.filename for i in members}) != len(members)
            or sum(i.file_size for i in members) > 32 * 1024**2
        ):
            raise ValueError("Duplicate members or workbook expansion bound exceeded")
        for info in members:
            name = PurePosixPath(info.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in info.filename
                or "\x00" in info.filename
                or info.flag_bits & 1
                or stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK
            ):
                raise ValueError("Unsafe or encrypted workbook member")
            binary = info.filename.lower().endswith(".bin")
            inert_print_settings = info.filename.startswith("xl/printerSettings/")
            if info.filename.lower().endswith((".exe", ".js")) or (
                binary and not inert_print_settings
            ):
                raise ValueError("Executable/macro workbook payload is outside this profile")
        workbook = xml_bytes(z.read("xl/workbook.xml"))
        if workbook.tag != S + "workbook":
            raise ValueError("Expected OOXML workbook")
        rels = xml_bytes(z.read("xl/_rels/workbook.xml.rels"))
        targets = {}
        for rel in rels:
            if rel.tag != P + "Relationship" or rel.get("Id") in targets:
                raise ValueError("Invalid workbook relationship")
            if rel.get("TargetMode") == "External":
                raise ValueError("External workbook relationships are outside this profile")
            target = rel.get("Target", "")
            if (
                not target
                or PurePosixPath(target).is_absolute()
                or ".." in PurePosixPath(target).parts
                or "\\" in target
            ):
                raise ValueError("Unsafe worksheet relationship target")
            targets[rel.get("Id")] = (posixpath.join("xl", target), rel.get("Type", ""))
        shared = (
            xml_bytes(z.read("xl/sharedStrings.xml"))
            if "xl/sharedStrings.xml" in z.namelist()
            else None
        )
        strings = [] if shared is None else [_text(x) for x in shared]
        styles = xml_bytes(z.read("xl/styles.xml")) if "xl/styles.xml" in z.namelist() else None
        result = {}
        sheet_node = workbook.find(S + "sheets")
        if sheet_node is None or len(sheet_node) != 2:
            raise ValueError("Exactly Data and Descriptions sheets required")
        for sheet in sheet_node:
            name = sheet.get("name")
            relationship = targets.get(sheet.get(R + "id"))
            if (
                name not in ("Data", "Descriptions")
                or name in result
                or relationship is None
                or not relationship[1].endswith("/worksheet")
            ):
                raise ValueError("Unexpected or duplicated declared worksheet")
            result[name] = _sheet(z.read(relationship[0]), strings)
        props = workbook.find(S + "workbookPr")
        return dict(
            sheets=result,
            date1904=props.get("date1904", "0") if props is not None else "0",
            styles_xml=None if styles is None else ET.tostring(styles, encoding="unicode"),
            members=[dict(name=i.filename, size_bytes=i.file_size) for i in members],
            formula_evaluation=False,
            external_link_following=False,
        )


def inspect_workbook(path):
    try:
        return _inspect_workbook(path)
    except (BadZipFile, KeyError, ET.ParseError, UnicodeError, RecursionError) as exc:
        raise ValueError("Malformed or unsupported source workbook: " + str(exc)) from exc
