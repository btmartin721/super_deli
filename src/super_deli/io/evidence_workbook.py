"""Stream bounded reference worksheets; formatting rows are not specimens.

This separate profile shares the original cell decoder, without changing the
amphipod source-audit reader or evaluating workbook formulas/links.
"""

import codecs
import re
import stat
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

from super_deli.io.audit_workbook import P, R, S, _sheet, _text, xml_bytes


def _events(handle):
    parser = ET.XMLPullParser(events=("start", "end"))
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    total, carry = 0, ""
    while raw := handle.read(64 * 1024):
        total += len(raw)
        if total > 96 * 1024**2:
            raise ValueError("Reference XML exceeds 96 MiB")
        chunk = decoder.decode(raw)
        scan = carry + chunk
        if "\x00" in scan or "<!DOCTYPE" in scan.upper() or "<!ENTITY" in scan.upper():
            raise ValueError("Reference XML must be UTF-8 without DTD/entities")
        carry = scan[-16:]
        parser.feed(chunk)
        yield from parser.read_events()
    parser.feed(decoder.decode(b"", final=True))
    parser.close()
    yield from parser.read_events()


def inspect_reference_workbook(path):
    try:
        return _inspect(path)
    except (BadZipFile, KeyError, UnicodeError, ET.ParseError, RecursionError) as exc:
        raise ValueError("Malformed reference workbook: " + str(exc)) from exc


def _inspect(path):
    with ZipFile(path) as archive:
        members = archive.infolist()
        if (
            len(members) > 4096
            or len({i.filename for i in members}) != len(members)
            or sum(i.file_size for i in members) > 128 * 1024**2
        ):
            raise ValueError("Reference ZIP inventory exceeds bounds or repeats members")
        for item in members:
            name = PurePosixPath(item.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in item.filename
                or "\x00" in item.filename
                or item.flag_bits & 1
                or stat.S_IFMT(item.external_attr >> 16) == stat.S_IFLNK
                or item.filename.lower().endswith((".exe", ".js"))
                or (
                    item.filename.lower().endswith(".bin")
                    and not item.filename.startswith("xl/printerSettings/")
                )
            ):
                raise ValueError("Unsafe reference ZIP member")
        workbook = xml_bytes(archive.read("xl/workbook.xml"))
        rels = xml_bytes(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {}
        for rel in rels:
            target = rel.get("Target", "")
            if (
                rel.tag != P + "Relationship"
                or not rel.get("Id")
                or rel.get("Id") in targets
                or rel.get("TargetMode") == "External"
                or not target
                or PurePosixPath(target).is_absolute()
                or ".." in PurePosixPath(target).parts
                or "\\" in target
            ):
                raise ValueError("Unsafe reference workbook relationship")
            targets[rel.get("Id")] = ("xl/" + target, rel.get("Type", ""))
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [_text(x) for x in xml_bytes(archive.read("xl/sharedStrings.xml"))]
        sheets, metadata = {}, []
        declared = workbook.find(S + "sheets")
        if workbook.tag != S + "workbook" or declared is None or not 1 <= len(declared) <= 8:
            raise ValueError("Expected one to eight declared reference worksheets")
        for sheet in declared:
            name = sheet.get("name", "")
            target = targets.get(sheet.get(R + "id"))
            if not name or name in sheets or target is None or not target[1].endswith("/worksheet"):
                raise ValueError("Missing, repeated or unexpected reference worksheet")
            rows, stack, physical, empty_ranges, last = [], [], 0, [], 0
            merged_ranges = []
            with archive.open(target[0]) as handle:
                for event, node in _events(handle):
                    if event == "start":
                        stack.append(node)
                        if len(stack) == 1 and node.tag != S + "worksheet":
                            raise ValueError("Expected reference worksheet root")
                        continue
                    if node.tag == S + "row":
                        number = node.get("r", "")
                        if (
                            not re.fullmatch(r"[1-9][0-9]*", number)
                            or not last < int(number) <= 1048576
                            or len(stack) < 2
                            or stack[-2].tag != S + "sheetData"
                        ):
                            raise ValueError("Invalid reference row coordinate or ordering")
                        last = int(number)
                        physical += 1
                        populated = any(
                            c.find(S + x) is not None for c in node for x in ("v", "is", "f")
                        )
                        if populated:
                            if len(rows) >= 10000:
                                raise ValueError("Reference source exceeds 10,000 populated rows")
                            # Reuse the strict original cell/cache decoder for populated rows.
                            wrapper = (
                                f'<worksheet xmlns="{S[1:-1]}"><sheetData>'.encode()
                                + ET.tostring(node)
                                + b"</sheetData></worksheet>"
                            )
                            decoded = _sheet(wrapper, strings)[0]
                            decoded["attributes"] = dict(node.attrib)
                            rows.append(decoded)
                        elif empty_ranges and empty_ranges[-1][1] == last - 1:
                            empty_ranges[-1][1] = last
                        else:
                            empty_ranges.append([last, last])
                        stack[-2].remove(node)
                        node.clear()
                    elif node.tag == S + "mergeCell":
                        merged_ranges.append(node.get("ref", ""))
                    stack.pop()
            sheets[name] = rows
            metadata.append(
                dict(
                    sheet=name,
                    serialized_rows=physical,
                    populated_rows=len(rows),
                    empty_rows=physical - len(rows),
                    empty_row_ranges=empty_ranges,
                    merged_ranges=merged_ranges,
                )
            )
        props = workbook.find(S + "workbookPr")
        styles = (
            xml_bytes(archive.read("xl/styles.xml"))
            if "xl/styles.xml" in archive.namelist()
            else None
        )
        return dict(
            sheets=sheets,
            worksheets=metadata,
            date1904=props.get("date1904", "0") if props is not None else "0",
            styles_xml=ET.tostring(styles, encoding="unicode") if styles is not None else None,
            zip_members=[dict(name=i.filename, size_bytes=i.file_size) for i in members],
            formula_evaluation=False,
            external_link_following=False,
        )
