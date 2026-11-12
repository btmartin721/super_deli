"""Separate ID-bound reference labels; no labels enter analytical feature tables."""

import csv
from collections import Counter
from pathlib import Path


def validate_reference_labels(unit_ids, labels):
    if not isinstance(labels, dict) or set(labels) != set(unit_ids):
        raise ValueError("Exactly one reference label for every training unit is required")
    values = [labels[u] for u in unit_ids]
    if any(not isinstance(v, str) or not v or v != v.strip() or len(v) > 256 for v in values):
        raise ValueError(
            "Reference labels must be nonempty unpadded strings of at most 256 characters"
        )
    counts = Counter(values)
    if not 2 <= len(counts) <= 32 or min(counts.values()) < 2:
        raise ValueError("Require 2–32 reference classes and at least two specimens per class")
    classes = tuple(sorted(counts))
    return classes, tuple(classes.index(v) for v in values), dict(sorted(counts.items()))


def read_reference_labels(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != 2
            or set(reader.fieldnames) != {"unit_id", "reference_label"}
        ):
            raise ValueError("Reference CSV requires exactly unit_id,reference_label columns")
        result = {}
        for row in reader:
            unit, label = row.get("unit_id"), row.get("reference_label")
            if (
                None in row
                or not unit
                or unit != unit.strip()
                or unit in result
                or not label
                or label != label.strip()
            ):
                raise ValueError("Invalid, duplicate or incomplete reference-label row")
            result[unit] = label
    if not result:
        raise ValueError("Reference labels are empty")
    return result
