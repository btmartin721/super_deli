"""Load declared rival specimen partitions from verified or analyst-supplied assignments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from super_deli.io.config import decode_json
from super_deli.io.delimited import identifier

from .archive_io import assignment_rows, id_index, require
from .boundary import groups_by_label


@dataclass(frozen=True)
class Hypothesis:
    source: dict
    assignments: Mapping[str, dict]

    @property
    def labels(self) -> dict[str, str | None]:
        return {unit: row.get("assignment") for unit, row in self.assignments.items()}


def _named_records(records, name):
    require(
        isinstance(records, list)
        and bool(records)
        and all(isinstance(row, dict) and "id" in row for row in records),
        f"Expected nonempty {name} with unique IDs",
    )
    return id_index(records, "id", name)


def read_hypothesis_manifest(path):
    path = Path(path).absolute()
    payload = path.read_bytes()
    manifest = decode_json(payload)
    require(
        isinstance(manifest, dict)
        and type(manifest.get("schema_version")) is int
        and manifest["schema_version"] == 1,
        "Unsupported hypothesis manifest schema",
    )
    require(
        manifest.get("target_level") == "specimen",
        "Unsupported target level; the sealed matched-multimodal adapter compares specimens",
    )
    require(
        manifest.get("cohort_policy") == "exact",
        "Only the exact cohort policy is supported",
    )
    declarations = _named_records(manifest.get("hypotheses"), "hypotheses")
    contrasts = _named_records(manifest.get("contrasts"), "contrasts")
    for name, record in contrasts.items():
        for side in ("left", "right"):
            ref = record.get(side)
            require(
                isinstance(ref, str) and ref in declarations,
                f"Contrast {name!r}: unknown {side} hypothesis {ref!r}",
            )
        require(
            record["left"] != record["right"],
            f"Contrast {name!r}: use two hypothesis IDs",
        )
    return (
        declarations,
        contrasts,
        {
            "path": str(path),
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
            "schema_version": manifest["schema_version"],
        },
    )


def _table_source(source, base_dir, roster):
    filename, origin = source.get("path"), source.get("origin")
    require(
        isinstance(filename, str) and bool(filename.strip()),
        "Assignment table needs a path",
    )
    require(
        isinstance(origin, str) and bool(origin.strip()),
        "Assignment table needs an origin",
    )
    relative = Path(filename)
    require(not relative.is_absolute(), "Assignment table path must be manifest-relative")
    data = (base_dir / relative).read_bytes()
    rows = assignment_rows(data, {"unit_id", "group_id", "status"}, filename)
    require(
        rows.keys() == roster,
        f"{filename}: cohort mismatch; missing={sorted(roster - rows.keys())!r}, "
        f"foreign={sorted(rows.keys() - roster)!r}",
    )
    assignments = {}
    for unit, row in rows.items():
        status, group = row["status"], row["group_id"]
        require(
            status in {"resolved", "unresolved", "abstained", "failed"},
            f"{filename}: invalid assignment status for {unit!r}: {status!r}",
        )
        if status == "resolved":
            identifier(group, f"{filename} unit {unit!r}")
        else:
            require(
                not group,
                f"{filename}: unresolved unit {unit!r} must have an empty group",
            )
        assignments[unit] = {
            "assignment": group or None,
            "status": status,
            "reason": None if group else f"Analyst table status: {status}",
        }
    return Hypothesis(
        {
            "kind": "assignment_table",
            "path": filename,
            "origin": origin,
            "bytes": len(data),
            "sha256": sha256(data).hexdigest(),
        },
        assignments,
    )


def _merge_source(name, source, parent):
    labels = source.get("groups")
    require(
        isinstance(labels, list)
        and len(labels) == 2
        and all(isinstance(label, str) for label in labels)
        and labels[0] != labels[1],
        f"Hypothesis {name!r}: a merge needs two distinct group labels",
    )
    groups = groups_by_label(parent.labels.items())
    require(
        all(label in groups for label in labels),
        f"Hypothesis {name!r}: a merge needs two nonempty resolved parent groups",
    )
    new_label = f"{labels[0]} + {labels[1]}"
    while new_label in groups:
        new_label += " (merged)"
    members = groups[labels[0]] | groups[labels[1]]
    return Hypothesis(
        {
            "kind": "merge",
            "parent": source["parent"],
            "groups": labels,
            "parent_member_sets": {label: sorted(groups[label]) for label in labels},
            "merged_group": new_label,
        },
        {
            unit: {**row, "assignment": new_label} if unit in members else row
            for unit, row in parent.assignments.items()
        },
    )


def resolve_hypotheses(declarations, base_dir, archive_arms, roster):
    resolved = {}

    def source_for(name):
        source = declarations[name].get("source")
        require(isinstance(source, dict), f"Hypothesis {name!r}: expected a source object")
        return source

    for name in declarations:
        pending = {}
        current = name
        while current not in resolved:
            require(
                current not in pending,
                f"Cyclic merge parents at hypothesis {current!r}",
            )
            source = source_for(current)
            kind = source.get("kind")
            if kind == "archive_arm":
                arm = source.get("arm")
                require(
                    isinstance(arm, str) and arm in archive_arms,
                    f"Hypothesis {current!r}: unknown archive arm {arm!r}",
                )
                resolved[current] = archive_arms[arm]
                break
            if kind == "assignment_table":
                resolved[current] = _table_source(source, base_dir, roster)
                break
            require(
                kind == "merge",
                f"Hypothesis {current!r}: unsupported source kind {kind!r}",
            )
            parent = source.get("parent")
            require(
                isinstance(parent, str) and parent in declarations,
                f"Hypothesis {current!r}: unknown merge parent {parent!r}",
            )
            pending[current] = source
            current = parent
        for child, source in reversed(pending.items()):
            resolved[child] = _merge_source(child, source, resolved[source["parent"]])
    return {name: resolved[name] for name in declarations}
