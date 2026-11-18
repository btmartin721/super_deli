"""Compare declared rival partitions on the same verified specimen roster."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import comb
from pathlib import Path, PurePosixPath

from super_deli.core.partition_comparison import compare_partitions

from .archive import ArchiveReport, _fresh_output, _read_run, _write_report
from .archive_io import ArchiveError, SealedRun, require
from .boundary import (
    boundary_agreement,
    evidence,
    groups_by_label,
    pair_counts,
    recurrence,
)
from .comparison_html import render_comparison
from .hypothesis_manifest import (
    Hypothesis,
    read_hypothesis_manifest,
    resolve_hypotheses,
)

# Resampling descriptions emitted by the schema-1 native specimen workflows.
_SPECIMEN_RESAMPLING_UNITS = {
    "integrative": "Paired independent target units across all views",
    "shape": "independent biological unit; all landmarks retained together",
    "tabular": "Independent biological specimen; whole rows",
}


def _together(labels):
    return sum(comb(n, 2) for n in Counter(labels).values())


def _directions(cells):
    left, right = Counter(), Counter()
    for (a, b), size in cells.items():
        left[a] += size
        right[b] += size
    both = sum(comb(n, 2) for n in cells.values())
    return sum(comb(n, 2) for n in left.values()) - both, sum(
        comb(n, 2) for n in right.values()
    ) - both


def _coverage(hypothesis):
    groups = groups_by_label(hypothesis.labels.items())
    exclusions = defaultdict(list)
    for unit, row in sorted(hypothesis.assignments.items()):
        if row.get("assignment") is None:
            exclusions[row["status"], row["reason"]].append(unit)
    return {
        "declared_units": len(hypothesis.assignments),
        "resolved_units": sum(map(len, groups.values())),
        "resolved_groups": len(groups),
        "groups": [
            {"label": label, "size": len(members), "unit_ids": sorted(members)}
            for label, members in sorted(groups.items())
        ],
        "exclusions": [
            {"status": status, "reason": reason, "count": len(units), "unit_ids": units}
            for (status, reason), units in exclusions.items()
        ],
    }


def _verified_measurement_sources(archive, manifest):
    originals = {view["name"]: view for view in manifest["dataset"]["modalities"]}
    expected = {}
    for view in manifest["frozen_dataset"]["modalities"]:
        modality = view["name"]
        for role in ("path", "popmapfile"):
            snapshot = view.get(role)
            require(
                (snapshot is None) == (originals[modality].get(role) is None),
                f"Frozen modality {modality!r} disagrees with its declared {role} role",
            )
            if snapshot is not None:
                expected[modality, role] = snapshot

    inputs_by_view = defaultdict(list)
    seen = set()
    records = manifest["sources"]
    require(isinstance(records, list), "Expected a list of frozen measurement source records")
    for record in records:
        modality, role = key = record["modality"], record["role"]
        require(key in expected, f"Unexpected frozen measurement source {key!r}")
        require(key not in seen, f"Repeated frozen measurement source {key!r}")
        snapshot = record["snapshot"]
        require(
            snapshot == expected[key],
            f"Frozen measurement snapshot for {key!r} disagrees with its modality declaration",
        )
        relative = PurePosixPath(snapshot)
        require(
            not relative.is_absolute() and relative.name and relative.as_posix() == snapshot,
            f"Invalid frozen measurement snapshot for {key!r}: {snapshot!r}",
        )
        name = f"inputs/{snapshot}"
        archive.read(name)
        verified = archive.verified[name]
        source = record["source"]
        require(
            verified["sha256"] == source["sha256"]
            and type(source["size_bytes"]) is int
            and verified["bytes"] == source["size_bytes"],
            f"{name}: frozen input differs from the declared source identity",
        )
        inputs_by_view[modality].append({**record, "verified_snapshot": verified})
        seen.add(key)
    missing = expected.keys() - seen
    require(not missing, f"Missing frozen measurement sources: {sorted(missing)!r}")
    return inputs_by_view


def _archive_hypotheses(archive, run, manifest):
    inputs_by_view = _verified_measurement_sources(archive, manifest)
    configurations = {arm["name"]: arm for arm in manifest["analysis"]["arms"]}
    arms = {}
    for partition in run["partitions"]:
        name = partition["name"]
        arms[name] = Hypothesis(
            {
                "kind": "archive_arm",
                "arm": name,
                "run": run["run"],
                "study_manifest_sha256": archive.manifest_sha256,
                "configuration": configurations[name],
                "artifacts": [
                    archive.verified[path]
                    for path in archive.verified
                    if path in {"assignments.csv", "manifest.json"}
                    or path.startswith(f"arms/{name}/")
                ],
                "measurement_sources": {
                    view: inputs_by_view[view] for view in partition["modalities"]
                },
            },
            {row["unit_id"]: row["arms"][name] for row in run["specimens"]},
        )
    return arms


def _aligned_metrics(left, right, ids):
    def encode(labels):
        codes = {label: index for index, label in enumerate(sorted(set(labels) - {None}))}
        return [codes.get(label, -1) for label in labels]

    a, b = (encode([labels[unit] for unit in ids]) for labels in (left, right))
    return compare_partitions(a, b, [label >= 0 for label in a], [label >= 0 for label in b])


def _support(arm, triples, possible, source_sides, direct_sides):
    known = [(a, b, arm[unit]) for unit, (a, b) in triples.items() if arm[unit] is not None]
    left_only, right_only = _directions(Counter((a, b) for a, b, _ in known))
    joint = _together(known)
    left_and_arm = _together((a, c) for a, _, c in known) - joint
    right_and_arm = _together((b, c) for _, b, c in known) - joint
    total = left_only + right_only
    agrees_left = left_and_arm + right_only - right_and_arm
    reason = None
    if not possible:
        reason = "The candidates have no discriminating pairs among jointly resolved units."
    elif not total:
        reason = "This arm resolves no discriminating pair."
    return evidence(
        "available" if total else "unavailable",
        reason,
        jointly_resolved_units=len(known),
        jointly_resolved_pairs=comb(len(known), 2),
        comparable_left_only=left_only,
        comparable_right_only=right_only,
        supporting_left=pair_counts(agrees_left, total, possible),
        supporting_right=pair_counts(total - agrees_left, total, possible),
        left_fraction=agrees_left / total if total else None,
        right_fraction=(total - agrees_left) / total if total else None,
        source_for_candidates=source_sides,
        agreement_by_construction_for=direct_sides,
        interpretation=(
            "Agreement with a candidate taken directly from this arm is by construction."
            if direct_sides
            else "Agreement on the shared discovery specimens."
        ),
    )


def _counts_with_fraction(counts, reason):
    comparable = counts["comparable"]
    return {
        **counts,
        **evidence("available" if comparable else "unavailable", None if comparable else reason),
        "fraction": counts["count"] / comparable if comparable else None,
    }


def _focal_counts(labels, a, b):
    result = boundary_agreement(labels, a, b)
    result["cross_group_separation"] = _counts_with_fraction(
        result["cross_group_separation"], "The arm resolves no cross-group pairs."
    )
    result["within_group_cohesion"] = [
        _counts_with_fraction(counts, "The arm resolves fewer than two members of the group.")
        for counts in result["within_group_cohesion"]
    ]
    return result


def _stability(archive, partition, native, order, focal, pair):
    if not native:
        return evidence("unavailable", "The arm has no verified point partition.")
    stability = partition["stability"]
    name = partition["name"]
    family = partition["family"]
    if family == "genomic":
        return evidence(
            "unavailable",
            "The genomic archive resamples locus blocks and holds specimens fixed.",
            requested=stability["requested"],
            successful=stability["successful"],
            failed=len(stability["failures"]) if "failures" in stability else stability["failed"],
            covered=0,
            resampling_unit="locus block",
        )
    unit = stability.get("resampling_unit")
    if unit != _SPECIMEN_RESAMPLING_UNITS[family]:
        return evidence(
            "unavailable",
            "The saved resampling unit is incompatible with this specimen workflow.",
            requested=stability["requested"],
            successful=stability["successful"],
            failed=stability["failed"],
            covered=0,
            resampling_unit=unit,
        )
    result = recurrence(
        archive,
        name,
        stability,
        order,
        groups_by_label((u, row["candidate_cluster"]) for u, row in native.items()),
        focal,
        pair,
        allow_incomplete=True,
    )
    result["verified_projection_artifact"] = archive.verified.get(
        f"arms/{name}/bootstrap_replicates.json"
    )
    possible = result["requested"]
    missing = pair_counts(0, 0, possible)
    reason = result["reason"] or "Exact recurrence requires a projection on the entire roster."
    result["exact_group_recurrence"] = {
        label: _counts_with_fraction(counts, reason)
        for label, counts in result.get(
            "exact_group_recurrence", dict.fromkeys(focal, missing)
        ).items()
    }
    result["joint_exact_group_recurrence"] = _counts_with_fraction(
        result.get("joint_exact_group_recurrence", missing), reason
    )
    separation = result.get("specimen_pair_separation", {})
    result["specimen_pair_separation"] = {
        "all_projected_draws": _counts_with_fraction(
            separation.get("all_projected_draws", missing),
            "No saved projection resolves both specimens.",
        ),
        "both_specimens_omitted": _counts_with_fraction(
            separation.get("both_specimens_omitted", pair_counts(0, 0, 0)),
            "No saved projection resolves both specimens in a draw omitting both.",
        ),
        "draws_with_unknown_omission_status": result["missing"],
    }
    return result


def _boundary(left, right, cells, relation, left_id, right_id):
    if relation not in {"left_refines_right", "right_refines_left"}:
        return evidence(
            "unavailable",
            "This contrast does not identify one two-group merge boundary.",
        )
    if {u for u in left if left[u] is not None} != {u for u in right if right[u] is not None}:
        return evidence(
            "unavailable",
            "Resolution masks differ. The relation applies to the jointly resolved subset.",
        )
    flipped = relation == "right_refines_left"
    fine, split_id, merged_id = (right, right_id, left_id) if flipped else (left, left_id, right_id)
    by_coarse = defaultdict(list)
    for a, b in cells:
        key, member = (a, b) if flipped else (b, a)
        by_coarse[key].append(member)
    merged = [(group, sorted(members)) for group, members in by_coarse.items() if len(members) > 1]
    if len(merged) != 1 or len(merged[0][1]) != 2:
        return evidence(
            "unavailable",
            "The refinement combines more than one pair of resolved groups.",
        )
    group, labels = merged[0]
    groups = groups_by_label(fine.items())
    return evidence(
        "available",
        split_hypothesis=split_id,
        merged_hypothesis=merged_id,
        merged_group=group,
        groups=[{"label": label, "unit_ids": sorted(groups[label])} for label in labels],
        specimen_pair=[min(groups[label]) for label in labels],
        specimen_pair_selection="First specimen by ID in each focal group.",
    )


def _origin(hypothesis, candidates):
    source = candidates[hypothesis].source
    while source["kind"] == "merge":
        source = candidates[source["parent"]].source
    return source.get("arm")


def _contrast(contrast, candidates, arms, ids, partitions, native, orders, archive):
    left_id, right_id = contrast["left"], contrast["right"]
    left, right = candidates[left_id].labels, candidates[right_id].labels
    paired = {u: (left[u], right[u]) for u in ids if left[u] is not None and right[u] is not None}
    metrics = _aligned_metrics(left, right, ids)
    cells = Counter(paired.values())
    left_only, right_only = _directions(cells)
    comparable = metrics["joint_pairs"]
    relation = "indeterminate"
    if comparable:
        if not left_only and not right_only:
            relation = "equivalent"
        elif not left_only:
            relation = "left_refines_right"
        elif not right_only:
            relation = "right_refines_left"
        else:
            relation = "non_nested"
    else:
        metrics.update(status="unavailable", reason="Fewer than two jointly resolved units.")
    metrics.update(
        relation=relation,
        relation_scope="jointly_resolved_units",
        together_only_left=left_only,
        together_only_right=right_only,
        unchanged_pairs=comparable - left_only - right_only,
        member_overlap=[
            {"left_group": a, "right_group": b, "units": n} for (a, b), n in sorted(cells.items())
        ],
    )
    possible = left_only + right_only
    arms_labels = {name: arm.labels for name, arm in arms.items()}
    origin = {side: _origin(contrast[side], candidates) for side in ("left", "right")}
    scores = {
        name: _support(
            labels,
            paired,
            possible,
            [side for side, arm in origin.items() if arm == name],
            [
                side
                for side in ("left", "right")
                if candidates[contrast[side]].source.get("arm") == name
            ],
        )
        for name, labels in arms_labels.items()
    }
    focal = _boundary(left, right, cells, relation, left_id, right_id)
    stability = evidence("unavailable", focal["reason"])
    if focal["status"] == "available":
        sets = {row["label"]: set(row["unit_ids"]) for row in focal["groups"]}
        a, b = sets.values()
        focal["originating_arm"] = _origin(focal["split_hypothesis"], candidates)
        focal["arms"] = {name: _focal_counts(labels, a, b) for name, labels in arms_labels.items()}
        per_arm = {
            name: _stability(
                archive,
                partition,
                native[name],
                orders[name],
                sets,
                focal["specimen_pair"],
            )
            for name, partition in partitions.items()
        }
        stability = evidence(
            "available"
            if any(row["status"] == "available" for row in per_arm.values())
            else "unavailable"
            if any(row["status"] == "unavailable" for row in per_arm.values())
            else "not_computed",
            "Recurrence uses saved specimen projections per arm. The specimen-pair measures "
            "use their own projection and omission denominators.",
            arms=per_arm,
        )
    return {
        **contrast,
        "coverage": {
            "declared_units": len(ids),
            "declared_pairs": comb(len(ids), 2),
            "jointly_resolved_units": len(paired),
            "jointly_resolved_pairs": comparable,
            "jointly_resolved_unit_ids": list(paired),
            "excluded_unit_ids": [unit for unit in ids if unit not in paired],
            "possible_discriminating_pairs": possible,
        },
        "partition_comparison": metrics,
        "same_cohort_agreement": evidence(
            "available"
            if any(row["status"] == "available" for row in scores.values())
            else "unavailable",
            "The saved arms share specimens and can share measurements. Each arm is shown once.",
            arms=scores,
        ),
        "focal_boundary": focal,
        "stability": stability,
        "fitted_split_vs_merge": evidence(
            "not_computed",
            "No saved fitted comparison evaluates these two exact hypotheses.",
        ),
        "independent_validation": evidence(
            "not_computed", "The saved evidence comes from the discovery cohort."
        ),
    }


def compare_hypotheses_from_archive(
    run_dir: str | Path,
    *,
    study_manifest: str | Path,
    hypotheses: str | Path,
    output_dir: str | Path,
) -> ArchiveReport:
    """Compare named specimen partitions and write local JSON and offline HTML.

    The schema-1 manifest names archive arms, analyst CSVs and two-group merges.
    Analyst CSVs declare exactly the archive roster with columns unit_id, group_id
    and status. Unresolved, abstained and failed rows have empty group IDs.
    """
    output = _fresh_output(output_dir)
    declarations, contrasts, source = read_hypothesis_manifest(hypotheses)
    archive = SealedRun(Path(run_dir), Path(study_manifest))
    require(
        not output.resolve().is_relative_to(archive.root),
        "Comparison destination must be outside the sealed study directory",
    )
    try:
        run, native, orders, manifest = _read_run(archive)
        arms = _archive_hypotheses(archive, run, manifest)
        ids = sorted(run["cohort"]["unit_ids"])
        candidates = resolve_hypotheses(declarations, Path(source["path"]).parent, arms, set(ids))
        partitions = {arm["name"]: arm for arm in run["partitions"]}
        document = {
            "schema_version": 1,
            "report_type": "hypothesis_comparison",
            "run": run["run"],
            "cohort": {**run["cohort"], "policy": "exact", "unit_ids": ids},
            "hypotheses": [
                {
                    "id": name,
                    "source": hypothesis.source,
                    "coverage": _coverage(hypothesis),
                    "assignments": [
                        {
                            "unit_id": unit,
                            "group_id": row.get("assignment"),
                            "status": row["status"],
                            "reason": row["reason"],
                        }
                        for unit, row in sorted(hypothesis.assignments.items())
                    ],
                }
                for name, hypothesis in candidates.items()
            ],
            "contrasts": [
                _contrast(
                    {"id": name, "left": spec["left"], "right": spec["right"]},
                    candidates,
                    arms,
                    ids,
                    partitions,
                    native,
                    orders,
                    archive,
                )
                for name, spec in contrasts.items()
            ],
            "archive_arms": [
                {
                    **partition,
                    "source": arms[name].source,
                    "coverage": _coverage(arms[name]),
                }
                for name, partition in partitions.items()
            ],
            "native_specimen_assignments": run["specimens"],
            "native_inference": run["inference"],
            "limitations": [
                *run["limitations"],
                "Pairs share specimens and do not supply independent significance trials.",
                "Analyst origins and agreement on shared specimens do not establish validation.",
            ],
            "provenance": {**run["provenance"], "hypothesis_manifest": source},
        }
        html = render_comparison(document)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ArchiveError(f"Malformed matched-multimodal comparison input: {exc}") from exc
    return _write_report(document, html, output, "comparison")
