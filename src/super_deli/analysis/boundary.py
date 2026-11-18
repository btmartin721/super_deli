"""Summarize a focal split using member sets and saved full-cohort projections."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable
from math import comb
from typing import Literal, TypedDict

from .archive_io import SealedRun, id_index, require


class PairCounts(TypedDict):
    count: int | None
    comparable: int
    possible: int


def evidence(
    status: Literal["available", "not_computed", "unavailable"],
    reason: str | None = None,
    **fields,
) -> dict:
    return {"status": status, "reason": reason, **fields}


def pair_counts(count: int, comparable: int, possible: int) -> PairCounts:
    return {
        "count": count if comparable else None,
        "comparable": comparable,
        "possible": possible,
    }


def groups_by_label(labels: Iterable[tuple[str, Hashable]]) -> dict:
    groups = {}
    for unit, label in labels:
        if label is not None:
            groups.setdefault(label, set()).add(unit)
    return groups


def boundary_agreement(partition, left, right):
    sizes = [
        Counter(partition[unit] for unit in group if partition[unit] is not None)
        for group in (left, right)
    ]
    a, b = sizes
    n_a, n_b = (sum(s.values()) for s in sizes)
    cross = n_a * n_b
    kept = sum(count * b[label] for label, count in a.items())
    cohesion = [
        pair_counts(sum(comb(v, 2) for v in s.values()), comb(n, 2), comb(len(group), 2))
        for s, n, group in zip(sizes, (n_a, n_b), (left, right), strict=True)
    ]
    return evidence(
        "available" if cross or any(row["comparable"] for row in cohesion) else "unavailable",
        None if cross or any(row["comparable"] for row in cohesion) else "No resolved pairs",
        cross_group_separation=pair_counts(cross - kept, cross, len(left) * len(right)),
        within_group_cohesion=cohesion,
    )


def _draws(records, original_order, requested):
    prefix = "Combined bootstrap"
    require(
        isinstance(records, list) and len(records) == requested,
        f"{prefix}: requested draw count mismatch",
    )
    n = len(original_order)
    seen = set()
    for row in records:
        index = row["replicate"]
        require(
            type(index) is int and 0 <= index < requested and index not in seen,
            f"{prefix}: invalid or repeated draw index",
        )
        seen.add(index)
        require(row["status"] in {"complete", "failed"}, f"{prefix}: unknown draw status")
        omitted = row.get("oob_indices")
        training = row.get("training_indices")
        require(
            isinstance(training, list)
            and len(training) == n
            and all(type(v) is int and 0 <= v < n for v in training)
            and isinstance(omitted, list)
            and all(type(v) is int and 0 <= v < n for v in omitted)
            and len(set(omitted)) == len(omitted)
            and set(omitted) == set(range(n)) - set(training),
            f"{prefix}: omission indices do not match training on the native cohort",
        )
        omitted_ids = {original_order[i] for i in omitted}
        if row["status"] == "failed":
            yield None, omitted_ids
            continue
        labels = row["labels"]
        k = row["selected_k"]
        require(
            type(k) is int
            and k >= 1
            and len(labels) == n
            and all(type(v) is int and 0 <= v < k for v in labels),
            f"{prefix}: malformed saved projection",
        )
        yield dict(zip(original_order, labels, strict=True)), omitted_ids


def recurrence(
    archive: SealedRun,
    arm_name: str,
    stability: dict,
    original_order: list[str],
    candidate_groups: dict,
    focal_groups: dict,
    pair: tuple[str, str],
) -> dict:
    prefix = f"arms/{arm_name}"
    records = archive.json(f"{prefix}/bootstrap_replicates.json", optional=True)
    native_clusters = archive.json(f"{prefix}/clusters.json", optional=True)
    if native_clusters is not None:
        native_groups = id_index(native_clusters, "label", prefix)
        require(
            all(len(row["unit_ids"]) == len(set(row["unit_ids"])) for row in native_clusters)
            and {k: set(row["unit_ids"]) for k, row in native_groups.items() if row["unit_ids"]}
            == candidate_groups,
            f"{prefix}: native cluster member sets disagree with assignments",
        )
    requested = stability["requested"]
    successful = stability["successful"]
    failed = stability["failed"]
    require(
        all(type(v) is int and v >= 0 for v in (requested, successful, failed))
        and successful + failed == requested,
        f"{prefix}: inconsistent bootstrap denominators",
    )
    metadata = dict(
        requested=requested,
        successful=successful,
        failed=failed,
        resampling_unit=stability["resampling_unit"],
    )
    if not requested:
        require(records is None or records == [], f"{prefix}: unexpected saved draws")
        return evidence(
            "not_computed", "The native run requested zero bootstrap draws.", **metadata
        )
    if records is None:
        return evidence("unavailable", "Saved bootstrap projections are absent.", **metadata)

    exact, native_exact = Counter(), Counter()
    covered = omitted_requested = omitted_covered = separated = omitted_separated = both = 0
    for projection, omitted in _draws(records, original_order, requested):
        pair_omitted = set(pair) <= omitted
        omitted_requested += pair_omitted
        if projection is None:
            continue
        covered += 1
        omitted_covered += pair_omitted
        apart = projection[pair[0]] != projection[pair[1]]
        separated += apart
        omitted_separated += apart and pair_omitted
        members = list(groups_by_label(projection.items()).values())
        recurrent = [g for g, group in focal_groups.items() if group in members]
        exact.update(recurrent)
        both += len(recurrent) == len(focal_groups)
        if native_clusters is not None:
            native_exact.update(g for g, group in candidate_groups.items() if group in members)
    require(covered == successful, f"{prefix}: successful projection count mismatch")
    if native_clusters is not None:
        require(
            all(
                row["exact_recovery_count"] == native_exact[label]
                for label, row in native_groups.items()
                if row["unit_ids"]
            ),
            f"{prefix}: exact group recurrence differs from the native ledger",
        )
    return evidence(
        "available" if covered else "unavailable",
        None if covered else "Every requested projection failed.",
        **metadata,
        covered=covered,
        exact_group_recurrence={g: pair_counts(exact[g], covered, requested) for g in focal_groups},
        joint_exact_group_recurrence=pair_counts(both, covered, requested),
        specimen_pair_separation={
            "all_projected_draws": pair_counts(separated, covered, requested),
            "both_specimens_omitted": pair_counts(
                omitted_separated, omitted_covered, omitted_requested
            ),
        },
    )
