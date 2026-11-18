"""Fit-free checks of the public comparison API using literal sealed records."""

import csv
import hashlib
import itertools
import json

import pytest

from super_deli.analysis import compare_hypotheses_from_archive
from super_deli.cli import main
from tests.test_foundation_archive_report import _csv, _json, _seal
from tests.test_foundation_archive_report import archive as archive

UNITS = ("u1", "u2", "u3", "u4")
SPLIT = ("cluster_1", "cluster_2", "cluster_1", "cluster_2")
COMBINED = {"id": "split", "source": {"kind": "archive_arm", "arm": "combined"}}
MERGE = {
    "id": "merge",
    "source": {"kind": "merge", "parent": "split", "groups": ["cluster_1", "cluster_2"]},
}
TABLE = {
    "id": "table",
    "source": {
        "kind": "assignment_table",
        "path": "analyst.csv",
        "origin": "Voucher ledger <2026>",
    },
}
DEFAULT = [{"id": "boundary", "left": "split", "right": "merge"}]


def _request(tmp_path, candidates=(MERGE, COMBINED), contrasts=None, **overrides):
    path = tmp_path / "candidates.json"
    _json(
        path,
        {
            "schema_version": 1,
            "target_level": "specimen",
            "cohort_policy": "exact",
            "hypotheses": candidates,
            "contrasts": contrasts if contrasts is not None else DEFAULT,
            **overrides,
        },
    )
    return path


def _table(tmp_path, groups, units=UNITS):
    path = tmp_path / "analyst.csv"
    _csv(
        path,
        [
            {
                "unit_id": unit,
                "group_id": group or "",
                "status": "resolved" if group else "abstained",
            }
            for unit, group in reversed(list(zip(units, groups, strict=True)))
        ],
    )
    return path


def _run(archive, path, output):
    run, manifest = archive
    return compare_hypotheses_from_archive(
        run, study_manifest=manifest, hypotheses=path, output_dir=output
    )


def _cli(archive, path, output):
    run, manifest = archive
    return [
        "analyze",
        "compare-hypotheses",
        str(run),
        "--study-manifest",
        str(manifest),
        "--hypotheses",
        str(path),
        "--output-dir",
        str(output),
    ]


def _enumerate(left, right, arm):
    results = [0] * 4
    for (a, b, c), (x, y, z) in itertools.combinations(zip(left, right, arm, strict=True), 2):
        if None in (a, b, x, y) or (a == x) == (b == y):
            continue
        results[0 if a == x else 1] += 1
        if c is not None and z is not None:
            results[2 if (a == x) == (c == z) else 3] += 1
    return results


@pytest.mark.parametrize(
    "groups,relation",
    [
        (("zeta", "alpha", "zeta", "alpha"), "equivalent"),
        (("m", "m", "n", "n"), "non_nested"),
        (("m", "m", "m", "m"), "left_refines_right"),
        (("a", "b", "c", "d"), "right_refines_left"),
        (("all", "all", "all", None), "left_refines_right"),
        (("a", None, None, None), "indeterminate"),
        ((None, None, None, None), "indeterminate"),
        (("a", "b", "c", "b"), "right_refines_left"),
    ],
)
def test_table_contrast_is_label_invariant_and_coverage_explicit(
    archive, tmp_path, groups, relation
):
    path = _table(tmp_path, groups)
    manifest = _request(
        tmp_path, [COMBINED, TABLE], [{"id": "compare", "left": "split", "right": "table"}]
    )
    result = _run(archive, manifest, tmp_path / "report")
    doc = json.loads(result.json_path.read_bytes())
    assert doc["hypotheses"][1]["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "Voucher ledger &lt;2026&gt;" in result.html_path.read_text()
    row = doc["contrasts"][0]
    assert row["independent_validation"]["status"] == "not_computed"
    assert row["coverage"]["declared_pairs"] == 6
    assert row["coverage"]["jointly_resolved_units"] == sum(bool(group) for group in groups)
    metrics = row["partition_comparison"]
    assert metrics["relation"] == relation
    for arm, labels in (("combined", SPLIT), ("genomic", ("g",) * 4)):
        left, right, supports_left, supports_right = _enumerate(SPLIT, groups, labels)
        assert (metrics["together_only_left"], metrics["together_only_right"]) == (left, right)
        assert metrics["discordant_pairs"] == left + right
        scores = row["same_cohort_agreement"]["arms"][arm]
        assert scores["supporting_left"]["count"] == (supports_left if left + right else None)
        assert scores["supporting_right"]["count"] == (supports_right if left + right else None)
        if left + right == 0:
            assert scores["left_fraction"] is scores["right_fraction"] is None
            assert scores["supporting_left"]["comparable"] == 0 and scores["reason"]
    if relation == "equivalent":
        assert metrics["adjusted_rand"] == metrics["coassignment_agreement"] == 1
    if None in groups:
        excluded = [unit for unit, label in zip(UNITS, groups, strict=True) if label is None]
        assert doc["hypotheses"][1]["coverage"]["exclusions"][0]["unit_ids"] == excluded
        assert row["focal_boundary"]["status"] == "unavailable"
    if groups == ("a", "b", "c", "b"):
        arms = row["focal_boundary"]["arms"]
        assert all(
            counts["fraction"] is None and counts["possible"] == 0 and counts["reason"]
            for counts in arms["combined"]["within_group_cohesion"]
        )


def test_merge_both_directions_cli_parity_and_historical_values(archive, tmp_path, capsys):
    run, _ = archive
    native = run / "arms/shape/result.json"
    result = json.loads(native.read_bytes())
    record = result["evidence"][0]
    record["multiplicity"] = {"method": "holm", "family_size": 4}
    _json(native, result)
    archive = run, _seal(run)
    repeated = {"id": "same_source", "source": COMBINED["source"]}
    path = _request(
        tmp_path,
        [MERGE, repeated, COMBINED],
        [
            *DEFAULT,
            {"id": "reverse", "left": "merge", "right": "split"},
            {"id": "duplicate", "left": "split", "right": "same_source"},
        ],
    )
    output = _run(archive, path, tmp_path / "py")
    assert main(_cli(archive, path, tmp_path / "cli")) == 0
    cli = json.loads(capsys.readouterr().out)
    for absolute in (output.json_path, output.html_path):
        assert absolute.read_bytes() == (tmp_path / "cli" / absolute.name).read_bytes()
    assert cli["json_path"] == str(tmp_path / "cli" / "comparison.json")
    doc = json.loads(output.json_path.read_bytes())
    assert doc["schema_version"] == 1
    assert doc["archive_arms"][1]["native_p_values"] == [record]
    assert str(record["value"]) in output.html_path.read_text()
    for index, side in enumerate(("left", "right")):
        row = doc["contrasts"][index]
        counts = row["partition_comparison"]
        assert counts[f"together_only_{side}"] == 0
        other = "right" if side == "left" else "left"
        assert counts[f"together_only_{other}"] == 4
        assert counts["unchanged_pairs"] == 2
        assert row["focal_boundary"]["originating_arm"] == "combined"
        scores = row["same_cohort_agreement"]["arms"]
        assert len(scores) == 3
        assert scores["combined"]["agreement_by_construction_for"] == [side]
        assert scores["combined"][f"{side}_fraction"] == 1
        assert row["fitted_split_vs_merge"]["status"] == "not_computed"
    assert doc["contrasts"][2]["partition_comparison"]["relation"] == "equivalent"
    stability = doc["contrasts"][0]["stability"]["arms"]["combined"]
    assert stability["joint_exact_group_recurrence"]["count"] == 2
    assert stability["specimen_pair_separation"]["both_specimens_omitted"]["comparable"] == 1
    assert stability["specimen_pair_separation"]["both_specimens_omitted"]["count"] == 1
    assert doc["hypotheses"][0]["source"]["parent_member_sets"] == {
        "cluster_1": ["u1", "u3"],
        "cluster_2": ["u2", "u4"],
    }


def test_fragmented_arm_can_separate_a_named_split(archive, tmp_path):
    _table(tmp_path, ("first", "second", "second", "first"))
    merge = {
        "id": "merge",
        "source": {"kind": "merge", "parent": "table", "groups": ["first", "second"]},
    }
    path = _request(
        tmp_path, [merge, TABLE], [{"id": "boundary", "left": "table", "right": "merge"}]
    )
    out = _run(archive, path, tmp_path / "out")
    row = json.loads(out.json_path.read_bytes())["contrasts"][0]
    boundary = row["focal_boundary"]["arms"]["combined"]
    assert boundary["cross_group_separation"]["count"] == 2
    assert boundary["cross_group_separation"]["comparable"] == 4
    assert [g["count"] for g in boundary["within_group_cohesion"]] == [0, 0]


@pytest.mark.parametrize("mode", ["failed", "missing_all", "missing_one", "partial"])
def test_projection_denominators_are_measure_specific(archive, tmp_path, mode):
    run, _ = archive
    path = run / "arms/combined/bootstrap_replicates.json"
    draws = json.loads(path.read_bytes())
    if mode == "missing_all":
        path.unlink()
    elif mode == "missing_one":
        _json(path, draws[:2])
    elif mode == "partial":
        draws[0]["labels"][0] = None
        _json(path, draws)
    else:
        draws[0]["status"] = "failed"
        _json(path, draws)
        for file in ("summary.json", "arms/combined/summary.json"):
            p = run / file
            native = json.loads(p.read_bytes())
            stability = native.get("bootstrap") or native["models"][-1]["stability"]
            stability.update(successful=2, failed=1)
            _json(p, native)
        clusters = run / "arms/combined/clusters.json"
        values = json.loads(clusters.read_bytes())
        for value in values:
            value["exact_recovery_count"] = 1
        _json(clusters, values)
    out = _run((run, _seal(run)), _request(tmp_path), tmp_path / "report")
    row = json.loads(out.json_path.read_bytes())["contrasts"][0]
    stability = row["stability"]["arms"]["combined"]
    assert stability["requested"] == 3
    assert stability["successful"] == (2 if mode == "failed" else 3)
    assert stability["failed"] == (1 if mode == "failed" else 0)
    assert stability["covered"] == (0 if mode == "missing_all" else 2)
    assert stability["missing"] == {"missing_one": 1, "missing_all": 3}.get(mode, 0)
    assert stability["joint_exact_group_recurrence"]["possible"] == 3
    assert stability["joint_exact_group_recurrence"]["count"] == (
        None if mode == "missing_all" else 1
    )
    pair = stability["specimen_pair_separation"]
    assert pair["all_projected_draws"]["comparable"] == (
        3 if mode == "partial" else 0 if mode == "missing_all" else 2
    )
    assert pair["both_specimens_omitted"]["possible"] == (0 if mode == "missing_all" else 1)
    assert pair["both_specimens_omitted"]["comparable"] == (
        0 if mode in {"failed", "missing_all"} else 1
    )
    assert row["independent_validation"]["status"] == "not_computed"


@pytest.mark.parametrize(
    "units,message",
    [
        (("u1", "u1", "u3", "u4"), "repeated unit_id"),
        (("u1", "u2", "u3"), "missing="),
        (("u1", "u2", "u3", "site1"), "foreign="),
    ],
)
def test_inconsistent_table_roster_rejected(archive, tmp_path, units, message):
    _table(tmp_path, ["assigned"] * len(units), units)
    path = _request(tmp_path, [COMBINED, TABLE], [{"id": "bad", "left": "split", "right": "table"}])
    with pytest.raises(ValueError, match=message):
        _run(archive, path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "change,message",
    [
        ({"target_level": "locality"}, "Unsupported target level"),
        ({"cohort_policy": "intersection"}, "Only the exact cohort"),
        ({"schema_version": True}, "Unsupported hypothesis manifest"),
        ({"hypotheses": [COMBINED, COMBINED]}, "repeated id"),
        ({"contrasts": DEFAULT * 2}, "repeated id"),
        ({"contrasts": [{"id": "unknown", "left": "split", "right": "none"}]}, "unknown right"),
        (
            {"hypotheses": [COMBINED, {**MERGE, "source": {**MERGE["source"], "parent": "merge"}}]},
            "Cyclic merge",
        ),
        (
            {
                "hypotheses": [
                    COMBINED,
                    {**MERGE, "source": {**MERGE["source"], "parent": "absent"}},
                ]
            },
            "unknown merge parent",
        ),
        (
            {
                "hypotheses": [
                    COMBINED,
                    {**MERGE, "source": {**MERGE["source"], "groups": ["cluster_1", "cluster_1"]}},
                ]
            },
            "two distinct",
        ),
        (
            {
                "hypotheses": [
                    COMBINED,
                    {**MERGE, "source": {**MERGE["source"], "groups": ["cluster_1", "missing"]}},
                ]
            },
            "two nonempty",
        ),
    ],
)
def test_invalid_manifest_has_no_completed_report(archive, tmp_path, capsys, change, message):
    path = _request(tmp_path, **change)
    dest = tmp_path / "out"
    assert main(_cli(archive, path, dest)) == 2
    assert message in capsys.readouterr().err
    assert not dest.exists()


def test_integrity_and_fresh_destination(archive, tmp_path):
    run, _ = archive
    request = _request(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    keep = occupied / "keep"
    keep.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        _run(archive, request, occupied)
    assert list(occupied.iterdir()) == [keep]
    path = run / "arms/combined/assignments.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["candidate_cluster"] = "cluster_9"
    _csv(path, rows)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Digest or size mismatch"):
        _run(archive, request, output)
    assert not output.exists()
    with pytest.raises(ValueError, match="native assignment ID or candidate mismatch"):
        _run((run, _seal(run)), request, output)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["abstained", "failed"])
def test_archive_arm_exclusions_reduce_pair_coverage(archive, tmp_path, mode):
    run, _ = archive
    path = run / "summary.json"
    summary = json.loads(path.read_bytes())
    if mode == "failed":
        summary["status"] = "partial"
        summary["models"][0].update(
            status="failed", error="Archived failure", selected_k=None, search_boundary=None
        )
        for row in summary["comparisons"][:2]:
            left, right = row["left"], row["right"]
            row.clear()
            row.update(
                left=left,
                right=right,
                status="unavailable",
                reason="A native point/cohort verification failed",
            )
        csv_path = run / "assignments.csv"
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.update(genomic__cluster="", genomic__maximum_conditional_membership="")
        _csv(csv_path, rows)
        arm = "genomic"
        expected_reason = "Archived failure"
        pairs, left, right = 0, None, None
    else:
        arm = "shape"
        expected_reason = "Archived abstention"
        pairs, left, right = 2, 2, 0
        csv_path = run / "arms/shape/assignments.csv"
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        next(row for row in rows if row["unit_id"] == "u1").update(
            assignment="", reason=expected_reason
        )
        _csv(csv_path, rows)
        result_path = run / "arms/shape/result.json"
        result = json.loads(result_path.read_bytes())
        next(row for row in result["assignments"] if row["unit_id"] == "u1").update(
            label=None, reason=expected_reason
        )
        _json(result_path, result)
        summary["comparisons"][0].update(
            right_resolved=3,
            joint_resolved=3,
            joint_coverage=0.75,
            joint_pairs=3,
            discordant_pairs=2,
            coassignment_agreement=1 - 2 / 3,
        )
        summary["comparisons"][2].update(
            left_resolved=3,
            joint_resolved=3,
            joint_coverage=0.75,
            joint_pairs=3,
        )
    _json(path, summary)
    candidate = {"id": "evidence_arm", "source": {"kind": "archive_arm", "arm": arm}}
    hypotheses = _request(tmp_path, [COMBINED, MERGE, candidate])
    output = _run((run, _seal(run)), hypotheses, tmp_path / "out")
    document = json.loads(output.json_path.read_bytes())
    assert document["hypotheses"][-1]["coverage"]["exclusions"][0]["reason"] == expected_reason
    row = document["contrasts"][0]["same_cohort_agreement"]["arms"][arm]
    assert row["supporting_left"] == {"count": left, "comparable": pairs, "possible": 4}
    assert row["supporting_right"] == {"count": right, "comparable": pairs, "possible": 4}


@pytest.mark.parametrize(
    "unit", ["locus blocks", "Independent biological population; whole rows", None]
)
def test_incompatible_resampling_unit_withholds_recurrence(archive, tmp_path, unit):
    run, _ = archive
    for file in ("summary.json", "arms/combined/summary.json"):
        path = run / file
        summary = json.loads(path.read_bytes())
        stability = summary.get("bootstrap") or summary["models"][-1]["stability"]
        stability["resampling_unit"] = unit
        _json(path, summary)
    report = _run((run, _seal(run)), _request(tmp_path), tmp_path / "report")
    contrast = json.loads(report.json_path.read_bytes())["contrasts"][0]
    stability = contrast["stability"]["arms"]["combined"]
    assert stability["status"] == "unavailable"
    assert "resampling unit is incompatible" in stability["reason"]
    assert stability["resampling_unit"] == unit
    assert (stability["requested"], stability["successful"], stability["failed"]) == (3, 3, 0)
    assert stability["covered"] == 0
    assert "joint_exact_group_recurrence" not in stability
    assert "specimen_pair_separation" not in stability
    assert contrast["same_cohort_agreement"]["status"] == "available"


@pytest.mark.parametrize(
    "fault,diagnostic",
    [
        ("empty", "Missing frozen measurement sources"),
        ("omitted_popmap", "Missing frozen measurement sources.*popmapfile"),
        ("swapped_modalities", "disagrees with its modality declaration"),
        ("duplicate", "Repeated frozen measurement source"),
        ("foreign_role", "Unexpected frozen measurement source"),
    ],
)
def test_frozen_measurement_provenance_is_bound_to_modality_and_role(
    archive, tmp_path, capsys, fault, diagnostic
):
    run, _ = archive
    path = run / "manifest.json"
    manifest = json.loads(path.read_bytes())
    markers, popmap, outline = manifest["sources"]
    if fault == "empty":
        manifest["sources"].clear()
    elif fault == "omitted_popmap":
        manifest["sources"].remove(popmap)
    elif fault == "swapped_modalities":
        markers["modality"], outline["modality"] = outline["modality"], markers["modality"]
    elif fault == "duplicate":
        manifest["sources"].append(markers)
    else:
        outline["role"] = "popmapfile"
    _json(path, manifest)
    sealed = run, _seal(run)
    hypotheses = _request(tmp_path)
    dest = tmp_path / "report"
    with pytest.raises(ValueError, match=diagnostic):
        _run(sealed, hypotheses, dest)
    assert main(_cli(sealed, hypotheses, dest)) == 2
    assert not dest.exists()
    assert "frozen measurement" in capsys.readouterr().err.lower()


def test_valid_frozen_measurement_and_popmap_provenance(archive, tmp_path):
    run, _ = archive
    manifest = json.loads((run / "manifest.json").read_bytes())
    report = _run(archive, _request(tmp_path), tmp_path / "report")
    doc = json.loads(report.json_path.read_bytes())
    arms = {arm["name"]: arm["source"]["measurement_sources"] for arm in doc["archive_arms"]}
    assert arms["combined"]["markers"] == arms["genomic"]["markers"]
    assert arms["combined"]["outline"] == arms["shape"]["outline"]
    for source in manifest["sources"]:
        declared = next(
            view
            for view in manifest["frozen_dataset"]["modalities"]
            if view["name"] == source["modality"]
        )
        snapshot = declared[source["role"]]
        entry = next(
            row for row in arms["combined"][source["modality"]] if row["role"] == source["role"]
        )
        payload = (run / "inputs" / snapshot).read_bytes()
        assert entry["snapshot"] == snapshot
        assert entry["verified_snapshot"] == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "study_path": f"run/inputs/{snapshot}",
        }


@pytest.mark.parametrize("coverage", ["missing", "partial"])
@pytest.mark.parametrize("native_count", [0, 3])
def test_partial_projection_counts_must_reconcile_with_native_ledger(
    archive, tmp_path, coverage, native_count
):
    run, _ = archive
    path = run / "arms/combined/bootstrap_replicates.json"
    draws = json.loads(path.read_bytes())
    # Draws 0 and 1 establish one exact recurrence. Draw 2 is uncovered.
    if coverage == "missing":
        draws.pop()
    else:
        draws[2]["labels"][0] = None
    _json(path, draws)
    path = run / "arms/combined/clusters.json"
    clusters = json.loads(path.read_bytes())
    clusters[0]["exact_recovery_count"] = native_count
    _json(path, clusters)

    dest = tmp_path / "report"
    with pytest.raises(ValueError, match="exact group recurrence differs from the native ledger"):
        _run((run, _seal(run)), _request(tmp_path), dest)
    assert not dest.exists()
