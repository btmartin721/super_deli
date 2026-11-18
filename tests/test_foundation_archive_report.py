"""Native-shaped archive fixtures assembled from declarations and literal evidence."""

import csv
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from super_deli.analysis import report_from_archive
from super_deli.analysis.archive_io import read_membership_matrix
from super_deli.cli import main
from super_deli.core.provenance import configuration_hash
from super_deli.core.schema import config_from_dict
from super_deli.models.config import BaselineConfig
from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
from super_deli.models.multimodal_analysis_config import (
    MultimodalAnalysisArm,
    MultimodalAnalysisConfig,
)
from super_deli.models.shape_cluster_config import ShapeClusterConfig

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/integrative/matched_dataset.json"


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _npy_bytes(descr, shape, data, *, fortran_order=False):
    header = repr({"descr": descr, "fortran_order": fortran_order, "shape": shape}).encode("ascii")
    header += b" " * (-(len(header) + 11) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data


def _npy(path, values, *, fortran_order=False):
    data_rows = zip(*values, strict=True) if fortran_order else values
    data = b"".join(struct.pack(f"<{len(row)}d", *row) for row in data_rows)
    path.write_bytes(
        _npy_bytes("<f8", (len(values), len(values[0])), data, fortran_order=fortran_order)
    )


def _seal(run):
    summary = run / "summary.json"
    payload = json.loads(summary.read_text())
    payload["artifacts"] = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file() and path != summary
    }
    _json(summary, payload)
    manifest = run.parent / "study-manifest.json"
    _json(
        manifest,
        {
            path.relative_to(run.parent).as_posix(): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in run.rglob("*")
            if path.is_file()
        },
    )
    return manifest


def _saved_comparisons():
    common = {
        "status": "available",
        "n_units": 4,
        "left_resolved": 4,
        "right_resolved": 4,
        "joint_resolved": 4,
        "joint_coverage": 1.0,
        "joint_pairs": 6,
        "right_clusters_on_intersection": 2,
        "interpretation": "Conditional on jointly resolved non-noise units; no accuracy claim",
    }
    return [
        dict(
            common,
            left=left,
            right=right,
            discordant_pairs=4 if left == "genomic" else 0,
            coassignment_agreement=1 - 4 / 6 if left == "genomic" else 1.0,
            adjusted_rand=0.0 if left == "genomic" else 1.0,
            left_clusters_on_intersection=1 if left == "genomic" else 2,
            degenerate=left == "genomic",
        )
        for left, right in (("genomic", "shape"), ("genomic", "combined"), ("shape", "combined"))
    ]


@pytest.fixture
def archive(tmp_path):
    run = tmp_path / "study" / "run"
    (run / "inputs").mkdir(parents=True)
    ids = ["u1", "u2", "u3", "u4"]
    native_order = ["u3", "u1", "u4", "u2"]
    shape = ShapeClusterConfig(True, n_bootstrap=0)
    genomic = BaselineConfig(2, "error", True, n_bootstrap=0)
    combined = IntegrativeConfig(
        True,
        n_bootstrap=3,
        views=(
            ViewConfig("markers", "genomic", ploidy=2),
            ViewConfig("outline", "morphometric", shape=shape.shape),
        ),
    )
    config = MultimodalAnalysisConfig(
        (
            MultimodalAnalysisArm("genomic", "genomic", ("markers",), genomic),
            MultimodalAnalysisArm("shape", "shape", ("outline",), shape),
            MultimodalAnalysisArm("combined", "integrative", ("markers", "outline"), combined),
        )
    ).to_dict()
    dataset = config_from_dict(
        {
            "schema_version": 1,
            "dataset_id": "archived<&>",
            "specimens": [{"specimen_id": unit} for unit in ids],
            "modalities": [
                {
                    "name": name,
                    "kind": kind,
                    "format": fmt,
                    "level": "specimen",
                    "path": f"{name}.txt",
                    "observations": [
                        {"unit_id": unit, "observation_id": name + unit} for unit in ids
                    ],
                    **options,
                }
                for name, kind, fmt, options in (
                    ("markers", "genomic", "phylip", {}),
                    (
                        "outline",
                        "morphometric",
                        "tps",
                        {
                            "morphometric_options": {
                                "dimensions": 2,
                                "landmark_ids": ["p", "q", "r"],
                            }
                        },
                    ),
                )
            ],
        }
    ).to_dict()
    cohort = {
        "unit_ids": ids,
        "independence": "assumed_not_verified",
        "ecology_measurement_scope": None,
        "views": [
            {
                "modality": m["name"],
                "kind": m["kind"],
                "feature_names": [],
                "observation_ids": [m["name"] + unit for unit in ids],
            }
            for m in dataset["modalities"]
        ],
    }
    digest = configuration_hash(config)
    _json(run / "inputs/cohort.json", cohort)
    _json(run / "inputs/dataset.json", dataset)
    _json(
        run / "manifest.json",
        {
            "schema_version": 1,
            "analysis": config,
            "cohort": cohort,
            "dataset": dataset,
            "frozen_dataset": dataset,
            "configuration_sha256": digest,
            "sources": [],
        },
    )

    labels = {"u1": 0, "u2": 1, "u3": 0, "u4": 1}
    membership = 0.9999999999914202
    other = 8.302500000000011e-12
    comparison = {
        unit: {
            "unit_id": unit,
            "markers__observation_id": "markers" + unit,
            "outline__observation_id": "outline" + unit,
        }
        for unit in reversed(ids)
    }
    models = []
    for arm in config["arms"]:
        name = arm["name"]
        path = run / "arms" / name
        path.mkdir(parents=True)
        configuration = {"dataset": dataset, "model": arm["model"]}
        is_genomic, is_combined = name == "genomic", name == "combined"
        k = 1 if is_genomic else 2
        stability = {
            "status": "available" if is_combined else "not_requested",
            "requested": 3 if is_combined else 0,
            "successful": 3 if is_combined else 0,
            "failed": 0,
            "resampling_unit": "paired specimens",
        }
        native = {
            "schema_version": 1,
            "status": "complete",
            "dataset_id": dataset["dataset_id"],
            "selected_k": k,
        }
        if is_combined:
            native.update(
                unit_ids=native_order,
                level="specimen",
                bootstrap=stability,
                views=[
                    {
                        "modality": modality,
                        "members": [
                            {"target_id": unit, "observation_ids": [modality + unit]}
                            for unit in ids
                        ],
                    }
                    for modality in arm["modalities"]
                ],
            )
        else:
            configuration["modality"] = arm["modalities"][0]
            if is_genomic:
                configuration["blocks"] = None
                native.update(
                    specimen_ids=native_order,
                    sample_ids=["markers" + unit for unit in native_order],
                    cluster_labels=["cluster_1"],
                    stability=stability,
                )
                _npy(path / "membership.npy", [(1.0,)] * 4)
                (path / "labels.npy").write_bytes(
                    _npy_bytes("<i8", (4,), struct.pack("<4q", 0, 0, 0, 0))
                )
            else:
                native.update(
                    unit_ids=native_order,
                    level="specimen",
                    bootstrap=stability,
                    observation_ids=["outline" + unit for unit in native_order],
                )
        native.update(
            provenance={
                "configuration": configuration,
                "configuration_sha256": configuration_hash(configuration),
            }
        )
        _json(path / "summary.json", native)
        rows = []
        for unit in reversed(native_order):
            zero_based = 0 if is_genomic else labels[unit]
            label = f"cluster_{zero_based + 1}"
            comparison[unit][f"{name}__cluster"] = zero_based
            peak, next_best = (0.75, 0.25) if is_combined and unit == "u3" else (membership, other)
            comparison[unit][f"{name}__maximum_conditional_membership"] = (
                1.0 if is_genomic else peak
            )
            rows.append(
                {
                    "unit_id": unit,
                    "observation_id": unit if is_combined else "outline" + unit,
                    "candidate_cluster": label,
                    "assignment": label,
                    "reason": "",
                    "membership_cluster_1": peak if zero_based == 0 else next_best,
                    "membership_cluster_2": peak if zero_based == 1 else next_best,
                }
            )
        if not is_genomic:
            _csv(path / "assignments.csv", rows)
        if name == "shape":
            _json(
                path / "result.json",
                {
                    "schema_version": 1,
                    "configuration_sha256": configuration_hash(configuration),
                    "dataset_id": dataset["dataset_id"],
                    "level": "specimen",
                    "assignments": [
                        {"unit_id": row["unit_id"], "label": row["assignment"], "reason": None}
                        for row in rows
                    ],
                    "evidence": [
                        {
                            "kind": "p_value",
                            "target": "shape_model",
                            "value": 0.12345678901234567,
                            "status": "available",
                            "null_hypothesis": "Historical fitted null",
                            "method": "Archived model check",
                            "reason": None,
                        }
                    ],
                },
            )
        models.append(
            {
                "name": name,
                "family": arm["family"],
                "status": "complete",
                "selected_k": k,
                "search_boundary": k == arm["model"]["max_k"],
                "error": None,
                "modalities": arm["modalities"],
                "configuration": arm["model"],
                "stability": stability,
                "evidence_type": "native bootstrap",
            }
        )
    _csv(run / "assignments.csv", list(comparison.values()))
    replications = [
        ([1, 1, 0, 0], [0, 2, 2, 2], [1, 3], 2),
        ([0, 0, 0, 0], [0, 1, 3, 3], [2], 1),
        ([0, 0, 1, 1], [0, 0, 2, 3], [1], 2),
    ]
    _json(
        run / "arms/combined/bootstrap_replicates.json",
        [
            {
                "replicate": index,
                "status": "complete",
                "labels": groups,
                "training_indices": training,
                "oob_indices": oob,
                "selected_k": k,
            }
            for index, (groups, training, oob, k) in enumerate(replications)
        ],
    )
    _json(
        run / "arms/combined/clusters.json",
        [
            {"label": "cluster_1", "unit_ids": ["u3", "u1"], "exact_recovery_count": 2},
            {"label": "cluster_2", "unit_ids": ["u2", "u4"], "exact_recovery_count": 2},
        ],
    )
    _json(
        run / "summary.json",
        {
            "schema_version": 1,
            "dataset_id": dataset["dataset_id"],
            "models": models,
            "comparisons": _saved_comparisons(),
            "status": "complete",
            "configuration_sha256": digest,
            "unit_ids": ids,
            "n_units": len(ids),
            "inference": {"status": "not_computed", "p_value": None},
            "limitations": [],
        },
    )
    return run, _seal(run)


def _cli(run, manifest, dest):
    return [
        "analyze",
        "report",
        str(run),
        "--study-manifest",
        str(manifest),
        "--output-dir",
        str(dest),
    ]


def test_verified_report_and_cli_parity(archive, tmp_path, capsys):
    run, manifest = archive
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run.parent.rglob("*")
        if path.is_file()
    }
    output = report_from_archive(run, study_manifest=manifest, output_dir=tmp_path / "py")
    document = json.loads(output.json_path.read_text())
    focal = document["focal_boundary"]
    arms = focal["descriptive_agreement"]["arms"]
    assert focal["specimen_pair"] == ["u1", "u2"]
    for arm, count in (("shape", 4), ("genomic", 0), ("combined", 4)):
        assert arms[arm]["cross_group_separation"] == {
            "count": count,
            "comparable": 4,
            "possible": 4,
        }
        assert [g["count"] for g in arms[arm]["within_group_cohesion"]] == [1, 1]
    assert focal["stability"]["joint_exact_group_recurrence"] == {
        "count": 2,
        "comparable": 3,
        "possible": 3,
    }
    assert focal["stability"]["specimen_pair_separation"] == {
        "all_projected_draws": {"count": 2, "comparable": 3, "possible": 3},
        "both_specimens_omitted": {"count": 1, "comparable": 1, "possible": 1},
    }
    assert focal["fitted_split_vs_merge"]["status"] == "not_computed"
    assert document["inference"]["held_out_gain"]["experimental"]
    assert document["inference"]["species_boundary_inference"]["status"] == "not_computed"
    assert document["partitions"][-1]["lowest_maximum_membership"] == {
        "unit_id": "u3",
        "membership": 0.75,
    }
    verified = document["provenance"]["verified_run_artifacts"]
    assert "inputs/dataset.json" in verified
    assert "arms/genomic/labels.npy" in verified
    specimen = document["specimens"][0]
    assert specimen["source_observation_ids"] == {"markers": "markersu1", "outline": "outlineu1"}
    assert specimen["arms"]["combined"]["maximum_conditional_membership"] == 0.9999999999914202
    assert (
        specimen["arms"]["combined"]["strongest_alternative"]["membership"] == 8.302500000000011e-12
    )
    historical = document["partitions"][1]["native_p_values"][0]
    assert historical["value"] == 0.12345678901234567
    assert historical["null_hypothesis"] == "Historical fitted null"
    html = output.html_path.read_text()
    assert "archived&lt;&amp;&gt;" in html
    assert "Specimen pair separated when both were omitted" in html
    assert repr(8.302500000000011e-12) in html
    assert "<td>u3</td><td>Lowest maximum membership</td>" in html
    assert "Species-boundary inference · not computed" in html
    assert "Saved stability by arm" in html
    cli_dest = tmp_path / "cli"
    assert main(_cli(run, manifest, cli_dest)) == 0
    assert json.loads(capsys.readouterr().out) == {
        "json_path": str(cli_dest / "report.json"),
        "html_path": str(cli_dest / "report.html"),
    }
    assert (cli_dest / "report.json").read_bytes() == output.json_path.read_bytes()
    assert (cli_dest / "report.html").read_text() == html
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run.parent.rglob("*")
        if path.is_file()
    }


def test_native_comparisons_can_be_reordered(archive, tmp_path):
    run, _ = archive
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["comparisons"].reverse()
    _json(path, summary)
    report = report_from_archive(run, study_manifest=_seal(run), output_dir=tmp_path / "report")
    assert json.loads(report.json_path.read_text())["partition_comparisons"] == _saved_comparisons()


@pytest.mark.parametrize(
    "field,saved,derived",
    [
        ("discordant_pairs", 3, 4),
        ("status", "undefined", "available"),
        ("joint_coverage", 0.75, 1.0),
        ("joint_pairs", 5, 6),
        ("left_resolved", 3, 4),
        ("adjusted_rand", 0.1, 0.0),
    ],
)
def test_resealed_comparison_contradiction_exits_two(
    archive, tmp_path, capsys, field, saved, derived
):
    run, _ = archive
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["comparisons"][0][field] = saved
    _json(path, summary)
    manifest = _seal(run)
    dest = tmp_path / "report"
    assert main(_cli(run, manifest, dest)) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert f"'genomic' vs 'shape': {field} saved {saved!r}, recomputed {derived!r}" in captured.err
    assert not dest.exists()


def test_undefined_comparison_preserves_null_agreement(archive, tmp_path):
    run, _ = archive
    path = run / "arms/shape/assignments.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["unit_id"] != "u1":
            row.update(assignment="", reason="Archived abstention")
    _csv(path, rows)
    path = run / "arms/shape/result.json"
    result = json.loads(path.read_text())
    for assignment in result["assignments"]:
        if assignment["unit_id"] != "u1":
            assignment.update(label=None, reason="Archived abstention")
    _json(path, result)
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    for pair in summary["comparisons"]:
        for side in ("left", "right"):
            if pair[side] == "shape":
                pair[f"{side}_resolved"] = 1
                pair.update(
                    status="undefined",
                    joint_resolved=1,
                    joint_coverage=0.25,
                    joint_pairs=0,
                    discordant_pairs=0,
                    coassignment_agreement=None,
                    adjusted_rand=None,
                    left_clusters_on_intersection=1,
                    right_clusters_on_intersection=1,
                    degenerate=True,
                )
    _json(path, summary)
    report = report_from_archive(run, study_manifest=_seal(run), output_dir=tmp_path / "report")
    actual = json.loads(report.json_path.read_text())["partition_comparisons"][0]
    assert actual == dict(
        summary["comparisons"][0],
        status="unavailable",
        reason="Fewer than two jointly resolved units",
    )


def test_optional_evidence_omitted_from_both_seals(archive, tmp_path):
    run, _ = archive
    (run / "arms/genomic/membership.npy").unlink()
    (run / "arms/combined/bootstrap_replicates.json").unlink()
    manifest = _seal(run)
    report = report_from_archive(run, study_manifest=manifest, output_dir=tmp_path / "report")
    document = json.loads(report.json_path.read_text())
    assert document["focal_boundary"]["stability"]["status"] == "unavailable"
    assert document["focal_boundary"]["stability"]["requested"] == 3
    genomic = document["specimens"][0]["arms"]["genomic"]
    assert genomic["membership_vector_evidence"]["status"] == "unavailable"
    assert genomic["strongest_alternative"]["status"] == "not_computed"


@pytest.mark.parametrize(
    "fault,diagnostic",
    [
        ("digest", "Digest or size mismatch"),
        ("study_only", "Native run digest mismatch"),
        ("cohort", "inconsistent assignment IDs"),
        ("native_label", "native genomic label mismatch"),
        ("frozen_dataset", "Saved frozen dataset differs"),
        ("sealed_optional", "Missing sealed artifact"),
        ("existing", "destination already exists"),
    ],
)
def test_verification_failures_leave_no_completed_report(
    archive, tmp_path, capsys, fault, diagnostic
):
    run, manifest = archive
    dest = tmp_path / "report"
    source = run / "arms/combined/assignments.csv"
    if fault == "existing":
        dest.mkdir()
        (dest / "keep").write_bytes(b"untouched")
    elif fault == "native_label":
        labels = run / "arms/genomic/labels.npy"
        labels.write_bytes(_npy_bytes("<i8", (4,), struct.pack("<4q", 1, 0, 0, 0)))
        manifest = _seal(run)
    elif fault == "frozen_dataset":
        frozen = run / "inputs/dataset.json"
        value = json.loads(frozen.read_text())
        value["dataset_id"] = "wrong-dataset"
        _json(frozen, value)
        manifest = _seal(run)
    elif fault == "sealed_optional":
        (run / "arms/genomic/membership.npy").unlink()
    else:
        content = source.read_text()
        source.write_text(content.replace("u4,u4", "u5,u4", 1))
        if fault == "study_only":
            study = json.loads(manifest.read_text())
            study["run/arms/combined/assignments.csv"] = {
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            _json(manifest, study)
        elif fault == "cohort":
            manifest = _seal(run)
    assert main(_cli(run, manifest, dest)) == 2
    captured = capsys.readouterr()
    assert diagnostic in captured.err
    assert not captured.out
    if fault == "existing":
        assert list(dest.iterdir()) == [dest / "keep"]
        assert (dest / "keep").read_bytes() == b"untouched"
    else:
        assert not dest.exists()


def test_entry_points_cannot_import_fitters_or_optional_dependencies(archive, tmp_path):
    run, manifest = archive
    script = """
import importlib.abc
import sys

attempted = []


class NoFitImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        forbidden = {"numpy", "scipy", "sklearn", "snpio", "torch", "threadpoolctl"}
        workflows = {
            "multimodal_analysis", "matched_comparison", "genomic_analysis_adapters",
            "matched_analysis", "matched_preprocessing", "procrustes",
            "genomic_baseline", "shape_gmm", "shape_pca", "tabular_gmm", "integrative",
            "heldout_support", "unit_bootstrap",
        }
        if fullname.split(".")[0] in forbidden or (
            fullname.startswith("super_deli.") and fullname.split(".")[-1] in workflows
        ):
            attempted.append(fullname)
            raise AssertionError("Fit or optional dependency import: " + fullname)

sys.meta_path.insert(0, NoFitImports())
from pathlib import Path
from super_deli.analysis import plan_analysis, report_from_archive
from super_deli.cli import main

dataset, run, manifest, destination = map(Path, sys.argv[1:])
plan = plan_analysis(dataset, modalities=["markers", "outline"], target_level="specimen")
assert plan.status == "ready"
assert main(["analyze", "plan", str(dataset), "--modality", "markers", "--modality",
             "outline", "--target-level", "specimen"]) == 0
tabular = dataset.parents[1] / "matched-analysis/dataset.json"
plan = plan_analysis(tabular, modalities=["morphology", "ecology"], target_level="specimen")
assert "analyze-matched-tabular" in plan.eligible_native_methods
assert main(["analyze", "plan", str(tabular), "--modality", "morphology", "--modality",
             "ecology", "--target-level", "specimen"]) == 0
report_from_archive(run, study_manifest=manifest, output_dir=destination / "api")
assert main(["analyze", "report", str(run), "--study-manifest", str(manifest),
             "--output-dir", str(destination / "cli")]) == 0
assert attempted == []
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(EXAMPLE), str(run), str(manifest), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fortran_order", [False, True])
def test_native_membership_layouts(tmp_path, fortran_order):
    matrix = ((0.75, 0.25), (0.1, 0.9), (0.125, 0.875))
    source = tmp_path / "membership.npy"
    _npy(source, matrix, fortran_order=fortran_order)
    assert read_membership_matrix(source.read_bytes(), rows=3, columns=2) == matrix


def test_failed_arm_and_draw_keep_coverage_denominators(archive, tmp_path):
    run, _ = archive
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["status"] = "partial"
    summary["models"][0].update(
        status="failed", error="Archived failure", selected_k=None, search_boundary=None
    )
    for pair in summary["comparisons"]:
        if pair["left"] == "genomic":
            left, right = pair["left"], pair["right"]
            pair.clear()
            pair.update(
                left=left,
                right=right,
                status="unavailable",
                reason="A native point/cohort verification failed",
            )
    bootstrap = summary["models"][-1]["stability"]
    bootstrap.update(successful=2, failed=1, status="unavailable")
    summary["models"][-1]["status"] = "partial"
    _json(path, summary)
    path = run / "assignments.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(genomic__cluster="", genomic__maximum_conditional_membership="")
    _csv(path, rows)
    path = run / "arms/combined/summary.json"
    native = json.loads(path.read_text())
    native["bootstrap"] = bootstrap
    _json(path, native)
    path = run / "arms/combined/bootstrap_replicates.json"
    draws = json.loads(path.read_text())
    draws[0].update(status="failed", error="Archived draw failure")
    draws[0].pop("labels")
    _json(path, draws)
    path = run / "arms/combined/clusters.json"
    clusters = json.loads(path.read_text())
    for row in clusters:
        row["exact_recovery_count"] = 1
    _json(path, clusters)

    report = report_from_archive(
        run,
        study_manifest=_seal(run),
        output_dir=tmp_path / "report",
        boundary_groups=("cluster_2", "cluster_1"),
        specimen_pair=("u1", "u2"),
    )
    data = json.loads(report.json_path.read_text())
    assert data["run"]["status"] == "partial"
    assert data["specimens"][0]["arms"]["genomic"] == {
        "status": "unavailable",
        "reason": "Archived failure",
    }
    comparison = data["partition_comparisons"][0]
    assert comparison["status"] == "unavailable"
    assert comparison["left_resolved"] == comparison["joint_pairs"] == 0
    assert comparison["right_resolved"] == 4
    assert comparison["discordant_pairs"] is None
    stability = data["focal_boundary"]["stability"]
    assert stability["joint_exact_group_recurrence"] == {"count": 1, "comparable": 2, "possible": 3}
    assert stability["specimen_pair_separation"]["both_specimens_omitted"] == {
        "count": None,
        "comparable": 0,
        "possible": 1,
    }
