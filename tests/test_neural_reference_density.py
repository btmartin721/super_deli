"""Invented full lifecycle and source-bound NumPy-only frozen support checks."""

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    from super_deli.benchmarking.known_reference_fixture import smoke

    out = tmp_path_factory.mktemp("known_reference") / "smoke"
    result = smoke(out, workers=1)
    assert result["passed"] and result["invented_only"]
    return out, result


def test_complete_known_only_budget_and_group_holdouts(complete):
    out, result = complete
    r = result["study"]
    assert r["observations"] == 1 and r["ordinary_heldout_folds"] == 4
    assert r["main_classifier_fits"] == 10 and r["main_neural_trainings"] == 5
    assert r["audited_optimizer_updates"] == 2500 and r["unknown_training_examples"] == 0
    assert r["primary_rates"] == 8 and r["paired_contrasts"] == 7
    assert not r["default_changes"] and not r["pvalue_changes"] and r["failed_workflows"] == 0
    assert json.loads((out / "study/main-audit.json").read_text())["passed"]
    folds = json.loads((out / "study/fold-groups.json").read_text())
    primary = json.loads((out / "study/primary-rates.json").read_text())
    assert all("simultaneous_95_interval" not in v["joint_target"] for v in folds)
    assert all("simultaneous_95_interval" in v for v in primary)


def test_numpy_only_reload_preserves_reference_probabilities(complete):
    out, _ = complete
    parent = out / "study/main/known_linear_standard_n32_0000/holdout-0"
    code = (
        "import sys;from super_deli.workflows.reference_density import load_reference_density;"
        "load_reference_density(sys.argv[1],reference_run=sys.argv[2]);"
        "assert 'torch' not in sys.modules and 'snpio' not in sys.modules"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(out / "density-calibration"),
            str(parent / "residual_0/run"),
        ],
        check=True,
    )
    import numpy as np

    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.io.config import load_config
    from super_deli.io.tabular import ingest_tabular
    from super_deli.models.reference_projection import aligned_reference_matrix
    from super_deli.workflows.residual_reference import load_residual_reference_model

    ds = load_config(out / "query/dataset.json")
    loaded = ingest_tabular(
        ds,
        base_dir=out / "query",
        output_dir=out / "read-query",
        modality_names=["morphology", "ecology"],
    )
    model = load_residual_reference_model(parent / "residual_0/run")
    x, _ = aligned_reference_matrix(loaded.tables, model.config)
    np.testing.assert_array_equal(
        arrays(out / "prediction/support-predictions.npz")["probabilities"],
        model.predict_matrix(x)["probabilities"],
    )


def test_source_and_numeric_tampering_are_rejected(complete):
    out, _ = complete
    from super_deli.benchmarking.known_reference_study import engine
    from super_deli.workflows.reference_density import load_reference_density

    study = engine(out / "study", out / "invented-source")
    old = (out / "invented-source/fixture.json").read_text()
    try:
        (out / "invented-source/fixture.json").write_text(old + " ")
        with pytest.raises(ValueError):
            study.binding()
    finally:
        (out / "invented-source/fixture.json").write_text(old)
    reference = out / "study/main/known_linear_standard_n32_0000/holdout-0/residual_0/run"
    p = out / "density-calibration/summary.json"
    text = p.read_text()
    meta = json.loads(text)
    try:
        meta["support_sha256"] = "0" * 64
        p.write_text(json.dumps(meta))
        with pytest.raises(ValueError):
            load_reference_density(
                p.parent,
                reference_run=reference,
            )
        meta = json.loads(text)
        meta["schema_version"] = True
        p.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="completed"):
            load_reference_density(
                p.parent,
                reference_run=reference,
            )
    finally:
        p.write_text(text)


def test_completed_units_are_verified_and_skipped(complete):
    out, _ = complete
    from super_deli.benchmarking.known_reference_study import engine

    root = out / "study"
    before = (root / "main-execution.json").read_bytes()
    result = engine(root, out / "invented-source").run("main", workers=1)
    assert result["completed"] and (root / "main-execution.json").read_bytes() == before


def test_prediction_leakage_and_unknown_calibration_classes_are_retained_failures(complete):
    import csv

    from super_deli.core.schema import config_from_dict
    from super_deli.models.reference_density_config import ReferenceDensityConfig
    from super_deli.workflows.reference_density import (
        calibrate_reference_density,
        predict_reference_density,
    )

    out, _ = complete
    folder = out / "study/main/known_linear_standard_n32_0000/holdout-0"
    ds = json.loads((folder / "input-configs.json").read_text())
    ref = folder / "residual_0/run"
    failed = out / "rejected_calibration_query"
    with pytest.raises(ValueError, match="disjoint"):
        predict_reference_density(
            config_from_dict(ds["calibration"]),
            reference_run=ref,
            calibration_run=out / "density-calibration",
            base_dir=folder / "calibration",
            output_dir=failed,
        )
    assert (failed / "failure.json").exists() and not (failed / "summary.json").exists()
    with (folder / "calibration-labels.csv").open(newline="") as stream:
        labels = list(csv.DictReader(stream))
    unknown = out / "unknown_labels.csv"
    with unknown.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["unit_id", "reference_label"])
        writer.writeheader()
        for row in labels:
            writer.writerow(
                {
                    **row,
                    "reference_label": "unseen"
                    if row["reference_label"] == "reference_0"
                    else row["reference_label"],
                }
            )
    failed = out / "rejected_unknown_calibration"
    with pytest.raises(ValueError, match="original reference"):
        calibrate_reference_density(
            config_from_dict(ds["training"]),
            config_from_dict(ds["calibration"]),
            reference_run=ref,
            calibration_labels=unknown,
            support_config=ReferenceDensityConfig(True),
            training_base_dir=folder / "training",
            calibration_base_dir=folder / "calibration",
            output_dir=failed,
        )
    assert (failed / "failure.json").exists() and not (failed / "summary.json").exists()
