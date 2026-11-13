"""Invented held-out-group and installed known-only policy checks; never evidence."""

from pathlib import Path


def smoke(output_dir, *, workers=1):
    import numpy as np

    from super_deli.benchmarking.known_reference_config import KnownReferenceStudy, tasks
    from super_deli.benchmarking.known_reference_data import fold_data, seed_registry
    from super_deli.benchmarking.known_reference_report import report
    from super_deli.benchmarking.known_reference_source import bind
    from super_deli.benchmarking.known_reference_study import engine, gate
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.benchmarking.residual_data import write_cohort
    from super_deli.core.schema import config_from_dict
    from super_deli.io.config import read_json, write_json
    from super_deli.models.reference_density_config import ReferenceDensityConfig
    from super_deli.workflows.reference_density import (
        calibrate_reference_density,
        predict_reference_density,
    )

    out = Path(output_dir)
    out.mkdir()
    source = out / "invented-source"
    source.mkdir()
    write_json(dict(kind="invented_known_reference_source_v1"), source / "fixture.json")
    root = out / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text(
        "Invented known-only installation fixture; not scientific evidence.\n"
    )
    config = KnownReferenceStudy(fixture=True)
    bind(source, root, fixture=True)
    write_json(seed_registry(config), root / "seed-registry.json")
    study = engine(root, source)
    study.freeze(config)
    study.run("preflight", workers=workers)
    study.audit("preflight")
    gate(root, source, staging_bytes=0)
    study.run("main", workers=workers)
    study.audit("main")
    summary = report(root, source)
    task = tasks(config, "main")[0]
    parent = root / "main" / task["key"]
    fold = parent / "holdout-0"
    ds = read_json(fold / "input-configs.json")
    calibration = out / "density-calibration"
    calibrate_reference_density(
        config_from_dict(ds["training"]),
        config_from_dict(ds["calibration"]),
        reference_run=fold / "residual_0/run",
        calibration_labels=fold / "calibration-labels.csv",
        support_config=ReferenceDensityConfig(True),
        training_base_dir=fold / "training",
        calibration_base_dir=fold / "calibration",
        output_dir=calibration,
    )
    values = fold_data({p.stem: arrays(p) for p in (parent / "parent").glob("*.npz")}, 0)[
        "cohorts"
    ]["iid"]["x"]
    query, _ = write_cohort(
        dict(
            morphology=values[:, :4],
            ecology=values[:, 4:],
            ecology_order=np.arange(len(values))[::-1],
        ),
        out / "query",
        tag="independent_fixture_query",
    )
    prediction = predict_reference_density(
        query,
        reference_run=fold / "residual_0/run",
        calibration_run=calibration,
        base_dir=out / "query",
        output_dir=out / "prediction",
    )
    return dict(
        passed=True,
        invented_only=True,
        study=summary,
        prediction_status=prediction["status"],
        unknown_training_examples=0,
    )
