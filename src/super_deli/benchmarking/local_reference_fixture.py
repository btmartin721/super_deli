"""Explicitly invented installation/test specimens, never scientific evidence."""

from pathlib import Path


def matrices():
    import numpy as np

    from super_deli.models.matched_preprocessing import prepare_views

    rng = np.random.default_rng(883471)
    y = np.repeat(np.arange(4), 8)
    x = rng.normal(size=(32, 12)) + 2 * y[:, None]
    yc = np.repeat(np.arange(4), 20)
    xc = rng.normal(size=(80, 12)) + 2 * yc[:, None]
    _, prep = prepare_views(x, 4)
    return x, y, xc, yc, prep


def reference_fixture(root):
    import numpy as np

    from super_deli.benchmarking.residual_config import model_for
    from super_deli.benchmarking.residual_data import write_cohort, write_labels
    from super_deli.workflows.residual_reference import train_matched_residual

    root = Path(root)
    x, y, xc, yc, _ = matrices()
    cohorts = {}
    for name, data, truth in (("train", x, y), ("cal", xc, yc), ("test", xc + 2, yc)):
        folder = root / name
        folder.mkdir()
        ds, ids = write_cohort(
            dict(morphology=data[:, :4], ecology=data[:, 4:], ecology_order=np.arange(len(data))),
            folder / "raw",
            tag=name,
        )
        write_labels(ids, truth, folder / "labels.csv")
        cohorts[name] = (ds, folder, ids)
    train_matched_residual(
        cohorts["train"][0],
        labels_path=cohorts["train"][1] / "labels.csv",
        model_config=model_for("residual", 0),
        base_dir=cohorts["train"][1] / "raw",
        output_dir=root / "model",
    )
    return root, cohorts


def study_fixture(output_dir, *, workers=1):
    """Complete sealed installation lifecycle on explicitly invented source records."""
    from super_deli.benchmarking.abstention_run import run_unit
    from super_deli.benchmarking.local_reference_config import LocalReferenceStudy
    from super_deli.benchmarking.local_reference_report import report
    from super_deli.benchmarking.local_reference_source import bind
    from super_deli.benchmarking.local_reference_study import engine, gate, seed_registry
    from super_deli.io.config import write_json

    out = Path(output_dir)
    out.mkdir()
    source, root = out / "invented-source", out / "study"
    source.mkdir()
    root.mkdir()
    (source / "main").mkdir()
    task = dict(
        key="invented_predecessor",
        case="linear_standard",
        n=32,
        fixture=True,
        replicate=0,
        seed_path=[871431, 741],
    )
    result = run_unit(
        dict(
            kind="fresh",
            task=task,
            phase_root=str(source / "main"),
            stop_file=str(source / "STOP.json"),
        )
    )
    if result["failed_points"]:
        raise ValueError("Invented predecessor fixture failed")
    write_json(
        dict(kind="invented_local_reference_source_v1", observations=["main/" + task["key"]]),
        source / "fixture.json",
    )
    (root / "approved-plan.md").write_text(
        "Invented class-local installation fixture; not scientific evidence.\n"
    )
    config = LocalReferenceStudy(fixture=True)
    bind(source, root, fixture=True)
    write_json(seed_registry(config), root / "seed-registry.json")
    study = engine(root, source)
    study.freeze(config)
    study.run("preflight", workers=workers)
    study.audit("preflight")
    gate(root, source, staging_bytes=0)
    study.run("main", workers=workers)
    study.audit("main")
    return report(root, source)


def smoke(output_dir, *, workers=1):
    import json
    import subprocess
    import sys

    import numpy as np

    from super_deli.io.config import write_json
    from super_deli.models.local_reference_config import LocalReferenceSupportConfig
    from super_deli.workflows.local_reference_support import load_local_reference_support

    out = Path(output_dir)
    out.mkdir()
    api = out / "api"
    api.mkdir()
    root, cohorts = reference_fixture(api)
    cfg = api / "support-config.json"
    write_json(LocalReferenceSupportConfig(True).to_dict(), cfg)
    cal = api / "support"
    commands = [
        [
            "calibrate-local-reference-support",
            str(cohorts["cal"][1] / "raw/dataset.json"),
            "--training-dataset",
            str(cohorts["train"][1] / "raw/dataset.json"),
            "--reference-run",
            str(api / "model"),
            "--labels",
            str(cohorts["cal"][1] / "labels.csv"),
            "--support-config",
            str(cfg),
            "--output-dir",
            str(cal),
        ],
        [
            "predict-local-reference-support",
            str(cohorts["test"][1] / "raw/dataset.json"),
            "--reference-run",
            str(api / "model"),
            "--calibration-run",
            str(cal),
            "--output-dir",
            str(api / "prediction"),
        ],
    ]
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "super_deli", *command], cwd=api, capture_output=True, text=True
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:])
        if json.loads(completed.stdout)["status"] != "complete":
            raise ValueError("Incomplete installed workflow")
    model, _, _, _, _ = load_local_reference_support(cal, reference_run=api / "model")
    with np.load(api / "prediction/support-predictions.npz", allow_pickle=False) as values:
        np.testing.assert_array_equal(
            values["probabilities"], model.predict_matrix(matrices()[2] + 2)["probabilities"]
        )
    code = (
        "import sys;"
        "from super_deli.workflows.local_reference_support import load_local_reference_support;"
        "load_local_reference_support(sys.argv[1],reference_run=sys.argv[2]);"
        'assert "torch" not in sys.modules and "snpio" not in sys.modules'
    )
    subprocess.run([sys.executable, "-c", code, str(cal), str(api / "model")], cwd=api, check=True)
    study = study_fixture(out / "benchmark", workers=workers)
    return dict(passed=True, invented_only=True, probability_parity=True, study=study)
