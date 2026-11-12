"""Invented installed-package lifecycle; never an empirical or performance study."""

from pathlib import Path

from super_deli.benchmarking.residual_config import ResidualStudy
from super_deli.benchmarking.residual_report import report
from super_deli.benchmarking.residual_source import bind
from super_deli.benchmarking.residual_study import engine, gate
from super_deli.io.config import write_json


def reference_example(directory):
    import numpy as np

    from super_deli.benchmarking.residual_data import write_cohort, write_labels

    rng = np.random.default_rng(87231)
    y = np.repeat(np.arange(4), 4)
    x = rng.normal(size=(16, 12))
    x[:, 0] += np.where(y // 2 == 0, -3.0, 3.0)
    x[:, 5] += np.where(y % 2 == 0, -3.0, 3.0)
    root = Path(directory)
    root.mkdir()
    data = dict(morphology=x[:, :4], ecology=x[:, 4:], ecology_order=rng.permutation(16))
    dataset, units = write_cohort(data, root / "raw", tag="invented_reference")
    write_labels(units, y, root / "references.csv")
    return dataset


def smoke(output_dir, *, workers=1):
    out = Path(output_dir)
    out.mkdir()
    source, root = out / "invented-source", out / "study"
    source.mkdir()
    root.mkdir()
    write_json({"kind": "invented_residual_source_v1"}, source / "fixture.json")
    (root / "approved-plan.md").write_text(
        "Invented installation fixture, not scientific observations.\n"
    )
    bind(source, root, fixture=True)
    study = engine(root, source)
    study.freeze(ResidualStudy(fixture=True))
    study.run("preflight", workers=workers)
    study.audit("preflight")
    measured = gate(root, source, staging_bytes=0)
    if not measured["passed"]:
        raise ValueError("Invented fixture resource guard failed")
    study.run("main", workers=workers)
    study.audit("main")
    return report(root, source)
