"""Explicit invented installation lifecycle; never empirical/performance evidence."""

from pathlib import Path

from super_deli.benchmarking.abstention_config import AbstentionStudy
from super_deli.benchmarking.abstention_data import seed_registry
from super_deli.benchmarking.abstention_report import report
from super_deli.benchmarking.abstention_source import bind
from super_deli.benchmarking.abstention_study import engine, gate
from super_deli.io.config import write_json


def smoke(output_dir, *, workers=1):
    out = Path(output_dir)
    out.mkdir()
    source, root = out / "invented-source", out / "study"
    source.mkdir()
    root.mkdir()
    write_json(dict(kind="invented_abstention_source_v1"), source / "fixture.json")
    (root / "approved-plan.md").write_text(
        "Invented installation fixture; not scientific observations.\n"
    )
    config = AbstentionStudy(fixture=True)
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
