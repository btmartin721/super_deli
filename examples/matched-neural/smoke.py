"""Installed paired CLI/API and full-settings engineering protocol; no biological claim."""

import argparse
from pathlib import Path

from super_deli.benchmarking.matched_config import StudyConfig
from super_deli.cli import main
from super_deli.io.config import read_json, write_json


def smoke(out):
    out.mkdir()
    root = out / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented installed engineering fixture")
    config = out / "fixture.json"
    write_json(StudyConfig(fixture=True).to_dict(), config)
    for action in (
        "freeze",
        "preflight",
        "audit-preflight",
        "gate",
        "main",
        "audit-main",
        "report",
    ):
        args = [
            "benchmark-matched-neural",
            action,
            "--output-dir",
            str(root),
            "--study-config",
            str(config),
        ]
        if action == "gate":
            args.extend(["--staging-bytes", "0"])
        assert main(args) == 0
    point = root / "main/four_complementary_n32_0000"
    cfg = read_json(point / "paired_0/model-config.json")
    cfg["n_subsamples"] = 0
    write_json(cfg, out / "model.json")
    assert (
        main(
            [
                "cluster-matched-neural",
                str(point / "raw/dataset.json"),
                "--model-config",
                str(out / "model.json"),
                "--output-dir",
                str(out / "cli-point"),
            ]
        )
        == 0
    )
    from super_deli.io.config import load_config
    from super_deli.io.tabular import ingest_tabular
    from super_deli.workflows.matched_neural import load_matched_neural_model

    loaded = ingest_tabular(
        load_config(point / "raw/dataset.json"),
        base_dir=point / "raw",
        output_dir=out / "reload-ingestion",
    )
    import numpy as np

    model = load_matched_neural_model(out / "cli-point")
    with np.load(out / "cli-point/point.npz") as a:
        np.testing.assert_array_equal(model.encode(loaded.tables)["scores"], a["scores"])
    assert read_json(root / "main-audit.json")["actual_workflow_attempts"] == 83
    assert read_json(root / "aggregate.json")["fixture"]
    assert (
        read_json(out / "cli-point/summary.json")["species_significance"]["status"]
        == "not_computed"
    )
    print("Installed matched neural CLI, numeric reload and gated fixture passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    smoke(p.parse_args().output_dir)
