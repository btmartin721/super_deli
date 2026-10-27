"""Installed grouped CLI and fit-free native evidence audit outside the checkout."""

import argparse
import json
from pathlib import Path


def main():
    from super_deli.benchmarking.grouped_analysis_study import model_arms
    from super_deli.benchmarking.grouped_study import simulate_grouped
    from super_deli.cli import main as cli
    from super_deli.io.config import read_json, write_json
    from super_deli.models.grouped_null import write_raw_views

    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    a.output_dir.mkdir()
    values, layout, truth = simulate_grouped("complementary", 32, 79501)
    write_raw_views(values, layout, a.output_dir / "raw")
    write_json(model_arms(2).to_dict(), a.output_dir / "analysis.json")
    status = cli(
        [
            "analyze-grouped-tabular",
            str(a.output_dir / "raw/dataset.json"),
            "--analysis-config",
            str(a.output_dir / "analysis.json"),
            "--output-dir",
            str(a.output_dir / "analysis"),
        ]
    )
    assert status == 0
    summary = read_json(a.output_dir / "analysis/summary.json")
    assert len(summary["models"]) == 6 and len(summary["comparisons"]) == 15
    assert all(g["ecology_count"] == 1 for g in summary["groups"])
    assert summary["models"][-1]["stability"]["requested"] == 2
    import numpy as np

    from super_deli.benchmarking.grouped_analysis_audit import audit_observation

    task = dict(
        case="complementary",
        n=32,
        data_seed=79501,
        draws=2,
        replay=False,
        key="installed_smoke",
        phase="installed_smoke",
        dataset=0,
    )
    write_json(layout, a.output_dir / "layout.json")
    write_json(
        {"task": task, "status": "complete", "error": None, "replays": []},
        a.output_dir / "validation.json",
    )
    with (a.output_dir / "truth.npz").open("xb") as handle:
        np.savez_compressed(handle, labels=truth)
    audit = audit_observation(a.output_dir, task)
    assert audit["searches"] == 28 and not audit["failures"]
    print(json.dumps({"installed_grouped_analysis_smoke": "passed", "arms": 6, "comparisons": 15}))


if __name__ == "__main__":
    main()
