"""Installed four-view reader/refit/report contracts on invented data only."""

import argparse
import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path


def smoke(output):
    from super_deli.benchmarking.individual_subsample_simulation import audit_law, generate
    from super_deli.benchmarking.signal_loss_source import digest
    from super_deli.cli import main
    from super_deli.io.config import read_json, write_json
    from super_deli.models.individual_subsample_config import IndividualSubsampleConfig
    from super_deli.workflows.integrative import run_integrative

    output = Path(output)
    output.mkdir()
    generator = Path(__file__).resolve().parents[1] / "multimodal/synthetic.py"
    data = output / "inputs"
    ds, model, truth = generate(
        data,
        parameter_seed=17561,
        innovation=17562,
        n=64,
        law="shape_balanced",
        tag="o",
        generator=generator,
    )
    audit_law(data)
    point = output / "point"
    run_integrative(ds, model_config=replace(model, n_bootstrap=1), base_dir=data, output_dir=point)
    before = {str(p.relative_to(point)): digest(p) for p in point.rglob("*") if p.is_file()}
    write_json(IndividualSubsampleConfig(2, 1601).to_dict(), output / "sampling.json")
    with redirect_stdout(io.StringIO()):
        assert (
            main(
                [
                    "subsample-integrative",
                    str(point),
                    "--stability-config",
                    str(output / "sampling.json"),
                    "--output-dir",
                    str(output / "subsamples"),
                ]
            )
            == 0
        )
    assert before == {str(p.relative_to(point)): digest(p) for p in point.rglob("*") if p.is_file()}
    summary = read_json(output / "subsamples/summary.json")
    assert (
        summary["successful"] == 2
        and summary["fraction"] == 0.8
        and summary["point_analysis_changed"] is False
        and summary["p_value"] is None
    )
    assert "Failed-outcome range" in (output / "subsamples/report.html").read_text()
    print(
        json.dumps(
            dict(
                passed=True,
                fixture=True,
                scientific_decision=False,
                native_searches=4,
                point_and_bootstrap_preserved=True,
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    smoke(p.parse_args().output_dir)
