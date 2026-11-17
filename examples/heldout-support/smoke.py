"""Installed four-view native source and separate predictive report on invented inputs."""

import argparse
import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path


def smoke(output):
    from super_deli.benchmarking.individual_subsample_simulation import generate
    from super_deli.benchmarking.signal_loss_source import digest
    from super_deli.cli import main
    from super_deli.io.config import read_json, write_json
    from super_deli.models.heldout_support_config import HeldoutSupportConfig
    from super_deli.models.individual_subsample_config import IndividualSubsampleConfig
    from super_deli.workflows.individual_subsampling import run_individual_subsampling
    from super_deli.workflows.integrative import run_integrative

    output = Path(output)
    output.mkdir()
    generator = Path(__file__).resolve().parents[1] / "multimodal/synthetic.py"
    ds, model, _ = generate(
        output / "inputs",
        parameter_seed=20861,
        innovation=20862,
        n=64,
        law="shape_balanced",
        tag="o",
        generator=generator,
    )
    run_integrative(
        ds,
        model_config=replace(model, n_bootstrap=0),
        base_dir=output / "inputs",
        output_dir=output / "point",
    )
    run_individual_subsampling(
        output / "point",
        stability_config=IndividualSubsampleConfig(2, 1601),
        output_dir=output / "subsample",
    )
    source = output / "subsample"
    before = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    write_json(HeldoutSupportConfig().to_dict(), output / "support.json")
    with redirect_stdout(io.StringIO()):
        assert (
            main(
                [
                    "assess-integrative-support",
                    str(source),
                    "--support-config",
                    str(output / "support.json"),
                    "--output-dir",
                    str(output / "predictive"),
                ]
            )
            == 0
        )
    assert before == {
        str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()
    }
    s = read_json(output / "predictive/summary.json")
    assert s["baseline_searches"] == s["successful_draws"] == 2
    assert s["full_cohort_mean"] is None and s["direction"] == "unavailable"
    assert s["p_value"] is None and not s["point_analysis_changed"]
    assert "same-sample point" in (output / "predictive/report.html").read_text()
    print(
        json.dumps(
            dict(
                passed=True,
                fixture=True,
                native_searches=3,
                baseline_searches=2,
                source_unchanged=True,
                scientific_decision=False,
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    smoke(p.parse_args().output_dir)
