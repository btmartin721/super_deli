"""Invented installed-package contracts; no scientific performance decision."""

import argparse
import importlib.util
import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path


def invented_generator():
    path = Path(__file__).resolve().parents[1] / "multimodal/synthetic.py"
    spec = importlib.util.spec_from_file_location("retention_fixture_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.invented


def smoke(output):
    from super_deli.benchmarking.full_retention_run import run_native
    from super_deli.benchmarking.multimodal_validation import equal_point_arrays
    from super_deli.cli import main
    from super_deli.io.config import read_json
    from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig

    output = Path(output)
    output.mkdir()
    source = output / "source"
    dataset, _ = invented_generator()(source, n=64, law="complementary", seed=74101, draws=0)
    before = (source / "analysis.json").read_bytes()
    configured = output / "retention.json"
    with redirect_stdout(io.StringIO()):
        assert (
            main(
                [
                    "full-retention-config",
                    str(source / "analysis.json"),
                    "--arm",
                    "combined",
                    "--min-cluster-size",
                    "9",
                    "--output-config",
                    str(configured),
                ]
            )
            == 0
        )
    assert before == (source / "analysis.json").read_bytes()
    analysis = MultimodalAnalysisConfig.from_dict(read_json(configured))
    model = analysis.arms[-1].model
    assert model.n_pcs == 8 and model.fusion_policy == "retain_all_v1"
    row = run_native(
        dataset,
        replace(model, n_bootstrap=2),
        source,
        output / "observed",
        law="complementary",
        truth=read_json(source / "truth-evaluation-only.json"),
    )
    assert row["status"] == "complete" and row["audit"]["passed"]
    comparison = output / "comparison"
    with redirect_stdout(io.StringIO()):
        assert (
            main(
                [
                    "analyze-matched-multimodal",
                    str(source / "dataset.json"),
                    "--analysis-config",
                    str(configured),
                    "--output-dir",
                    str(comparison),
                ]
            )
            == 0
        )
    summary = read_json(comparison / "summary.json")
    assert summary["status"] == "complete" and summary["inference"]["p_value"] is None
    assert summary["models"][-1]["fusion_policy"] == "retain_all_v1"
    assert "all 8 selected view-PC coordinates" in (comparison / "report.html").read_text()
    parity = equal_point_arrays(comparison / "arms/combined", output / "observed")
    result = dict(
        passed=True,
        fixture=True,
        scientific_decision=False,
        selected_dimensions=8,
        native_and_unified_arrays=parity,
        audited_draws=row["audit"].get("draws"),
        component_checks=row["audit"]["component_checks"],
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    smoke(p.parse_args().output_dir)
