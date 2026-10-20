"""Study audit regenerates raw inputs and detects ledger/byte tampering without refits."""

# ruff: noqa: E402
import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("snpio")

from super_deli.benchmarking.pipeline_analysis import audit_pipeline_study
from super_deli.benchmarking.pipeline_study import run_pipeline_study, summarize_records
from super_deli.io.config import read_json, write_json
from super_deli.models.pipeline_config import CASES, PipelineStudyConfig

pytestmark = pytest.mark.genomic


def test_study_audit_without_fitting_and_tamper_detection(tmp_path, monkeypatch):
    from super_deli.workflows import integrative

    example = Path(__file__).resolve().parents[1] / "examples/pipeline-calibration/study.json"
    study = PipelineStudyConfig.from_dict(read_json(example))
    policy = study.calibration
    study = replace(
        study,
        n_datasets=1,
        calibration=replace(
            policy,
            n_simulations=1,
            model=replace(policy.model, n_init=1, max_k=2),
            reference=replace(policy.reference, n_units=20, n_loci=16),
        ),
    )
    root = tmp_path / "study"
    root.mkdir()
    for case in CASES:
        run_pipeline_study(study, case=case, output_dir=root / case)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}

    def refuse(*args, **kwargs):
        raise AssertionError("Auditor must not refit the model")

    monkeypatch.setattr(integrative, "run_integrative", refuse)
    summary = audit_pipeline_study(root, output_dir=tmp_path / "analysis")
    assert summary["datasets"] == 4
    assert summary["verified_successful_pipelines"] == summary["planned_fits"] == 8
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(FileExistsError):
        audit_pipeline_study(root, output_dir=tmp_path / "analysis")
    with pytest.raises(ValueError, match="outside"):
        audit_pipeline_study(root, output_dir=root / "analysis")
    ledger = root / CASES[0] / "dataset_0000/calibration/draw_0000.json"
    record = read_json(ledger)
    record["simulation_seed"] += 1
    ledger.unlink()
    write_json(record, ledger)
    with pytest.raises(ValueError, match="identity or seed"):
        audit_pipeline_study(root, output_dir=tmp_path / "tampered")

    record["simulation_seed"] -= 1
    ledger.unlink()
    write_json(record, ledger)
    shard = root / CASES[0]
    completed = read_json(shard / "summary.json")
    completed["datasets"][0]["statistic"] += 1
    (shard / "datasets.jsonl").write_text(json.dumps(completed["datasets"][0]) + "\n")
    (shard / "summary.json").unlink()
    write_json(summarize_records(completed["datasets"], study, CASES[0]), shard / "summary.json")
    with pytest.raises(ValueError, match="inference mismatch"):
        audit_pipeline_study(root, output_dir=tmp_path / "tampered-statistic")
