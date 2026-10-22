"""Independent audit detects fitted-null ledger/parameter tampering without refits."""

# ruff: noqa: E402
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.empirical_analysis import audit_empirical_study
from super_deli.benchmarking.empirical_study import run_empirical_study, summarize_empirical
from super_deli.io.config import read_json, write_json
from super_deli.models.empirical_config import CASES, EmpiricalStudyConfig

pytestmark = pytest.mark.baseline


def test_empirical_study_audit_and_tampering(tmp_path, monkeypatch):
    from super_deli.workflows import integrative

    path = Path(__file__).resolve().parents[1] / "examples/empirical-calibration/smoke.json"
    study = EmpiricalStudyConfig.from_dict(read_json(path))
    root = tmp_path / "study"
    root.mkdir()
    for case in CASES:
        run_empirical_study(study, case=case, output_dir=root / case)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}

    def refuse(*args, **kwargs):
        raise AssertionError("Audit must not refit clustering")

    monkeypatch.setattr(integrative, "run_integrative", refuse)
    result = audit_empirical_study(root, output_dir=tmp_path / "audit")
    assert result["datasets"] == 4 and result["verified_successful_pipelines"] == 8
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(FileExistsError):
        audit_empirical_study(root, output_dir=tmp_path / "audit")
    with pytest.raises(ValueError, match="outside"):
        audit_empirical_study(root, output_dir=root / "audit")
    shard = root / CASES[0]
    summary = read_json(shard / "summary.json")
    summary["datasets"][0]["statistic"] += 1
    (shard / "datasets.jsonl").write_text(json.dumps(summary["datasets"][0]) + "\n")
    (shard / "summary.json").unlink()
    write_json(summarize_empirical(summary["datasets"], study, CASES[0]), shard / "summary.json")
    with pytest.raises(ValueError, match="inference differs"):
        audit_empirical_study(root, output_dir=tmp_path / "tampered")
