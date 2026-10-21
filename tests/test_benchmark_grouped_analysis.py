"""Audit raw observations, null scores, failures and numerical artifacts without refits."""

# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
from super_deli.benchmarking.grouped_analysis import audit_grouped_study
from super_deli.benchmarking.grouped_study import run_grouped_study
from super_deli.io.config import read_json
from super_deli.models.grouped_config import CASES, GroupedStudyConfig

pytestmark = pytest.mark.baseline


def test_read_only_audit_preserves_failed_datasets_and_detects_tampering(tmp_path, monkeypatch):
    from super_deli.workflows import integrative

    path = Path(__file__).resolve().parents[1] / "examples/grouped-calibration/smoke.json"
    study = GroupedStudyConfig.from_dict(read_json(path))
    study = replace(study, n_units=64)
    root = tmp_path / "study"
    root.mkdir()
    for case in CASES:
        run_grouped_study(study, case=case, output_dir=root / case)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}

    def refuse(*args, **kwargs):
        raise AssertionError("Audit must not refit clustering or null models")

    monkeypatch.setattr(integrative, "run_integrative", refuse)
    import super_deli.models.grouped_gaussian as gaussian

    monkeypatch.setattr(gaussian, "minimize", refuse)
    result = audit_grouped_study(root, output_dir=tmp_path / "audit")
    assert result["datasets"] == 4 and result["planned_fits"] == 12
    assert result["attempted_fits"] + result["unattempted_fits"] == 12
    assert result["verified_successful_pipelines"] > 0
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(FileExistsError):
        audit_grouped_study(root, output_dir=tmp_path / "audit")
    with pytest.raises(ValueError, match="outside"):
        audit_grouped_study(root, output_dir=root / "audit")
    raw = root / CASES[0] / "dataset_0000/observed/view_000.csv"
    raw.write_text(raw.read_text().replace("feature_0", "tampered_feature"))
    with pytest.raises(ValueError, match="raw input differs"):
        audit_grouped_study(root, output_dir=tmp_path / "tampered")
