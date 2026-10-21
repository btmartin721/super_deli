"""Seven-case audit and paired raw-byte comparisons without optimizer calls."""

# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.grouped_analysis import audit_grouped_study
from super_deli.benchmarking.grouped_comparison import compare_grouped_studies
from super_deli.benchmarking.grouped_study import run_grouped_study
from super_deli.io.config import read_json
from super_deli.models.grouped_config import CASES, GroupedStudyConfig, study_cases

pytestmark = pytest.mark.baseline
ROOT = Path(__file__).resolve().parents[1]


def test_versioned_audit_pairing_and_immutable_sources(tmp_path, monkeypatch):
    from super_deli.models import grouped_conditional, grouped_gaussian
    from super_deli.workflows import integrative

    studies = []
    for name in ("grouped-calibration", "boundary-calibration"):
        c = GroupedStudyConfig.from_dict(read_json(ROOT / f"examples/{name}/smoke.json"))
        studies.append(replace(c, calibration=replace(c.calibration, n_simulations=1)))
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for case in CASES:
        run_grouped_study(studies[0], case=case, output_dir=old / case)
    for case in study_cases(studies[1]):
        run_grouped_study(studies[1], case=case, output_dir=new / case)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for root in (old, new)
        for p in root.rglob("*")
        if p.is_file()
    }

    def refuse(*args, **kwargs):
        raise AssertionError("Read-only analysis must not fit")

    monkeypatch.setattr(grouped_conditional, "minimize", refuse)
    monkeypatch.setattr(grouped_gaussian, "minimize", refuse)
    monkeypatch.setattr(integrative, "run_integrative", refuse)
    legacy = audit_grouped_study(old, output_dir=tmp_path / "legacy-audit")
    current = audit_grouped_study(new, output_dir=tmp_path / "new-audit")
    assert legacy["datasets"] == 4 and current["datasets"] == 7 and current["successful_fits"] == 14
    paired = compare_grouped_studies(old, new, output_dir=tmp_path / "paired")
    assert paired["paired_datasets"] == 4 and paired["current_available"] == 4
    assert paired["raw_byte_identity_verified"]
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(FileExistsError):
        compare_grouped_studies(old, new, output_dir=tmp_path / "paired")
    with pytest.raises(ValueError, match="outside"):
        compare_grouped_studies(old, new, output_dir=new / "paired")
    p = new / CASES[0] / "dataset_0000/observed/view_000.csv"
    p.write_text(p.read_text().replace("feature_0", "changed"))
    with pytest.raises(ValueError, match="byte-identical"):
        compare_grouped_studies(old, new, output_dir=tmp_path / "tampered")
