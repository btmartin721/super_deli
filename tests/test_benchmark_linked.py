"""Paired observed byte identity, independent audits and comparator rejection."""

# ruff: noqa: E402
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.genomic_comparison import compare_genomic_studies
from super_deli.benchmarking.genomic_study import run_genomic_study
from super_deli.benchmarking.linked_analysis import audit_linked_study
from super_deli.benchmarking.linked_study import run_linked_study
from super_deli.cli import main
from super_deli.io.config import read_json
from super_deli.models.genomic_config import GenomicStudyConfig
from super_deli.models.linked_config import CASES, PAIRED_CASES, LinkedStudyConfig

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.genomic,
    pytest.mark.skipif(importlib.util.find_spec("snpio") is None, reason="Install genomics extra"),
]


def test_full_study_audit_and_byte_paired_comparison(tmp_path, monkeypatch):
    linked = LinkedStudyConfig.from_dict(read_json(ROOT / "examples/linked-calibration/smoke.json"))
    legacy = GenomicStudyConfig.from_dict(
        read_json(ROOT / "examples/genomic-calibration/smoke.json")
    )
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for case in CASES:
        result = run_linked_study(linked, case=case, output_dir=new / case)
        assert result["successful_fits"] == 3
        if case in PAIRED_CASES:
            run_genomic_study(legacy, case=case, output_dir=old / case)
    from super_deli.models import mixture

    monkeypatch.setattr(
        mixture, "select_gaussian_mixture", lambda *a, **k: pytest.fail("Audit must not refit")
    )
    result = audit_linked_study(new, output_dir=tmp_path / "audit")
    assert result["verified_successful_pipelines"] == 18 and result["datasets"] == 6
    paired = compare_genomic_studies(old, new, output_dir=tmp_path / "paired")
    assert (
        paired["paired_datasets"] == paired["current_available"] == paired["legacy_available"] == 4
    )
    assert (
        main(
            [
                "compare-genomic-calibration",
                str(old),
                "--current-study",
                str(new),
                "--output-dir",
                str(tmp_path / "cli-paired"),
            ]
        )
        == 0
    )
    raw = new / "common_hwe/dataset_0000/observed/genotypes.phy"
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="raw bytes"):
        compare_genomic_studies(old, new, output_dir=tmp_path / "bad-paired")
    path = new / "common_hwe/dataset_0000/calibration/summary.json"
    value = read_json(path)
    value["successful_fits"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        audit_linked_study(new, output_dir=tmp_path / "bad-audit")
