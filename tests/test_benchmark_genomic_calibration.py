"""Independent study seeds, numerical audits, tamper rejection and CLI packaging."""

# ruff: noqa: E402
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.genomic_analysis import audit_genomic_study
from super_deli.benchmarking.genomic_study import run_genomic_study
from super_deli.cli import main
from super_deli.io.config import read_json
from super_deli.models.genomic_config import CASES, GenomicStudyConfig

pytestmark = [
    pytest.mark.genomic,
    pytest.mark.skipif(
        importlib.util.find_spec("snpio") is None, reason="Install the genomics extra"
    ),
]
ROOT = Path(__file__).resolve().parents[1]


def test_study_reconstructs_all_fits_and_detects_ledger_tampering(tmp_path, monkeypatch):
    study = GenomicStudyConfig.from_dict(
        read_json(ROOT / "examples/genomic-calibration/smoke.json")
    )
    root = tmp_path / "study"
    root.mkdir()
    for case in CASES:
        result = run_genomic_study(study, case=case, output_dir=root / case)
        assert result["successful_fits"] == 3
    from super_deli.models import mixture

    monkeypatch.setattr(
        mixture,
        "select_gaussian_mixture",
        lambda *a, **k: pytest.fail("Audit must not refit clustering"),
    )
    result = audit_genomic_study(root, output_dir=tmp_path / "audit")
    assert result["datasets"] == 5 and result["verified_successful_pipelines"] == 15
    assert result["failed_fits"] == result["unattempted_fits"] == 0
    assert (
        main(
            ["analyze-genomic-calibration", str(root), "--output-dir", str(tmp_path / "cli-audit")]
        )
        == 0
    )
    path = root / CASES[0] / "dataset_0000/calibration/summary.json"
    value = read_json(path)
    value["successful_fits"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        audit_genomic_study(root, output_dir=tmp_path / "bad-audit")
