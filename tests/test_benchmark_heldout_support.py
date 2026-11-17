"""Independent audit, minority recovery and conservative event denominators."""

# ruff: noqa: E402
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("snpio")
from super_deli.benchmarking.heldout_audit import audit_report
from super_deli.benchmarking.heldout_metrics import recovery
from super_deli.benchmarking.heldout_report import event_rate
from super_deli.benchmarking.heldout_run import parent
from super_deli.io.config import read_json

pytestmark = pytest.mark.baseline


def test_minority_jaccard_not_hidden_by_global_ari():
    truth = np.repeat(np.arange(4), [29, 16, 13, 6])
    labels = truth.copy()
    labels[26:29] = 3
    result = recovery(truth, labels)
    assert result["ari"] > 0.8 and result["rare_recall"] == 1
    assert result["rare_precision"] == 2 / 3 and not result["correct"]
    r = event_rate([True, False, None, None])
    assert r["identification_lower"] == 0.25 and r["identification_upper"] == 0.75
    assert r["simultaneous_40_rate_95_interval"][0] <= r["marginal_exact_95_interval"][0]


def test_full_four_view_parent_and_independent_audit_tampering(tmp_path):
    generator = Path(__file__).resolve().parents[1] / "examples/multimodal/synthetic.py"
    task = dict(index=0, cell=5, n=64, design="rare_six", counts=[29, 16, 13, 6], seed=20831)
    result = parent(task, tmp_path / "case", generator, draws=1)
    assert result["planned_searches"] == 3 and result["attempted_searches"] == 3
    assert result["predictive_audit"]["passed"] and result["predictive_audit"]["new_fits"] == 0
    report = tmp_path / "case/predictive"
    original = (report / "individual_scores.npz").read_bytes()
    with np.load(report / "individual_scores.npz", allow_pickle=False) as a:
        values = {k: a[k].copy() for k in a.files}
    values["requested"][0] += 1
    np.savez(report / "individual_scores.npz", **values)
    with pytest.raises(AssertionError):
        audit_report(report, write=False)
    (report / "individual_scores.npz").write_bytes(original)
    assert audit_report(report, write=False) == read_json(report / "independent-audit.json")
