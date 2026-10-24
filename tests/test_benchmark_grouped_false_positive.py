"""Independent interval inversion, missing outcomes, paired arms and audit tampering."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from scipy.stats import binom  # noqa: E402

from super_deli.benchmarking.grouped_boundary_diagnostics import run_case  # noqa: E402
from super_deli.benchmarking.grouped_boundary_reference import CASES  # noqa: E402
from super_deli.benchmarking.grouped_false_positive import (  # noqa: E402
    audit_study,
    exact_interval,
    paired_decisions,
    rejection_rate,
    summarize_tails,
)
from super_deli.io.config import read_json  # noqa: E402
from super_deli.models.grouped_likelihood_config import GroupedLikelihoodStudyConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("events,n", [(0, 500), (25, 500), (500, 500), (3, 25)])
def test_exact_interval_independent_binomial_inversion(events, n):
    lo, hi = exact_interval(events, n)
    if events:
        assert binom.sf(events - 1, n, lo) == pytest.approx(0.025, abs=1e-12)
    else:
        assert lo == 0
    if events < n:
        assert binom.cdf(events, n, hi) == pytest.approx(0.025, abs=1e-12)
    else:
        assert hi == 1


def test_multiplicity_and_unavailable_outcomes():
    r = rejection_rate(25, 500, 500)
    assert r["mcse_among_available"] == pytest.approx(0.009746794344808964)
    assert r["bonferroni_familywise_95"][0] < r["exact_interval_95"][0]
    assert r["bonferroni_familywise_95"][1] > r["exact_interval_95"][1]
    assert rejection_rate(0, 0, 500)["exact_interval_95"] is None
    assert rejection_rate(0, 0, 500)["withheld_outcome_bounds"] == [0, 1]
    partial = rejection_rate(1, 8, 10)
    assert partial["withheld_outcome_bounds"] == [0.1, 0.3]
    assert partial["all_requested_pointwise_envelope"][0] == exact_interval(1, 10)[0]
    assert partial["all_requested_pointwise_envelope"][1] == exact_interval(3, 10)[1]


@pytest.mark.parametrize(
    "events,n,error", [(-1, 10, 0.05), (11, 10, 0.05), (1.0, 10, 0.05), (1, 10, 0)]
)
def test_invalid_intervals_fail_closed(events, n, error):
    with pytest.raises(ValueError):
        exact_interval(events, n, error=error)


def test_pairing_uses_discordances_and_excludes_incomplete_pairs():
    a = {0: True, 1: False, 2: False, 3: None}
    b = {0: False, 1: True, 2: True, 3: True}
    r = paired_decisions(a, b)
    assert r["available_pairs"] == 3 and r["withheld_pairs"] == 1
    assert r["known_only_rejections"] == 2
    assert r["fitted_only_rejections"] == 1
    assert r["known_minus_fitted"] == pytest.approx(1 / 3)
    assert r["conservative_exact_difference_interval"][0] < 1 / 3
    assert r["conservative_exact_difference_interval"][1] > 1 / 3
    assert paired_decisions({0: False}, {0: False})["conservative_exact_difference_interval"] != [
        0,
        0,
    ]
    with pytest.raises(ValueError, match="identical"):
        paired_decisions(a, {0: False})


def test_primary_summary_rejects_duplicate_and_missing_identities():
    rows = [
        {
            "case": c,
            "arm": a,
            "repetition": 0,
            "statistic_kind": "likelihood",
            "p_value": 0.2,
            "reject": False,
        }
        for c in CASES
        for a in ("fitted", "known")
    ]
    requested = dict.fromkeys(CASES, 1)
    r = summarize_tails(rows, requested)
    assert len(r["primary_rates"]) == 10
    assert len(r["paired_decisions"]) == 5
    for damaged in (rows[:-1], [*rows, rows[0]]):
        with pytest.raises(ValueError, match="identity"):
            summarize_tails(damaged, requested)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("larger-null-smoke")
    study = GroupedLikelihoodStudyConfig.from_dict(
        read_json(ROOT / "examples/grouped-false-positive/smoke.json")
    )
    for case in CASES:
        result = run_case(study, case=case, output_dir=root / case)
        assert result["successful_fits"] == 3
    return root


def test_streaming_audit_matches_existing_auditor_without_refitting(
    completed, tmp_path, monkeypatch
):
    from sklearn.mixture import GaussianMixture

    from super_deli.benchmarking.grouped_boundary_analysis import audit_study as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("No model fitting is permitted in an audit")

    monkeypatch.setattr(GaussianMixture, "fit", forbidden)
    actual = audit_study(completed, output_dir=tmp_path / "stream")
    previous = legacy(completed, output_dir=tmp_path / "legacy")
    assert actual["unique_pipeline_records_audited"] == 15
    assert actual["archive_members"] == previous["archive_members"]
    assert actual["fit_accounting"]["successful_fits"] == 15
    assert len(actual["primary_rates"]) == 10


def test_parallel_and_sequential_audits_agree(completed, tmp_path):
    left = audit_study(completed, output_dir=tmp_path / "one", workers=1)
    right = audit_study(completed, output_dir=tmp_path / "two", workers=2)
    assert left == right


def test_seed_tampering_is_rejected_before_reporting(completed, tmp_path):
    root = tmp_path / "damaged"
    shutil.copytree(completed, root)
    ledger = root / CASES[0] / "datasets.jsonl"
    row = json.loads(ledger.read_text())
    row["data_seed"] += 1
    ledger.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="seed differs"):
        audit_study(root, output_dir=tmp_path / "audit")


def test_excess_concurrency_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="one to four"):
        audit_study(tmp_path, output_dir=tmp_path / "audit", workers=5)


def test_retained_raw_fit_failure_is_withheld_without_replacement(completed, tmp_path):
    root = tmp_path / "raw-failure"
    shutil.copytree(completed, root)
    case = root / CASES[0]
    base = case / "dataset_0000"
    for arm in ("calibration", "known"):
        shutil.rmtree(base / arm)
    (base / "failure.json").write_text(json.dumps({"error": "Injected pre-point raw-null failure"}))
    ledger = json.loads((case / "datasets.jsonl").read_text())
    ledger.update(
        status="failed",
        attempted_fits=0,
        successful_fits=0,
        failed_fits=0,
        unattempted_fits=ledger["planned_fits"],
    )
    (case / "datasets.jsonl").write_text(json.dumps(ledger) + "\n")
    summary = read_json(case / "summary.json")
    for key in (
        "planned_fits",
        "attempted_fits",
        "successful_fits",
        "failed_fits",
        "unattempted_fits",
    ):
        summary[key] = ledger[key]
    (case / "summary.json").write_text(json.dumps(summary))
    result = audit_study(root, output_dir=tmp_path / "failure-audit")
    assert result["unique_pipeline_records_audited"] == 12
    unavailable = [r for r in result["primary_rates"] if r["case"] == CASES[0]]
    assert all(r["available"] == 0 and r["withheld"] == 1 for r in unavailable)
    assert all(r["withheld_outcome_bounds"] == [0, 1] for r in unavailable)
