"""Known-parameter reference isolation, paired fits and independent audit failures."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.grouped_boundary_analysis import (  # noqa: E402
    audit_study,
    tails,
    verify_reference,
)
from super_deli.benchmarking.grouped_boundary_diagnostics import run_case  # noqa: E402
from super_deli.benchmarking.grouped_boundary_metrics import count_association  # noqa: E402
from super_deli.benchmarking.grouped_boundary_reference import (  # noqa: E402
    CASES,
    known_model,
    load_model,
)
from super_deli.io.config import read_json  # noqa: E402
from super_deli.models.grouped_likelihood_config import GroupedLikelihoodStudyConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study():
    return GroupedLikelihoodStudyConfig.from_dict(
        read_json(ROOT / "examples/grouped-boundary-diagnostics/smoke.json")
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory, study):
    root = tmp_path_factory.mktemp("boundary-diagnostic-study")
    for case in CASES:
        result = run_case(study, case=case, output_dir=root / case)
        assert result["successful_fits"] == result["planned_fits"] == 3
    return root


def test_known_laws_and_pairing_without_repeated_observed_fits(completed, study):
    from super_deli.benchmarking.grouped_boundary_analysis import regenerated

    for case in CASES:
        base = completed / case / "dataset_0000"
        fitted, known = (load_model(base / name) for name in ("calibration", "known"))
        verify_reference(known, case)
        seed = read_json(base / "calibration/draw_0000.json")["simulation_seed"]
        assert seed == read_json(base / "known/draw_0000.json")["simulation_seed"]
        assert read_json(base / "known/summary.json")["shared_observed_fit"] is True
        assert (base / "known/observed.json").read_bytes() == (
            base / "calibration/observed.json"
        ).read_bytes()
        np.testing.assert_array_equal(fitted.counts, known.counts)
        assert all(np.isfinite(v).all() for v in regenerated(known, seed).values())
        replacement = known_model(fitted, case)
        np.testing.assert_array_equal(replacement.covariance, known.covariance)
        with pytest.raises(ValueError, match="five declared"):
            known_model(fitted, "complementary")
    assert study.calibration.model.n_init == 10


def test_no_refit_audit_all_laws(completed, tmp_path, monkeypatch):
    from sklearn.mixture import GaussianMixture

    def forbidden(*args, **kwargs):
        raise AssertionError("Audit must not refit a mixture")

    monkeypatch.setattr(GaussianMixture, "fit", forbidden)
    result = audit_study(completed, output_dir=tmp_path / "audit")
    assert result["no_model_refits"] is True
    assert result["new_unique_successful_fits"] == 15
    assert result["unique_pipeline_records_audited"] == 15
    assert len(result["summaries"]) == 20
    assert all(r["available"] == r["requested"] == 1 for r in result["summaries"])


def test_known_covariance_tampering_rejected(completed):
    model = load_model(completed / CASES[2] / "dataset_0000/known")
    damaged = replace(model, covariance=model.covariance + np.eye(8) * 0.01)
    with pytest.raises(AssertionError):
        verify_reference(damaged, CASES[2])


def test_seed_and_statistic_tampering_rejected(completed, tmp_path):
    import shutil

    copied = tmp_path / "copy"
    shutil.copytree(completed, copied)
    path = copied / CASES[0] / "dataset_0000/known/draw_0000.json"
    record = read_json(path)
    record["simulation_seed"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="ledger differs"):
        audit_study(copied, output_dir=tmp_path / "audit")


def test_failure_and_unattempted_draws_withhold_tails(completed, study):
    base = completed / CASES[2] / "dataset_0000/calibration"
    observed = read_json(base / "observed.json")
    record = read_json(base / "draw_0000.json")
    failed = {**record, "status": "failed", "error": "Retained injected failure"}
    for records, policy in (
        ([failed], study.calibration),
        ([record], replace(study.calibration, n_simulations=2)),
    ):
        for row in tails(observed, records, policy):
            assert row["p_value"] is row["reject"] is None
            assert row["failed_draws"] + row["unattempted_draws"] == 1


def test_count_association_is_descriptive():
    counts = np.repeat(np.arange(1, 5), 5)
    assert count_association(counts, counts) == pytest.approx(1)
    assert count_association(np.zeros(20, dtype=int), counts) == 0


def test_archived_branch_and_secondary_checker(completed, study, tmp_path):
    import importlib.util

    from super_deli.benchmarking.grouped_likelihood import run_grouped_likelihood_study

    archived = tmp_path / "archived"
    archived.mkdir()
    legacy = replace(study, random_seed=20261115)
    for case in CASES:
        run_grouped_likelihood_study(legacy, case=case, output_dir=archived / case)
    result = audit_study(completed, archived_study=archived, output_dir=tmp_path / "audit")
    assert result["unique_pipeline_records_audited"] == 25
    path = ROOT / "examples/grouped-boundary-diagnostics/check_tails.py"
    spec = importlib.util.spec_from_file_location("boundary_independent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked = module.check(completed, tmp_path / "audit/tails.csv", archived)
    assert checked["verified_unique_pipeline_statistics"] == 25
    assert checked["tails_checked"] == 30
    assert checked["withheld_tails"] == 0


def test_retained_reference_failure_withholds_actual_run(completed, study, tmp_path, monkeypatch):
    import shutil

    import super_deli.benchmarking.grouped_boundary_diagnostics as diagnostics

    root = tmp_path / "study"
    shutil.copytree(completed, root)
    case = CASES[0]
    shutil.rmtree(root / case)

    def fail(staging, declaration, directory, prefix, **kwargs):
        return {
            "status": "failed",
            "archive_prefix": prefix,
            "error": "Injected retained known-reference pipeline failure",
        }

    monkeypatch.setattr(diagnostics, "point_fitter", fail)
    result = run_case(study, case=case, output_dir=root / case)
    assert result["failed_fits"] == 1 and result["unattempted_fits"] == 0
    assert result["withheld_observations"] == 1
    assert (
        read_json(root / case / "dataset_0000/known/summary.json")["inference"]["p_value"] is None
    )
    audited = audit_study(root, output_dir=tmp_path / "audit")
    selected = [r for r in audited["summaries"] if r["case"] == case and r["arm"] == "known"]
    assert all(r["available"] == 0 and r["withheld_bounds"] == [0, 1] for r in selected)


def test_raw_null_failure_retains_unattempted_work(completed, study, tmp_path, monkeypatch):
    import shutil

    import super_deli.workflows.grouped_likelihood as workflow

    root = tmp_path / "study"
    shutil.copytree(completed, root)
    case = CASES[0]
    shutil.rmtree(root / case)

    def fail(*args, **kwargs):
        raise ValueError("Injected nonconverged raw-null estimator")

    monkeypatch.setattr(workflow, "run_grouped_likelihood", fail)
    result = run_case(study, case=case, output_dir=root / case)
    assert result["attempted_fits"] == 0 and result["unattempted_fits"] == 3
    assert result["withheld_observations"] == 1 and result["failed_observations"] == 1
    assert (root / case / "dataset_0000/failure.json").exists()
    audited = audit_study(root, output_dir=tmp_path / "audit")
    selected = [r for r in audited["summaries"] if r["case"] == case]
    assert all(r["available"] == 0 and r["withheld_bounds"] == [0, 1] for r in selected)
