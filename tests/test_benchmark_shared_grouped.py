"""Shared-family grouped refits, fixed counts, independent audit and failed inference."""

# ruff: noqa: E402
import io
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.grouped_analysis import audit_grouped_study, point, verify_model
from super_deli.benchmarking.grouped_study import run_grouped_study, simulate_grouped
from super_deli.io.config import read_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.grouped_config import (
    GroupedCalibrationConfig,
    GroupedStudyConfig,
    study_cases,
)
from super_deli.models.grouped_null import fit_null, simulate_null, write_raw_views
from super_deli.workflows.grouped_calibration import run_grouped_calibration

pytestmark = pytest.mark.baseline
ROOT = Path(__file__).resolve().parents[1]


def policy(family="tied", level="locality"):
    c = GroupedCalibrationConfig.from_dict(
        read_json(ROOT / "examples/shared-grouped-calibration/calibration.json")
    )
    return replace(
        c,
        n_simulations=2,
        model=replace(c.model, covariance_type=family, target_level=level, n_init=1, max_k=2),
    )


@pytest.mark.parametrize("family", ["full", "tied"])
@pytest.mark.parametrize("level", ["locality", "population"])
def test_full_raw_refits_preserve_unequal_counts_and_audit_family(
    tmp_path, monkeypatch, family, level
):
    from super_deli.benchmarking.pipeline_analysis import audit_pipeline_point
    from super_deli.workflows import integrative

    c = policy(family, level)
    values, layout, _ = simulate_grouped("correlated_gaussian", 64, 193, target_level=level)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "raw").iterdir()}
    runner, calls = integrative.run_integrative, []

    def counted(*args, **kwargs):
        calls.append(kwargs["model_config"])
        return runner(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", counted)
    out = tmp_path / "calibration"
    result = run_grouped_calibration(dataset, config=c, base_dir=tmp_path / "raw", output_dir=out)
    assert result["successful_fits"] == 3 and len(calls) == 3
    assert all(m.covariance_type == family and m.n_bootstrap == 0 for m in calls)
    assert all(m.random_seed == c.model.random_seed for m in calls)
    assert ("fixed before analysis" in result["scope"]) == (family == "tied")
    evidence = read_json(out / "evidence.json")
    assert ("prespecified shared covariance" in evidence["method"]) == (family == "tied")
    assert "nonregular" in result["inference"]["p_value_kind"]
    fitted, draw, n, d = verify_model(tmp_path, out, values, layout, c)
    assert (n, d) == (64, 8) and fitted["counts"] == layout["counts"]
    assert set(fitted["counts"]) == {1, 2, 3, 4}
    with ZipFile(out / "pipeline_artifacts.zip") as archive:
        point(archive, result["point"], c, values, fitted)
        other = replace(
            c, model=replace(c.model, covariance_type="full" if family == "tied" else "tied")
        )
        with pytest.raises(ValueError, match="configuration mismatch"):
            audit_pipeline_point(archive, result["point"], other)
        means = []
        for record in read_json(out / "null_replicates.json"):
            sample = draw(record["simulation_seed"])
            assert sample["ecological"].shape == (64, 4)
            assert sample["morphological"].shape == (sum(layout["counts"]), 4)
            point(archive, record, c, sample, fitted)
            prefix = record["archive_prefix"] + "/run/"
            with np.load(
                io.BytesIO(archive.read(prefix + "view_000/parameters/pca.npz"))
            ) as arrays:
                means.append(arrays["feature_means"].copy())
        assert not np.array_equal(*means)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_raw_null_estimates_and_draws_do_not_depend_on_gmm_family(tmp_path):
    values, layout, _ = simulate_grouped("zero_conditional_gaussian", 64, 193)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    tables = ingest_tabular(
        dataset, base_dir=tmp_path / "raw", output_dir=tmp_path / "ingestion"
    ).tables
    full, tied = (fit_null(dataset, tables, policy(f)) for f in ("full", "tied"))
    assert full.layout == tied.layout
    for name, array in full.model_arrays().items():
        np.testing.assert_array_equal(array, tied.model_arrays()[name])
    for kind, array in simulate_null(full, 31).items():
        np.testing.assert_array_equal(array, simulate_null(tied, 31)[kind])


@pytest.mark.parametrize("failure_call", [1, 2])
def test_failed_shared_point_or_draw_withholds_p_without_replacement(
    tmp_path, monkeypatch, failure_call
):
    from super_deli.workflows import integrative

    values, layout, _ = simulate_grouped("independent_gaussian", 64, 193)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    runner, calls = integrative.run_integrative, []

    def injected(*args, **kwargs):
        calls.append(None)
        if len(calls) == failure_call:
            raise ValueError("injected shared grouped failure")
        return runner(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", injected)
    result = run_grouped_calibration(
        dataset, config=policy(), base_dir=tmp_path / "raw", output_dir=tmp_path / "out"
    )
    assert result["inference"]["p_value"] is None
    assert result["inference"]["reject_grouped_raw_gaussian"] is None
    assert result["failed_fits"] == 1 and len(calls) == (1 if failure_call == 1 else 3)
    assert result["unattempted_fits"] == (2 if failure_call == 1 else 0)


@pytest.mark.parametrize("failure", ["optimizer", "archive"])
def test_unresolved_null_or_archive_retains_attempt_counts(tmp_path, failure):
    values, layout, _ = simulate_grouped("complementary", 64, 193)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    c = policy()
    if failure == "optimizer":
        c = replace(c, max_iterations=1, solver=replace(c.solver, max_polish_iterations=1))
    else:
        c = replace(c, max_archive_bytes=1)
    with pytest.raises(ValueError):
        run_grouped_calibration(
            dataset, config=c, base_dir=tmp_path / "raw", output_dir=tmp_path / "out"
        )
    record = read_json(tmp_path / "out/failure.json")
    assert record["attempted_fits"] == (0 if failure == "optimizer" else 1)
    assert record["unattempted_fits"] == (3 if failure == "optimizer" else 2)
    assert not (tmp_path / "out/evidence.json").exists()


def test_seven_case_shared_study_audit_cannot_refit_or_accept_tampering(tmp_path, monkeypatch):
    from super_deli.models import grouped_conditional, grouped_gaussian, mixture
    from super_deli.workflows import integrative

    study = GroupedStudyConfig.from_dict(
        read_json(ROOT / "examples/shared-grouped-calibration/smoke.json")
    )
    study = replace(study, calibration=replace(study.calibration, n_simulations=1))
    root = tmp_path / "study"
    root.mkdir()
    for case in study_cases(study):
        run_grouped_study(study, case=case, output_dir=root / case)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}

    def refuse(*args, **kwargs):
        raise AssertionError("Audit must not fit")

    monkeypatch.setattr(grouped_conditional, "minimize", refuse)
    monkeypatch.setattr(grouped_gaussian, "minimize", refuse)
    monkeypatch.setattr(integrative, "run_integrative", refuse)
    monkeypatch.setattr(mixture, "select_gaussian_mixture", refuse)
    result = audit_grouped_study(root, output_dir=tmp_path / "audit")
    assert result["datasets"] == 7 and result["successful_fits"] == 14
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    raw = root / "independent_gaussian/dataset_0000/observed/view_000.csv"
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="raw input"):
        audit_grouped_study(root, output_dir=tmp_path / "corrupt")
