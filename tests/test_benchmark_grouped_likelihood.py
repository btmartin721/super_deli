"""AIC/likelihood isolation, all-candidate densities, failure and legacy checks."""

import hashlib
import importlib.util
import io
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from super_deli.io.config import load_config, read_json
from super_deli.models.grouped_config import GroupedCalibrationConfig, GroupedStudyConfig
from super_deli.models.grouped_likelihood_config import (
    CASES,
    GroupedLikelihoodStudyConfig,
)

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="Install clustering extra"
)


def study():
    return GroupedLikelihoodStudyConfig.from_dict(
        read_json(ROOT / "examples/grouped-likelihood/smoke.json")
    )


def test_legacy_serialization_and_scope_guards():
    prior = read_json(ROOT / "examples/full-grouped-diagnostics/results/source_study.json")
    assert json.loads(json.dumps(GroupedStudyConfig.from_dict(prior).to_dict())) == prior
    s = study()
    assert s.to_dict()["calibration"]["model"]["selection_criterion"] == "aic"
    with pytest.raises(ValueError):
        GroupedCalibrationConfig.from_dict(s.calibration.to_dict())
    legacy = {k: v for k, v in s.calibration.to_dict().items() if k != "statistic"}
    with pytest.raises(ValueError, match="separate grouped likelihood"):
        GroupedCalibrationConfig.from_dict(legacy)
    m = s.calibration.model
    for changes in (
        {"selection_criterion": []},
        {"selection_criterion": "aicc"},
        {"target_level": "specimen"},
    ):
        with pytest.raises(ValueError):
            replace(m, **changes)
    with pytest.raises(ValueError, match="full-covariance"):
        replace(s.calibration, model=replace(m, covariance_type="tied"))
    with pytest.raises(ValueError):
        replace(s.calibration, model=replace(m, selection_criterion="bic"))
    with pytest.raises(ValueError):
        replace(s.calibration, statistic="unknown")
    with pytest.raises(ValueError):
        replace(s, large_n=s.n_units)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    from super_deli.benchmarking.grouped_likelihood import run_grouped_likelihood_study

    root = tmp_path_factory.mktemp("likelihood")
    for case in CASES:
        run_grouped_likelihood_study(study(), case=case, output_dir=root / case)
    return root


@SCIENTIFIC
def test_every_candidate_and_read_only_auditor(completed, tmp_path, monkeypatch):
    from sklearn.mixture import GaussianMixture

    from super_deli.benchmarking.grouped_likelihood_analysis import audit_grouped_likelihood
    from super_deli.benchmarking.recovery_inputs import inventory

    before = inventory(completed)

    def no_fit(*args, **kwargs):
        pytest.fail("Read-only audit attempted a model fit")

    monkeypatch.setattr(GaussianMixture, "fit", no_fit)
    result = audit_grouped_likelihood(completed, output_dir=tmp_path / "audit")
    assert result["successful_fits"] == result["planned_fits"] == 20
    assert result["datasets"] == 10 and inventory(completed) == before
    paired = read_json(tmp_path / "audit/paired.json")
    assert len(paired["summaries"]) == 20 and len(paired["paired_changes"]) == 10


@SCIENTIFIC
def test_legacy_candidate_fits_exactly_preserved(completed, tmp_path):
    import numpy as np

    from super_deli.workflows.integrative import run_integrative

    base = completed / "complementary/dataset_0000"
    m = study().calibration.model
    result = run_integrative(
        load_config(base / "observed/dataset.json"),
        model_config=replace(m, selection_criterion="bic"),
        base_dir=base / "observed",
        output_dir=tmp_path / "legacy",
    )
    original = read_json(base / "calibration/observed.json")
    fields = {"aic", "log_likelihood"}
    candidates = [{k: v for k, v in c.items() if k not in fields} for c in original["candidates"]]
    assert candidates == read_json(tmp_path / "legacy/candidates.json")
    assert result["selected_k"] == original["bic_selected_k"]
    assert not (tmp_path / "legacy/gmm_candidates.npz").exists()
    with ZipFile(base / "calibration/pipeline_artifacts.zip") as archive:
        with (
            np.load(
                io.BytesIO(archive.read("observed/run/gmm_candidates.npz")), allow_pickle=False
            ) as pool,
            np.load(tmp_path / "legacy/mixture.npz", allow_pickle=False) as old,
        ):
            np.testing.assert_array_equal(pool["scores"], old["scores"])
    assert "selection_criterion" not in result


@SCIENTIFIC
def test_independent_nonselected_density_rejects_tampering(completed):
    import numpy as np

    from super_deli.benchmarking.config import derived_seed
    from super_deli.benchmarking.grouped_likelihood import case_policy
    from super_deli.benchmarking.grouped_likelihood_analysis import audit_candidates
    from super_deli.benchmarking.grouped_study import simulate_grouped

    s = study()
    seed = derived_seed(s.random_seed, "complementary", 0, "observed_data")
    values, layout, _ = simulate_grouped("complementary", s.n_units, seed)
    base = completed / "complementary/dataset_0000/calibration"
    record = read_json(base / "observed.json")
    with ZipFile(base / "pipeline_artifacts.zip") as archive:
        audit_candidates(
            archive,
            record,
            case_policy(s, "complementary", 0),
            values,
            json.loads(json.dumps(layout)),
        )
        data = {}
        with np.load(
            io.BytesIO(archive.read("observed/run/gmm_candidates.npz")), allow_pickle=False
        ) as arrays:
            pool = {k: arrays[k].copy() for k in arrays.files}
        k = next(c["k"] for c in record["candidates"] if c["k"] != record["selected_k"])
        pool[f"k{k}_covariances"] *= 1.2
        handle = io.BytesIO()
        np.savez(handle, **pool)
        data["observed/run/gmm_candidates.npz"] = handle.getvalue()
        digest = hashlib.sha256(handle.getvalue()).hexdigest()
        summary = json.loads(archive.read("observed/run/summary.json"))
        summary["provenance"]["candidate_sha256"] = digest
        data["observed/run/summary.json"] = json.dumps(summary).encode()

        class Altered:
            def read(self, name):
                return data[name] if name in data else archive.read(name)

        with pytest.raises(AssertionError):
            audit_candidates(
                Altered(),
                {**record, "candidate_sha256": digest},
                case_policy(s, "complementary", 0),
                values,
                json.loads(json.dumps(layout)),
            )


@SCIENTIFIC
@pytest.mark.parametrize("failed_call", [1, 2])
def test_failed_point_or_draw_withholds_p_without_replacements(
    completed, tmp_path, monkeypatch, failed_call
):
    from super_deli.workflows import grouped_likelihood as workflow

    base = completed / "independent_gaussian/dataset_0000"
    dependencies = workflow._dependencies()
    original = dependencies[-1]
    calls = []

    def fail(dataset, **kwargs):
        calls.append(kwargs["output_dir"])
        if len(calls) == failed_call:
            raise ValueError("Injected numerical failure")
        return original(dataset, **kwargs)

    monkeypatch.setattr(workflow, "_dependencies", lambda: (*dependencies[:-1], fail))
    config = replace(study().calibration, n_simulations=2)
    result = workflow.run_grouped_likelihood(
        load_config(base / "observed/dataset.json"),
        config=config,
        base_dir=base / "observed",
        output_dir=tmp_path / "failed",
    )
    assert result["inference"]["p_value"] is None
    assert result["inference"]["reject_grouped_raw_gaussian"] is None
    assert len(calls) == (1 if failed_call == 1 else 3)
    assert result["failed_fits"] == 1
    assert result["unattempted_fits"] == (2 if failed_call == 1 else 0)


@SCIENTIFIC
def test_aic_whole_group_bootstrap_uses_same_selection_rule(completed, tmp_path):
    from super_deli.workflows.integrative import run_integrative

    base = completed / "complementary/dataset_0000"
    run_integrative(
        load_config(base / "observed/dataset.json"),
        model_config=replace(study().calibration.model, n_bootstrap=3),
        base_dir=base / "observed",
        output_dir=tmp_path / "bootstrap",
    )
    rows = read_json(tmp_path / "bootstrap/bootstrap_replicates.json")
    assert len(rows) == 3
    for r in rows:
        if r["status"] == "complete":
            eligible = [c for c in r["candidates"] if c["status"] == "eligible"]
            assert r["selected_k"] == min(eligible, key=lambda c: (c["aic"], c["k"]))["k"]
            assert all(min(c["distinct_unit_sizes"]) >= 5 for c in eligible)


@SCIENTIFIC
def test_separate_statistic_tail_audit_detects_tampering(completed, tmp_path, monkeypatch):
    from super_deli.benchmarking.full_grouped_diagnostic_analysis import csv_rows
    from super_deli.benchmarking.grouped_likelihood_analysis import paired_rows

    spec = importlib.util.spec_from_file_location(
        "independent_tails", ROOT / "examples/grouped-likelihood/check_tails.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = tmp_path / "paired.csv"
    csv_rows(paired_rows(completed, study()), table)
    result = module.check(completed, table)
    assert result["verified_successful_pipeline_statistics"] == 20
    original = module.read

    def altered(path):
        data = original(path)
        if path.name == "null_replicates.json":
            data[0]["statistic"] += 1
        return data

    monkeypatch.setattr(module, "read", altered)
    with pytest.raises(ValueError, match="statistic mismatch"):
        module.check(completed, table)
