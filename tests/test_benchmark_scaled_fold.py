"""Paired full-pipeline fits and independent audits of effective fold minima."""
# ruff: noqa: E402

from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.local_archives import materialize
from super_deli.benchmarking.scaled_fold_audit import audit_observation
from super_deli.benchmarking.scaled_fold_study import observation
from super_deli.io.config import read_json
from super_deli.models.mixture import select_gaussian_mixture
from super_deli.models.predictive_fold_size import PredictiveFoldMixture
from super_deli.models.scaled_fold_study import ScaledFoldStudy

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = ScaledFoldStudy.from_dict(read_json(ROOT / "examples/scaled-fold-stability/smoke.json"))
    return replace(c, cells=(c.cells[3],), n_bootstrap=1, min_free_bytes=1024**2)


def test_newly_feasible_candidates_are_actually_fitted():
    c = config()
    rng = np.random.default_rng(202612290001)
    means = np.asarray(
        [[0, 0, 0, 0], [100, 0, 0, 0], [0, 100, 0, 0], [100, 100, 0, 0]], dtype=float
    )
    distinct = np.repeat(means, 2, axis=0) + rng.normal(0, 0.02, (8, 4))
    scores = np.repeat(distinct, 2, axis=0)
    ids = np.repeat(np.arange(8), 2)
    old = select_gaussian_mixture(scores, c.model, 2841, covariance_type="tied", unit_indices=ids)
    assert old.candidates[-1]["status"] == "insufficient_samples"
    new = select_gaussian_mixture(
        scores,
        PredictiveFoldMixture.from_model(c.model, 2),
        2841,
        covariance_type="tied",
        unit_indices=ids,
    )
    assert new.candidates[-1]["status"] != "insufficient_samples"
    assert "k4_membership" in new.candidate_arrays
    assert sum(new.candidates[-1]["distinct_unit_sizes"]) == 8


def test_paired_pipeline_preserves_full_minimum_and_legacy_contract(tmp_path):
    c = config()
    root = tmp_path / "study"
    root.mkdir()
    d = c.plan()[0]
    row = observation((str(root), c.to_dict(), d))
    assert row["successful_fits"] == 24 and row["unattempted_fits"] == 0
    audit = audit_observation((str(root), c.to_dict(), d, row, str(tmp_path / "scratch")))
    assert audit["audited_searches"] == 24
    with materialize(root / row["archive"], row, scratch=tmp_path / "scratch") as base:
        for arm in ("legacy", "scaled"):
            result = read_json(base / arm / "summary.json")
            assert ("fold_size_policy" in result) == (arm == "scaled")
            assert read_json(base / arm / "point/full/model.json")["min_cluster_size"] == 5
            for f in range(5):
                path = base / arm / "point" / f"fold_{f:03d}"
                assert (path / "fold_support.json").exists() == (arm == "scaled")
                if arm == "scaled":
                    e = read_json(path / "fold_support.json")
                    assert e["training_minimum"] in (4, 5)
                    assert read_json(path / "model.json")["min_cluster_size"] == 5
                    assert (
                        read_json(path / "run/mixture_settings.json")["min_cluster_size"]
                        == e["training_minimum"]
                    )
        a = read_json(base / "legacy/summary.json")["replicates"][0]
        b = read_json(base / "scaled/summary.json")["replicates"][0]
        for key in ("training_indices", "multiplicities", "oob_indices", "draw_seed"):
            assert a[key] == b[key]
        assert (base / "common_stability.npz").exists()


def test_failed_draw_is_retained_without_replacement_or_k1(tmp_path, monkeypatch):
    import super_deli.workflows.predictive_resampling as module

    c = replace(config(), n_bootstrap=2)
    actual = module.occurrence_screen
    failures = []

    def fail_one(*args, **kwargs):
        if kwargs.get("fold_size_policy") is not None and not failures:
            failures.append(True)
            raise ValueError("Prespecified test failure; no replacement")
        return actual(*args, **kwargs)

    monkeypatch.setattr(module, "occurrence_screen", fail_one)
    root = tmp_path / "study"
    root.mkdir()
    d = c.plan()[0]
    row = observation((str(root), c.to_dict(), d))
    assert row["failed_draws"] == 1 and row["unattempted_fits"] == 6
    assert row["arms"]["scaled"]["successful_draws"] == 1
    audit = audit_observation((str(root), c.to_dict(), d, row, str(tmp_path / "scratch")))
    assert audit["failed_draws"] == 1 and audit["audited_searches"] == 30
    import importlib.util
    import json

    from super_deli.io.config import write_json

    write_json(c.to_dict(), root / "configuration.json")
    write_json(c.plan(), root / "plan.json")
    (root / "observations.jsonl").write_text(json.dumps(row) + "\n")
    write_json(dict(attempted_fits=row["attempted_fits"]), root / "summary.json")
    spec = importlib.util.spec_from_file_location(
        "scaled_byte_audit", ROOT / "examples/scaled-fold-stability/check_study.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert checker.verify(root)["audited_searches"] == 30
    with materialize(root / row["archive"], row, scratch=tmp_path / "scratch") as base:
        summary = read_json(base / "scaled/summary.json")
        failed = summary["replicates"][0]
        assert failed["status"] == "failed" and "selected_k" not in failed
        assert len(summary["replicates"]) == 2
        with np.load(base / "scaled/stability.npz", allow_pickle=False) as p:
            assert np.all(p["all_requested"] == 2) and np.all(p["all_successful"] == 1)
            assert np.isnan(p["all_frequency"]).all()


def test_effective_threshold_is_recorded_before_fitting(tmp_path, monkeypatch):
    import super_deli.models.integrative as engine
    from super_deli.benchmarking.predictive_study import simulate
    from super_deli.workflows.predictive import fold_model
    from super_deli.workflows.predictive_resampling import occurrence_search

    c = config()
    cell = c.cells[0]
    values, layout, _ = simulate(cell, cell.n_units, 202612290002)
    indices = np.arange(51, dtype=int)
    support = c.fold_size_policy.evidence(5, range(64), indices.tolist())
    target = tmp_path / "fold_000"

    def fail(*args, **kwargs):
        assert kwargs["fold_minimum"] == 4
        assert read_json(target / "fold_support.json")["training_minimum"] == 4
        assert read_json(target / "run/mixture_settings.json")["min_cluster_size"] == 4
        raise ValueError("Test failure after frozen evidence, before estimator")

    monkeypatch.setattr(engine, "_fit_once", fail)
    outcome = occurrence_search(
        tmp_path,
        "fold_000",
        values,
        layout,
        indices,
        fold_model(c.model, 0),
        fold_minimum=4,
        fold_support=support,
    )
    assert outcome["status"] == "failed" and "before estimator" in outcome["error"]
