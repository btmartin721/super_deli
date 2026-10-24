"""Dependency-free design, pooling, failure withholding and legacy boundaries."""

from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json
from super_deli.models.predictive_config import PredictiveScreenConfig
from super_deli.models.predictive_selection import screen_candidates
from super_deli.models.predictive_study_config import PredictiveCell, PredictiveStudyConfig

ROOT = Path(__file__).resolve().parents[1]


def test_storage_accounting_tolerates_atomic_archive_publication(tmp_path, monkeypatch):
    from super_deli.benchmarking.predictive_study import retained_bytes

    final = tmp_path / "dataset.tar.gz"
    final.write_bytes(b"verified evidence")
    partial = tmp_path / "dataset.tar.gz.partial"
    partial.write_bytes(b"verified evidence")
    original = Path.stat

    def concurrent_stat(path, *args, **kwargs):
        if path == partial:
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", concurrent_stat)
    assert retained_bytes(tmp_path) == len(b"verified evidence")


def configuration(name="study.json"):
    return PredictiveStudyConfig.from_dict(read_json(ROOT / "examples/predictive-aic" / name))


def candidate(k, aic=100, status="eligible"):
    return {"k": k, "aic": aic, "bic": 100 + k, "status": status}


def folds(gains=(4, -2)):
    return [
        {
            "fold": i,
            "status": "complete",
            "training_indices": ([1, 2, 3] if i == 0 else [0]),
            "validation_indices": ([0] if i == 0 else [1, 2, 3]),
            "candidates": [candidate(1), candidate(2, 90)],
            "log_densities": {
                "1": [0] * (1 if i == 0 else 3),
                "2": [gains[i]] * (1 if i == 0 else 3),
            },
        }
        for i in range(2)
    ]


def test_unit_weighting_not_equal_fold_weighting():
    result = screen_candidates([candidate(1), candidate(2, 90)], folds(), n_units=4, n_folds=2)
    assert result["ordinary_aic_k"] == 2 and result["screened_k"] == 1
    assert result["candidates"][1]["pooled_validation_gain"] == -0.5


@pytest.mark.parametrize("gain,chosen", [(0, 1), (-1, 1), (1e-12, 2)])
def test_strict_positive_boundary(gain, chosen):
    r = screen_candidates(
        [candidate(1), candidate(2, 90)], folds((gain, gain)), n_units=4, n_folds=2
    )
    assert r["screened_k"] == chosen
    assert r["status"] == "complete"


def test_original_k_one_is_never_promoted_by_screening():
    result = screen_candidates(
        [candidate(1), candidate(2, 110)], folds((1, 1)), n_units=4, n_folds=2
    )
    assert result["ordinary_aic_k"] == result["screened_k"] == 1


def test_ineligible_fold_blocks_candidate_and_ties_favor_smaller_k():
    f = folds((1, 1))
    f[0]["candidates"][1]["status"] = "small_component"
    r = screen_candidates([candidate(1), candidate(2, 90)], f, n_units=4, n_folds=2)
    assert r["screened_k"] == 1 and r["candidates"][1]["reason"] == "ineligible_full_or_fold"
    r = screen_candidates([candidate(1, 90), candidate(2, 90)], folds((1, 1)), n_units=4, n_folds=2)
    assert r["screened_k"] == 1


@pytest.mark.parametrize(
    "problem", ["failed", "missing_fold", "missing_density", "nonfinite", "baseline_ineligible"]
)
def test_failures_are_withheld_not_k_one(problem):
    f = folds((1, 1))
    if problem == "failed":
        f[0]["status"] = "failed"
    elif problem == "missing_fold":
        f = f[:1]
    elif problem == "missing_density":
        del f[0]["log_densities"]["2"]
    elif problem == "nonfinite":
        f[0]["log_densities"]["2"] = [float("inf")]
    else:
        f[0]["candidates"][0]["status"] = "small_component"
    r = screen_candidates([candidate(1), candidate(2, 90)], f, n_units=4, n_folds=2)
    assert r["status"] == "withheld" and r["screened_k"] is None and r["error"]


def test_duplicate_validation_and_overlapping_training_are_rejected():
    for duplicate in (True, False):
        f = folds()
        f[1]["validation_indices"] = [0, 2, 3] if duplicate else [1, 2, 3]
        if not duplicate:
            f[0]["training_indices"] = [0, 1, 2]
        with pytest.raises(ValueError):
            screen_candidates([candidate(1), candidate(2, 90)], f, n_units=4, n_folds=2)


def test_frozen_main_budget_and_probe_settings():
    c = configuration()
    assert PredictiveStudyConfig.from_dict(c.to_dict()) == c
    assert len(c.plan()) == 625 and sum(r["planned_fits"] for r in c.plan()) + 25 == 3775
    assert c.probe() == configuration("preflight.json")
    assert len(c.probe().plan()) == 17 and sum(r["planned_fits"] for r in c.probe().plan()) == 102
    assert c.screening.n_folds == 5 and c.n_test == 2048 and c.model.n_init == 10
    assert c.model.random_seed == 71323 and c.max_artifact_bytes == 2 * 1024**3
    assert sum(x.role == "control" for x in c.cells) == 8
    assert sum(x.role == "alternative" for x in c.cells) == 6
    seeds = [r[k] for r in c.plan() for k in ("data_seed", "test_seed", "split_seed")]
    assert len(seeds) == len(set(seeds)) == 1875
    assert set(seeds).isdisjoint(
        r[k] for r in c.probe().plan() for k in ("data_seed", "test_seed", "split_seed")
    )
    assert sorted(c.plan(), key=str) == sorted(
        replace(c, cells=tuple(reversed(c.cells))).plan(), key=str
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"n_folds": True},
        {"n_folds": 1},
        {"n_folds": 11},
        {"random_seed": -1},
        {"random_seed": True},
        {"schema_version": True},
    ],
)
def test_invalid_screen_configs(updates):
    with pytest.raises(ValueError):
        PredictiveScreenConfig(**updates)


def test_model_bootstrap_and_unknown_config_are_rejected():
    c = configuration()
    with pytest.raises(ValueError):
        PredictiveScreenConfig.from_dict({"schema_version": 1, "unknown": 1})
    for update in ({"n_bootstrap": 1}, {"selection_criterion": "bic"}, {"n_threads": 2}):
        with pytest.raises(ValueError):
            c.screening.validate_model(replace(c.model, **update))
    with pytest.raises(ValueError):
        PredictiveCell("bad", "independent_gaussian", "alternative", 64, 50)


def test_full_study_requires_measured_matching_preflight(tmp_path):
    from super_deli.benchmarking.predictive_study import run_study

    with pytest.raises(ValueError, match="Measured storage"):
        run_study(configuration(), output_dir=tmp_path / "unlaunched")
    assert not (tmp_path / "unlaunched").exists()


@pytest.mark.parametrize("condition", ["pass", "space", "cap", "failed_probe"])
def test_measured_gate_preserves_failed_outcome(tmp_path, monkeypatch, condition):
    import json
    from types import SimpleNamespace

    from super_deli.benchmarking import predictive_preflight as gate
    from super_deli.io.config import write_json

    c = configuration()
    if condition == "cap":
        c = replace(c, max_artifact_bytes=128 * 1024**2)
    probe = c.probe()
    study, audit = tmp_path / "probe", tmp_path / "audit"
    study.mkdir()
    audit.mkdir()
    write_json(probe.to_dict(), study / "configuration.json")
    write_json(
        {
            "successful_fits": 101 if condition == "failed_probe" else 102,
            "unattempted_fits": 0,
            "screening_withheld": 0,
        },
        audit / "summary.json",
    )
    (study / "observations.jsonl").write_text(
        "\n".join(json.dumps({**r, "size_bytes": 10000}) for r in probe.plan())
    )
    monkeypatch.setattr(
        gate.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=(7 if condition == "space" else 20) * 1024**3),
    )
    output = tmp_path / "gate.json"
    arguments = dict(
        probe_study=study, probe_audit=audit, artifact_root=tmp_path, output_file=output
    )
    if condition == "pass":
        result = gate.storage_preflight(c, **arguments)
        assert result["status"] == "passed" and result["main_searches"] == 3775
    elif condition == "failed_probe":
        with pytest.raises(ValueError, match="successful independently audited"):
            gate.storage_preflight(c, **arguments)
        assert not output.exists()
    else:
        with pytest.raises(RuntimeError, match="Storage gate failed"):
            gate.storage_preflight(c, **arguments)
        assert read_json(output)["status"] == "failed"
