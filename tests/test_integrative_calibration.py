"""Gaussian simulations, selection-inclusive statistics, failure accounting and run safety."""

# ruff: noqa: E402

import json
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli import load_config
from super_deli.cli import main
from super_deli.io.config import read_json
from super_deli.models import gaussian_calibration as gc
from super_deli.models.calibration_config import GaussianCalibrationConfig
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.workflows.integrative import run_integrative
from super_deli.workflows.score_calibration import run_integrative_calibration

pytestmark = pytest.mark.baseline
EXAMPLES = Path(__file__).resolve().parents[1] / "examples/integrative"


def search(**changes):
    return TabularClusterConfig(True, n_bootstrap=0, n_init=2, **changes)


def test_plus_one_ties_intervals_and_failure_denominators():
    records = [{"status": "complete", "statistic": 0.0} for _ in range(19)]
    assert gc.monte_carlo_summary(0, records, alpha=0.05)["p_value"] == 1
    result = gc.monte_carlo_summary(1, records, alpha=0.05)
    assert result["p_value"] == 0.05 and result["reject_single_gaussian"]
    assert result["conditional_null_tail_interval_95"][1] > 0
    records[-1] = {"status": "failed", "error": "retained failure"}
    result = gc.monte_carlo_summary(1, records, alpha=0.05)
    assert result["p_value"] is None and result["reject_single_gaussian"] is None
    assert result["requested"] == 19 and result["failed"] == 1
    assert result["failed_outcome_bounds"] == [0.05, 0.1]


def test_selected_k_search_in_statistic():
    candidates = [
        {"k": 1, "status": "eligible", "bic": 100},
        {"k": 2, "status": "small_component", "bic": 0},
        {"k": 3, "status": "eligible", "bic": 90},
    ]
    assert gc.bic_improvement(candidates) == 10


def test_reconstruct_draws_and_seeds_without_refitting():
    scores = np.random.default_rng(21).normal(size=(40, 2))
    config = GaussianCalibrationConfig(True, n_simulations=4, random_seed=71)
    fit = gc.calibrate_gaussian_scores(scores, search=search(), config=config)
    seeds = np.random.SeedSequence(71).spawn(5)
    np.testing.assert_allclose(fit.covariance, np.cov(scores, rowvar=False, bias=True))
    assert fit.summary["observed_statistic"] >= 0
    for seed, record in zip(seeds[1:], fit.records, strict=True):
        draw_seed, fit_seed = map(int, seed.generate_state(2))
        draw = fit.mean + np.random.default_rng(draw_seed).normal(size=scores.shape) @ (
            np.linalg.cholesky(fit.covariance).T
        )
        assert record["scores_sha256"] == gc.score_hash(draw)
        assert record["fit_seed"] == fit_seed
        assert len(record["candidates"]) == 4


def test_inner_failure_is_retained_without_retries(monkeypatch):
    original, calls = gc.select_gaussian_mixture, []

    def fail_once(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise ValueError("injected null fit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(gc, "select_gaussian_mixture", fail_once)
    fit = gc.calibrate_gaussian_scores(
        np.random.default_rng(2).normal(size=(40, 2)),
        search=search(),
        config=GaussianCalibrationConfig(True, n_simulations=3),
    )
    assert len(calls) == 4
    assert fit.summary["inference"]["failed"] == 1
    assert fit.summary["inference"]["p_value"] is None
    assert "injected" in fit.records[0]["error"]


@pytest.mark.parametrize(
    "scores,options,policy",
    [
        (np.ones((20, 2)), {}, {}),
        (np.zeros((20, 3)), {}, {}),
        (np.full((20, 2), np.nan), {}, {}),
        (np.ones((20, 2)), {"max_k": 1}, {}),
        (np.ones((20, 2)), {}, {"max_working_bytes": 1}),
    ],
)
def test_unsupported_scores_fail(scores, options, policy):
    with pytest.raises(ValueError):
        gc.calibrate_gaussian_scores(
            scores,
            search=search(**options),
            config=GaussianCalibrationConfig(True, n_simulations=1, **policy),
        )


@pytest.fixture
def reference(tmp_path):
    options = IntegrativeConfig.from_dict(read_json(EXAMPLES / "matched_model.json"))
    options = replace(
        options,
        views=tuple(v for v in options.views if v.kind != "genomic"),
        n_bootstrap=0,
        n_init=2,
    )
    output = tmp_path / "reference"
    run_integrative(
        load_config(EXAMPLES / "matched_dataset.json"),
        model_config=options,
        base_dir=EXAMPLES,
        output_dir=output,
    )
    return output


def test_integrative_cli_and_preservation(reference, tmp_path, capsys):
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in reference.rglob("*") if p.is_file()
    }
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(GaussianCalibrationConfig(True, n_simulations=2).to_dict()))
    output = tmp_path / "calibrated"
    assert (
        main(
            [
                "calibrate-integrative",
                str(reference),
                "--calibration-config",
                str(policy),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["inference"]["p_value"] == 1 / 3
    assert result["observed_selected_k"] == 4
    evidence = read_json(output / "evidence.json")
    assert evidence["target"] == "single_gaussian_frozen_scores"
    assert evidence["multiplicity"].startswith("none")
    assert evidence["null_hypothesis"] == result["null_hypothesis"]
    assert "Not a species-boundary" in result["scope"]
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in reference.rglob("*") if p.is_file()
    }
    with pytest.raises(FileExistsError):
        run_integrative_calibration(
            reference, config=GaussianCalibrationConfig(True), output_dir=output
        )
    with pytest.raises(ValueError, match="outside"):
        run_integrative_calibration(
            reference, config=GaussianCalibrationConfig(True), output_dir=reference / "new"
        )


def test_corrupt_source_refused_before_output(reference, tmp_path):
    with (reference / "mixture.npz").open("ab") as handle:
        handle.write(b"corruption")
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="checksum"):
        run_integrative_calibration(
            reference, config=GaussianCalibrationConfig(True), output_dir=output
        )
    assert not output.exists()


def test_saved_point_mismatch_stops_before_null_draws(monkeypatch):
    original, calls = gc.select_gaussian_mixture, []

    def record(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(gc, "select_gaussian_mixture", record)
    with pytest.raises(ValueError, match="reproduce"):
        gc.calibrate_gaussian_scores(
            np.random.default_rng(51).normal(size=(40, 2)),
            search=search(),
            config=GaussianCalibrationConfig(True, n_simulations=3),
            expected_point=(999, 0),
        )
    assert len(calls) == 1
