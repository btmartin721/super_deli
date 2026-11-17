"""Independent density, omission weighting, equality and retained-failure checks."""

# ruff: noqa: E402
import csv
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from super_deli.models.heldout_predictive import logdensity, score_draws, summarize_scores
from super_deli.models.heldout_support_config import HeldoutSupportConfig

pytestmark = pytest.mark.baseline


def test_density_against_independent_scipy():
    x = np.random.default_rng(20801).normal(size=(15, 3))
    p = dict(
        weights=np.array([0.3, 0.7]),
        means=np.array([[0.0, 0, 0], [1, 2, 3]]),
        covariances=np.array([[2.0, 0.1, 0], [0.1, 1, 0], [0, 0, 0.5]]),
    )
    actual, prob, labels = logdensity(x, p)
    terms = np.stack(
        [
            np.log(w) + multivariate_normal.logpdf(x, mean=m, cov=p["covariances"])
            for w, m in zip(p["weights"], p["means"], strict=True)
        ],
        axis=1,
    )
    np.testing.assert_allclose(actual, logsumexp(terms, axis=1), rtol=1e-12)
    np.testing.assert_allclose(prob, np.exp(terms - logsumexp(terms, axis=1)[:, None]), rtol=1e-12)
    np.testing.assert_array_equal(labels, terms.argmax(1))
    with pytest.raises(ValueError):
        logdensity(x, {**p, "covariances": -np.eye(3)})


def test_individual_first_omission_weighting_and_no_finite_failure_bounds(tmp_path):
    source = SimpleNamespace(
        point=SimpleNamespace(
            units=("a", "b", "c"),
            labels=np.zeros(3, dtype=int),
            model=SimpleNamespace(min_cluster_size=1, max_k=1),
            summary={"selected_k": 1},
        ),
        records=[
            dict(
                training_indices=[1, 2],
                status="complete",
                selected_k=1,
                candidates=[dict(k=1, status="eligible")],
            ),
            dict(
                training_indices=[2],
                status="complete",
                selected_k=1,
                candidates=[dict(k=1, status="eligible")],
            ),
            dict(training_indices=[0, 2], status="failed"),
        ],
    )
    records = [
        dict(
            status="complete", omitted_indices=[0], arrays="a.npz", baseline_search_attempted=True
        ),
        dict(
            status="complete",
            omitted_indices=[0, 1],
            arrays="b.npz",
            baseline_search_attempted=True,
        ),
        dict(status="unavailable_native", omitted_indices=[1], baseline_search_attempted=False),
    ]
    np.savez(tmp_path / "a.npz", gain=np.array([2.0, 999.0, 999.0]))
    np.savez(tmp_path / "b.npz", gain=np.array([4.0, -2.0, 999.0]))
    s, a, g = summarize_scores(source, records, tmp_path, HeldoutSupportConfig())
    assert s["covered_individual_mean"] == 0.5  # mean of individual means 3 and -2
    assert s["coverage"] == 2 / 3 and s["requested_comparisons"] == 4
    assert s["unavailable_requested_comparisons"] == 1 and s["direction"] == "unavailable"
    assert s["full_cohort_mean"] is None and "no finite" in s["missing_gain_bounds"]
    np.testing.assert_array_equal(a["successful"], [2, 1, 0])
    assert np.isnan(a["means"][2]) and g[0]["covered_individuals"] == 2


def test_one_group_equality_keeps_missing_coverage_explicit(tmp_path):
    from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
    from super_deli.models.mixture import select_gaussian_mixture

    m = IntegrativeConfig(
        True,
        views=(ViewConfig("m", "morphological"), ViewConfig("e", "ecological")),
        n_pcs=4,
        min_cluster_size=5,
        n_init=100,
        max_k=1,
        covariance_type="tied",
        fusion_policy="retain_all_v1",
    )
    x = np.random.default_rng(20803).normal(size=(64, 4))
    draw = np.arange(51)
    fit = select_gaussian_mixture(x[draw], m, 42, covariance_type="tied", unit_indices=draw)
    folder = tmp_path / "subsamples/draw_0000"
    folder.mkdir(parents=True)
    np.savez(
        folder / "fit.npz",
        training_scores=x[draw],
        projected_scores=x,
        weights=fit.mixture.weights_,
        means=fit.mixture.means_,
        covariances=fit.mixture.covariances_,
        training_membership=np.ones((51, 1)),
        projected_membership=np.ones((64, 1)),
        projected_labels=np.zeros(64, dtype=int),
    )
    r = dict(
        replicate=0,
        training_indices=draw.tolist(),
        omitted_indices=list(range(51, 64)),
        parameters="draw_0000",
        status="complete",
        selected_k=1,
        fit_seed=42,
        candidates=fit.candidates,
        projected_labels=[0] * 64,
    )
    source = SimpleNamespace(
        root=tmp_path,
        records=[r],
        point=SimpleNamespace(
            units=tuple(map(str, range(64))),
            model=m,
            labels=np.zeros(64, dtype=int),
            summary={"selected_k": 1},
        ),
        reverify=lambda: None,
    )
    rows = score_draws(source, tmp_path / "scores")
    assert rows[0]["status"] == "complete"
    with np.load(tmp_path / "scores" / rows[0]["arrays"], allow_pickle=False) as a:
        np.testing.assert_array_equal(a["gain"], 0)
    s, _, _ = summarize_scores(source, rows, tmp_path / "scores", HeldoutSupportConfig())
    assert s["covered_individual_mean"] == 0 and s["full_cohort_mean"] is None
    assert s["coverage"] == 13 / 64 and s["direction"] == "unavailable"


def test_native_source_preserved_and_failed_baselines_not_one_group(tmp_path):
    from super_deli.benchmarking.signal_loss_source import digest
    from super_deli.io.heldout_support_source import load_subsample_source
    from super_deli.models.full_retention import full_retention_model
    from super_deli.models.individual_subsample_config import IndividualSubsampleConfig
    from super_deli.workflows.heldout_support import run_heldout_support
    from super_deli.workflows.individual_subsampling import run_individual_subsampling
    from super_deli.workflows.integrative import run_integrative

    path = Path(__file__).resolve().parents[1] / "examples/multimodal/synthetic.py"
    spec = importlib.util.spec_from_file_location("heldout_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ds, analysis = module.invented(tmp_path / "inputs", seed=20805, n=64, law="shape_only", draws=0)
    ds, analysis = module.select_views(ds, analysis, {"traits", "habitat", "outline"})
    model = full_retention_model(
        replace(analysis.arms[-1].model, n_init=100, n_bootstrap=0), min_cluster_size=7
    )
    run_integrative(
        ds, model_config=model, base_dir=tmp_path / "inputs", output_dir=tmp_path / "point"
    )
    run_individual_subsampling(
        tmp_path / "point",
        stability_config=IndividualSubsampleConfig(2, 1601),
        output_dir=tmp_path / "subsample",
    )
    root = tmp_path / "subsample"
    before = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    result = run_heldout_support(
        root, support_config=HeldoutSupportConfig(), output_dir=tmp_path / "report"
    )
    assert result["baseline_searches"] == 2 and result["successful_draws"] == 2
    assert result["p_value"] is None and result["point_analysis_changed"] is False
    assert before == {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    assert len(list(csv.DictReader((tmp_path / "report/individual_scores.csv").open()))) == 64
    assert "same-sample point" in (tmp_path / "report/report.html").read_text()
    with patch(
        "super_deli.models.heldout_predictive.replay_baseline",
        side_effect=ValueError("invented baseline failure"),
    ):
        s = run_heldout_support(
            root, support_config=HeldoutSupportConfig(), output_dir=tmp_path / "failed-report"
        )
    assert s["successful_draws"] == 0 and s["covered_individual_mean"] is None
    assert s["direction"] == "unavailable" and s["unavailable_requested_comparisons"] == 26
    source = load_subsample_source(root)
    original = (root / "summary.json").read_bytes()
    (root / "summary.json").write_bytes(original + b" ")
    with pytest.raises(ValueError, match="changed"):
        source.reverify()
    (root / "summary.json").write_bytes(original)
    with pytest.raises(ValueError, match="outside"):
        run_heldout_support(root, support_config=HeldoutSupportConfig(), output_dir=root / "nested")
