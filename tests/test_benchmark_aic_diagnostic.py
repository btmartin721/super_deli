"""Numerical mechanism, independent prediction, corruption and failure checks."""

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from super_deli.benchmarking.aic_diagnostic_audit import (  # noqa: E402
    audit_observation,
    check_metrics,
    check_pool,
    independent_projection,
    independent_raw,
)
from super_deli.benchmarking.aic_diagnostic_metrics import candidate_metrics  # noqa: E402
from super_deli.benchmarking.aic_diagnostic_simulation import (  # noqa: E402
    decompose,
    heldout_layout,
    homogeneous,
    project_saved,
)
from super_deli.benchmarking.aic_diagnostic_study import _observation  # noqa: E402
from super_deli.benchmarking.local_archives import materialize  # noqa: E402
from super_deli.io.config import read_json  # noqa: E402
from super_deli.models.aic_diagnostic_config import AICDiagnosticConfig  # noqa: E402
from super_deli.models.mixture import select_gaussian_mixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    c = AICDiagnosticConfig.from_dict(
        read_json(ROOT / "examples/aic-gaussian-diagnostics/smoke.json")
    )
    return replace(
        c, cases=("zero_conditional_gaussian",), sizes=(20,), n_test=20, min_free_bytes=1024**2
    )


@pytest.mark.parametrize(
    "case", ["independent_gaussian", "correlated_gaussian", "zero_conditional_gaussian"]
)
def test_independent_law_and_noise_equalization(case):
    values, layout, mu, err = decompose(case, 64, 197)
    other, identity, mean, residual = independent_raw(case, 64, 197)
    assert layout == identity
    np.testing.assert_allclose(mu, mean, atol=1e-14)
    np.testing.assert_allclose(err, residual, atol=1e-14)
    for key in values:
        np.testing.assert_allclose(values[key], other[key], atol=1e-14)
    count = np.asarray(layout["counts"])
    index = np.repeat(np.arange(64), count)
    r = np.mean(1 / count)
    changed = homogeneous(values, layout, mu, err, r)
    np.testing.assert_allclose(
        changed["morphological"] - mu[index], err * np.sqrt(count[index, None] * r), atol=1e-14
    )
    np.testing.assert_array_equal(changed["ecological"], values["ecological"])
    # Analytic group-mean covariance is W*r in every count stratum.
    np.testing.assert_allclose(count * r / count, r)
    held = heldout_layout(layout)
    assert set(held["unit_ids"]).isdisjoint(layout["unit_ids"])
    assert held["counts"] == layout["counts"]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("aic-diagnostic")
    c = configuration()
    declaration = c.plan()[0]
    record = _observation((str(root), c.to_dict(), declaration))
    assert record["successful_fits"] == 3
    return root, c, declaration, record


def test_independent_complete_pipeline_audit_and_frozen_projection(fitted, tmp_path):
    root, c, declaration, record = fitted
    rows = audit_observation((str(root), declaration, record, c.to_dict(), str(tmp_path)))
    assert len(rows) == 3 and all(r["status"] == "complete" for r in rows)
    with materialize(root / record["archive"], record, scratch=tmp_path) as base:
        with np.load(base / "heldout_input.npz", allow_pickle=False) as p:
            values = {k: p[k] for k in ("morphological", "ecological")}
        layout = read_json(base / "heldout_layout.json")
        run = base / "baseline/run"
        x = project_saved(run, values, layout)
        np.testing.assert_allclose(x, independent_projection(run, values, layout), atol=1e-12)
        # Altering other test units cannot change one unit's frozen projection.
        changed = {k: v.copy() for k, v in values.items()}
        changed["morphological"][layout["counts"][0] :] += 500
        changed["ecological"][1:] -= 500
        np.testing.assert_allclose(project_saved(run, changed, layout)[0], x[0], atol=1e-12)


def test_density_criteria_count_and_covariance_tampering_rejected(fitted, tmp_path):
    root, c, _, record = fitted
    with materialize(root / record["archive"], record, scratch=tmp_path) as base:
        run = base / "baseline/run"
        raw = (run / "gmm_candidates.npz").read_bytes()
        candidates = read_json(run / "candidates.json")
        counts = np.asarray(read_json(base / "layout.json")["counts"])
        check_pool(raw, candidates, counts=counts, config=c.model)
        altered = json.loads(json.dumps(candidates))
        altered[0]["aic"] += 1
        with pytest.raises(AssertionError):
            check_pool(raw, altered, counts=counts, config=c.model)
        metrics = read_json(base / "baseline/metrics.json")
        check_metrics(raw, metrics, counts, c.model.reg_covar)
        metrics["candidates"][0]["count_by_component"][0][0] += 1
        with pytest.raises(AssertionError):
            check_metrics(raw, metrics, counts, c.model.reg_covar)


def test_failure_is_retained_without_replacement(tmp_path, monkeypatch):
    import super_deli.models.mixture as mixture

    def fail(*args, **kwargs):
        raise ValueError("deliberate unresolved candidate search")

    monkeypatch.setattr(mixture, "select_gaussian_mixture", fail)
    c = configuration()
    declaration = next(r for r in c.plan() if r["case"] == "gaussian_scores")
    record = _observation((str(tmp_path), c.to_dict(), declaration))
    assert record["failed_fits"] == record["attempted_fits"] == 1
    assert record["points"][0]["aic_k"] is None
    assert "deliberate" in record["points"][0]["error"]
    rows = audit_observation((str(tmp_path), declaration, record, c.to_dict(), str(tmp_path)))
    assert len(rows) == 1 and rows[0]["status"] == "failed"


def test_selection_uses_training_only():
    from threadpoolctl import threadpool_limits

    c = configuration().model
    train = np.random.default_rng(391).normal(size=(64, 4))
    with threadpool_limits(limits=1):
        fit = select_gaussian_mixture(
            train, c, 19, covariance_type="full", unit_indices=np.arange(64)
        )
    buffer = io.BytesIO()
    np.savez(buffer, scores=train, **fit.candidate_arrays)
    a, _ = candidate_metrics(buffer.getvalue(), fit.candidates, test_scores=train * 50)
    b, _ = candidate_metrics(buffer.getvalue(), fit.candidates, test_scores=train / 50)
    assert (a["aic_k"], a["bic_k"]) == (b["aic_k"], b["bic_k"])


def test_failed_replay_verification_keeps_evidence(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import super_deli.benchmarking.aic_diagnostic_sources as sources
    import super_deli.workflows.grouped_likelihood as workflow

    c = configuration()

    def fit(*args, **kwargs):
        return {"status": "complete"}

    def reject(*args, **kwargs):
        raise ValueError("deliberate candidate mismatch")

    @contextmanager
    def original(*args, **kwargs):
        yield tmp_path

    monkeypatch.setattr(workflow, "point_fitter", fit)
    monkeypatch.setattr(sources, "_verify_replay", reject)
    monkeypatch.setattr(sources, "observation", original)
    task = {
        "n": 20,
        "repetition": 0,
        "case": "independent_gaussian",
        "data_seed": 383,
        "model": c.model.to_dict(),
    }
    row = sources._replay((str(tmp_path), task, c.to_dict(), str(tmp_path)))
    assert row["status"] == "complete" and not row["verified"]
    assert "deliberate" in row["verification_error"]
    with materialize(tmp_path / row["archive"], row, scratch=tmp_path) as base:
        assert "deliberate" in read_json(base / "verification_failure.json")["error"]
