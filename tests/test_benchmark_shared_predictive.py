"""Shared-family fit geometry, independent audit, leakage and failure coverage."""

from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.local_archives import materialize  # noqa: E402
from super_deli.benchmarking.shared_predictive_analysis import audit_observation  # noqa: E402
from super_deli.benchmarking.shared_predictive_study import observation  # noqa: E402
from super_deli.io.config import read_json  # noqa: E402
from super_deli.models.shared_predictive_config import SharedPredictiveConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    c = SharedPredictiveConfig.from_dict(read_json(ROOT / "examples/shared-predictive/smoke.json"))
    cell = next(v for v in c.study.cells if v.role == "alternative")
    return replace(c, study=replace(c.study, cells=(cell,)))


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared-paired")
    c = configuration()
    d = c.plan()[0]
    row = observation((str(root), c.to_dict(), d))
    return root, c, d, row


def test_all_candidate_covariances_and_aic_counts(fitted, tmp_path):
    root, _, _, row = fitted
    assert row["successful_fits"] == 12 and row["failed_fits"] == 0
    assert row["maps_verified_identical"] and row["map_comparisons"] == 6
    with materialize(root / row["archive"], row, scratch=tmp_path) as base:
        for family in ("full", "tied"):
            r = row["arms"][family]
            with materialize(base / family / r["archive"], r, scratch=tmp_path) as arm:
                for prefix in ["full"] + [f"fold_{j:03d}" for j in range(5)]:
                    run = arm / "workflow" / prefix / "run"
                    with np.load(run / "gmm_candidates.npz", allow_pickle=False) as p:
                        n, d = p["scores"].shape
                        for c in read_json(run / "candidates.json"):
                            k = c["k"]
                            cov = p[f"k{k}_covariances"]
                            assert cov.shape == (k, d, d)
                            if family == "tied":
                                np.testing.assert_array_equal(cov, np.repeat(cov[:1], k, axis=0))
                            q = k - 1 + k * d + (1 if family == "tied" else k) * d * (d + 1) // 2
                            assert c["aic"] == pytest.approx(-2 * c["log_likelihood"] + 2 * q)
                            assert c["bic"] == pytest.approx(
                                -2 * c["log_likelihood"] + np.log(n) * q
                            )


def test_independent_raw_density_gate_and_export_audit(fitted, tmp_path):
    root, c, d, row = fitted
    audited = audit_observation((str(root), c.to_dict(), d, row, str(tmp_path)))
    assert audited == row


def test_fold_failure_keeps_paired_reference_and_withholds_shared_screen(tmp_path, monkeypatch):
    import super_deli.models.integrative as integration

    original = integration.select_gaussian_mixture
    called = []

    def injected(scores, config, seed, **kwargs):
        if kwargs["covariance_type"] == "tied":
            called.append(1)
            if len(called) == 2:
                raise ValueError("deliberate shared fold failure")
        return original(scores, config, seed, **kwargs)

    monkeypatch.setattr(integration, "select_gaussian_mixture", injected)
    c = configuration()
    d = c.plan()[0]
    row = observation((str(tmp_path), c.to_dict(), d))
    assert row["successful_fits"] == 11 and row["failed_fits"] == 1
    assert row["arms"]["full"]["selection"]["status"] == "complete"
    assert row["arms"]["tied"]["selection"]["status"] == "withheld"
    assert row["arms"]["tied"]["performance"]["screened"] is None
    assert row["map_comparisons"] == 5 and not row["maps_verified_identical"]
    assert audit_observation((str(tmp_path), c.to_dict(), d, row, str(tmp_path))) == row


def test_direct_score_shared_extension_is_rejected(tmp_path):
    from super_deli.workflows.predictive import run_score_screen

    c = configuration()
    with pytest.raises(ValueError, match="full-family"):
        run_score_screen(
            np.zeros((32, 4)),
            model=c.family("tied").model,
            screen_config=c.study.screening,
            output_dir=tmp_path / "bad",
        )
    assert not (tmp_path / "bad").exists()


def test_shared_heldout_values_cannot_enter_training(fitted, tmp_path):
    from super_deli.io.predictive_inputs import subset_raw
    from super_deli.workflows.predictive import fold_model, raw_search

    root, c, _, row = fitted
    r = row["arms"]["tied"]
    with (
        materialize(root / row["archive"], row, scratch=tmp_path) as base,
        materialize(base / "tied" / r["archive"], r, scratch=tmp_path) as arm,
    ):
        layout = read_json(arm / "generating_layout.json")
        fold = read_json(arm / "workflow/fold_assignments.json")[0]
        with np.load(arm / "values.npz", allow_pickle=False) as p:
            values = {k: p[k].copy() for k in p.files}
        changed = {k: v.copy() for k, v in values.items()}
        for view in layout["views"]:
            changed[view["kind"]][np.isin(view["target_ids"], fold["validation_unit_ids"])] += 1e6
        a, al = subset_raw(values, layout, fold["training_indices"])
        b, bl = subset_raw(changed, layout, fold["training_indices"])
        assert al == bl
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        record = raw_search(tmp_path, "replayed", b, bl, fold_model(c.family("tied").model, 0))
        assert record["status"] == "complete"
        for filename in ("gmm_candidates.npz", "fusion.npz"):
            with (
                np.load(arm / "workflow/fold_000/run" / filename, allow_pickle=False) as x,
                np.load(tmp_path / "replayed/run" / filename, allow_pickle=False) as y,
            ):
                for key in x.files:
                    np.testing.assert_array_equal(x[key], y[key])
