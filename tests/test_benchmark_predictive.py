"""Numerical whole-locality, leakage, retained-parameter and audit contracts."""

from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from super_deli.benchmarking.local_archives import materialize  # noqa: E402
from super_deli.benchmarking.predictive_audit import audit_observation  # noqa: E402
from super_deli.benchmarking.predictive_study import _observation  # noqa: E402
from super_deli.io.config import read_json  # noqa: E402
from super_deli.io.predictive_inputs import subset_raw  # noqa: E402
from super_deli.models.predictive_config import PredictiveScreenConfig  # noqa: E402
from super_deli.models.predictive_selection import fold_assignments  # noqa: E402
from super_deli.models.predictive_study_config import PredictiveStudyConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    c = PredictiveStudyConfig.from_dict(read_json(ROOT / "examples/predictive-aic/smoke.json"))
    return replace(c, cells=(replace(c.cells[0], n_units=32),), min_free_bytes=1024**2)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("predictive-screen")
    c = configuration()
    declaration = c.plan()[0]
    record = _observation((str(root), c.to_dict(), declaration))
    assert record["successful_fits"] == 6 and record["failed_fits"] == 0
    return root, c, declaration, record


def test_every_locality_held_out_once_with_all_members(fitted, tmp_path):
    root, c, _, record = fitted
    with materialize(root / record["archive"], record, scratch=tmp_path) as base:
        layout = read_json(base / "generating_layout.json")
        with np.load(base / "values.npz", allow_pickle=False) as p:
            values = {k: p[k] for k in p.files}
        seen = []
        for f in read_json(base / "workflow/fold_assignments.json"):
            seen.extend(f["validation_indices"])
            arrays, selected = subset_raw(values, layout, f["training_indices"])
            expected = set(f["training_unit_ids"])
            assert set(selected["unit_ids"]) == expected
            assert len(arrays["morphological"]) == sum(selected["counts"])
            assert all(s["locality_id"] in expected for s in selected["specimens"])
            assert set(selected["views"][0]["target_ids"]) == expected
        assert sorted(seen) == list(range(32)) and len(seen) == 32
        assert fold_assignments(
            layout["unit_ids"], PredictiveScreenConfig(random_seed=record["split_seed"])
        ) == read_json(base / "workflow/fold_assignments.json")


def test_independent_numerical_audit_and_exported_memberships(fitted, tmp_path):
    root, c, declaration, record = fitted
    row = audit_observation((str(root), declaration, record, c.to_dict(), str(tmp_path)))
    assert row["selection"]["status"] == "complete"
    assert row["performance"]["screened"]["k"] == row["selection"]["screened_k"]


def test_changed_validation_values_cannot_change_training_fold(fitted, tmp_path):
    from threadpoolctl import threadpool_limits

    from super_deli.workflows.predictive import raw_search

    root, c, _, record = fitted
    with materialize(root / record["archive"], record, scratch=tmp_path) as base:
        layout = read_json(base / "generating_layout.json")
        split = read_json(base / "workflow/fold_assignments.json")[0]
        with np.load(base / "values.npz", allow_pickle=False) as p:
            original = {k: p[k].copy() for k in p.files}
        changed = {k: v.copy() for k, v in original.items()}
        for view in layout["views"]:
            mask = np.isin(view["target_ids"], split["validation_unit_ids"])
            changed[view["kind"]][mask] += 1000
        a, la = subset_raw(original, layout, split["training_indices"])
        b, lb = subset_raw(changed, layout, split["training_indices"])
        assert la == lb
        for kind in a:
            np.testing.assert_array_equal(a[kind], b[kind])
        # The very arrays re-ingested for the fold stay identical after arbitrary
        # changes to validation units; the saved original fold is audited separately.
        with threadpool_limits(limits=1):
            result = raw_search(
                tmp_path,
                "refit",
                b,
                lb,
                type(c.model).from_dict(read_json(base / "workflow/fold_000/model.json")),
            )
        assert result["status"] == "complete"
        for i in range(2):
            with (
                np.load(
                    base / f"workflow/fold_000/run/view_{i:03d}/parameters/pca.npz",
                    allow_pickle=False,
                ) as x,
                np.load(
                    tmp_path / f"refit/run/view_{i:03d}/parameters/pca.npz", allow_pickle=False
                ) as y,
            ):
                for name in x.files:
                    np.testing.assert_array_equal(x[name], y[name])


def test_fold_failure_withholds_screen_without_replacing_searches(tmp_path, monkeypatch):
    import super_deli.workflows.predictive as workflow
    from super_deli.benchmarking.grouped_study import simulate_grouped
    from super_deli.models.grouped_null import write_raw_views

    c = configuration()
    values, layout, _ = simulate_grouped("independent_gaussian", 32, 38777)
    dataset = write_raw_views(values, layout, tmp_path / "input")

    def fail(base, prefix, *args, **kwargs):
        directory = Path(base) / prefix
        directory.mkdir()
        return {"status": "failed", "prefix": prefix, "error": "deliberate fold failure"}

    monkeypatch.setattr(workflow, "raw_search", fail)
    result = workflow.run_predictive(
        dataset,
        model_config=c.model,
        screen_config=c.screening,
        base_dir=tmp_path / "input",
        output_dir=tmp_path / "fit",
    )
    assert result["attempted_fits"] == 6 and result["failed_fits"] == 5
    assert result["selection"]["status"] == "withheld" and result["selection"]["screened_k"] is None
    assert not (tmp_path / "fit/selected_model.npz").exists()


def test_real_pipeline_failure_is_archived_and_independently_audited(tmp_path, monkeypatch):
    import super_deli.models.integrative as integration

    c = configuration()
    original = integration.select_gaussian_mixture
    calls = []

    def fail_fold(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("deliberate failed fold candidate search")
        return original(*args, **kwargs)

    monkeypatch.setattr(integration, "select_gaussian_mixture", fail_fold)
    declaration = c.plan()[0]
    row = _observation((str(tmp_path), c.to_dict(), declaration))
    assert row["successful_fits"] == 5 and row["failed_fits"] == 1
    assert row["selection"]["status"] == "withheld" and row["performance"]["screened"] is None
    audited = audit_observation((str(tmp_path), declaration, row, c.to_dict(), str(tmp_path)))
    assert audited["selection"]["screened_k"] is None
    with materialize(tmp_path / row["archive"], row, scratch=tmp_path) as base:
        assert "deliberate" in read_json(base / "workflow/fold_000/record.json")["error"]


def test_screened_output_is_rejected_by_score_significance_adapter(fitted, tmp_path):
    from types import SimpleNamespace

    from super_deli.workflows.score_calibration import _source_scores

    root, _, _, record = fitted
    with materialize(root / record["archive"], record, scratch=tmp_path) as base:
        with pytest.raises(ValueError, match="supported integrative clustering run"):
            _source_scores(base / "workflow", SimpleNamespace(max_working_bytes=268435456))
