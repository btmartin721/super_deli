"""Native parity, explicit group counts, failures and fixed-ID resampling."""

# ruff: noqa: E402
import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.grouped_analysis_study import compare_replay, model_arms
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.cli import main
from super_deli.io.config import read_json
from super_deli.models.grouped_analysis_config import GroupedAnalysisConfig
from super_deli.models.grouped_null import write_raw_views
from super_deli.models.predictive_fold_size import PredictiveFoldSizePolicy
from super_deli.workflows import grouped_analysis as api
from super_deli.workflows.grouped_analysis_adapters import run_grouped_arm

pytestmark = pytest.mark.baseline


@pytest.fixture
def source(tmp_path):
    values, layout, _ = simulate_grouped("complementary", 32, 79501)
    return write_raw_views(values, layout, tmp_path / "raw")


def run(dataset, tmp_path, c=None, name="analysis"):
    return api.run_grouped_tabular_analysis(
        dataset,
        analysis_config=c or model_arms(),
        base_dir=tmp_path / "raw",
        output_dir=tmp_path / name,
    )


def test_all_six_native_point_replays_and_counts(source, tmp_path):
    c = model_arms()
    summary = run(source, tmp_path, c)
    assert summary["status"] == "complete"
    assert len(summary["comparisons"]) == 15 and summary["inference"]["p_value"] is None
    assert sum(g["morphology_count"] for g in summary["groups"]) == len(source.specimens)
    assert all(g["ecology_count"] == 1 for g in summary["groups"])
    assert len({g["morphology_count"] for g in summary["groups"]}) > 1
    for arm in c.arms:
        run_grouped_arm(arm, source, tmp_path / "raw", tmp_path / arm.name)
        compare_replay(arm, tmp_path / "analysis/arms" / arm.name, tmp_path / arm.name)
    with pytest.raises(FileExistsError):
        run(source, tmp_path, c)
    assert not list((tmp_path / "analysis").rglob("truth.npz"))
    html = (tmp_path / "analysis/report.html").read_text()
    assert "Explicit target sampling" in html and "Native screening decisions" in html


def test_shared_predictive_stability_and_scaled_native_parity(source, tmp_path):
    arm = replace(model_arms(2).arms[-1], fold_size_policy=PredictiveFoldSizePolicy())
    summary = run(source, tmp_path, GroupedAnalysisConfig((arm,)))
    assert summary["models"][0]["stability"]["requested"] == 2
    run_grouped_arm(arm, source, tmp_path / "raw", tmp_path / "native")
    actual = tmp_path / "analysis/arms" / arm.name
    assert read_json(actual / "stability.json") == read_json(tmp_path / "native/stability.json")
    for p in actual.glob("draw_*.json"):
        assert p.read_bytes() == (tmp_path / "native" / p.name).read_bytes()
    assert read_json(actual / "point/fold_assignments.json") == read_json(
        tmp_path / "native/point/fold_assignments.json"
    )


@pytest.mark.parametrize("i", [0, 1, 2])
def test_supported_ordinary_stability_is_native(source, tmp_path, i):
    arm = model_arms().arms[i]
    arm = replace(arm, model=replace(arm.model, n_bootstrap=2))
    summary = run(source, tmp_path, GroupedAnalysisConfig((arm,)))
    assert summary["models"][0]["stability"]["requested"] == 2
    run_grouped_arm(arm, source, tmp_path / "raw", tmp_path / "native")
    assert (tmp_path / "native/bootstrap_replicates.json").read_bytes() == (
        tmp_path / "analysis/arms" / arm.name / "bootstrap_replicates.json"
    ).read_bytes()


def test_native_screen_withheld_preserves_ordinary_fit(source, tmp_path, monkeypatch):
    native = api.run_grouped_arm

    def fail_after_full(arm, *args):
        native(arm, *args)
        if arm.kind == "predictive":
            path = args[-1] / "summary.json"
            summary = read_json(path)
            summary["selection"] = {
                "status": "withheld",
                "screened_k": None,
                "error": "retained test failure",
            }
            path.write_text(json.dumps(summary))

    monkeypatch.setattr(api, "run_grouped_arm", fail_after_full)
    c = model_arms()
    summary = run(source, tmp_path, GroupedAnalysisConfig((c.arms[0], c.arms[4])))
    assert summary["status"] == "partial"
    row = summary["models"][1]
    assert row["ordinary_available"] and row["ordinary_k"] is not None
    assert row["selected_k"] is None and row["status"] == "withheld"
    assert summary["comparisons"][0]["status"] == "unavailable"


def test_late_exception_preserves_ordinary_and_other_arms(source, tmp_path, monkeypatch):
    native = api.run_grouped_arm

    def fail(arm, *args):
        native(arm, *args)
        if arm.kind == "predictive":
            (args[-1] / "summary.json").unlink()
            raise RuntimeError("late screen exception")

    monkeypatch.setattr(api, "run_grouped_arm", fail)
    c = model_arms()
    result = run(source, tmp_path, GroupedAnalysisConfig((c.arms[0], c.arms[4])))
    assert result["models"][0]["status"] == "complete"
    assert result["models"][1]["ordinary_available"]
    assert result["models"][1]["status"] == "withheld"


def test_population_targets_point_analysis(source, tmp_path):
    values, layout, _ = simulate_grouped(
        "independent_gaussian", 32, 79509, target_level="population"
    )
    dataset = write_raw_views(values, layout, tmp_path / "population")
    result = api.run_grouped_tabular_analysis(
        dataset,
        analysis_config=model_arms(target_level="population"),
        base_dir=tmp_path / "population",
        output_dir=tmp_path / "analysis",
    )
    assert result["status"] == "complete" and result["level"] == "population"
    assert result["models"][-1]["stability_status"] == "unsupported"


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "count",
        "categorical",
        "duplicate_ecology",
        "missing_target",
        "repeated_morphology",
    ],
)
def test_bad_inputs_rejected_before_fitting(source, tmp_path, monkeypatch, mode):
    m, e = source.modalities
    if mode in {"count", "categorical"}:
        f = replace(
            m.features[0], dtype=mode, categories=("a", "b") if mode == "categorical" else ()
        )
        m = replace(m, features=(f, *m.features[1:]))
        source = replace(source, modalities=(m, e))
    elif mode == "missing":
        path = tmp_path / "raw" / m.path
        lines = path.read_text().splitlines()
        cells = lines[1].split(",")
        cells[1] = ""
        lines[1] = ",".join(cells)
        path.write_text("\n".join(lines) + "\n")
    elif mode == "duplicate_ecology":
        e = replace(
            e,
            observations=(
                e.observations[0],
                replace(e.observations[1], unit_id=e.observations[0].unit_id),
                *e.observations[2:],
            ),
        )
        source = replace(source, modalities=(m, e))
    elif mode == "repeated_morphology":
        m = replace(
            m,
            observations=(
                m.observations[0],
                replace(m.observations[1], unit_id=m.observations[0].unit_id),
                *m.observations[2:],
            ),
        )
        source = replace(source, modalities=(m, e))
    else:
        source = replace(source, localities=(*source.localities, "unobserved"))

    def forbidden(*args):
        raise AssertionError("Should not fit")

    monkeypatch.setattr(api, "run_grouped_arm", forbidden)
    with pytest.raises(ValueError):
        run(source, tmp_path)
    assert (tmp_path / "analysis/failure.json").exists()


def test_cli_exit_and_source_mutation_isolation(source, tmp_path, monkeypatch, capsys):
    native = api.run_grouped_arm

    def mutate(*args):
        result = native(*args)
        for path in (tmp_path / "raw").glob("*.csv"):
            path.write_text("changed original after snapshot")
        return result

    monkeypatch.setattr(api, "run_grouped_arm", mutate)
    (tmp_path / "analysis.json").write_text(json.dumps(model_arms().to_dict()))
    assert (
        main(
            [
                "analyze-grouped-tabular",
                str(tmp_path / "raw/dataset.json"),
                "--analysis-config",
                str(tmp_path / "analysis.json"),
                "--output-dir",
                str(tmp_path / "analysis"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert all(r["status"] == "available" for r in result["comparisons"])


def test_identity_mismatch_withholds_point(source, tmp_path, monkeypatch):
    original = api.read_grouped_arm

    def wrong(arm, *args):
        result = original(arm, *args)
        if arm.name == "shared_bic":
            result["parameters"]["fusion.npz:mean"] += 1
        return result

    monkeypatch.setattr(api, "read_grouped_arm", wrong)
    c = model_arms()
    summary = run(source, tmp_path, GroupedAnalysisConfig(c.arms[:2]))
    assert summary["models"][1]["status"] == "failed"
    assert summary["comparisons"][0]["status"] == "unavailable"


def test_independent_geometry_density_screen_and_raw_checks(source, tmp_path):
    from super_deli.benchmarking.grouped_analysis_audit import (
        PointGeometryPaths,
        audit_screen,
        audit_winner,
        independent_raw,
    )
    from super_deli.benchmarking.grouped_analysis_study import CASES, generate
    from super_deli.benchmarking.predictive_bootstrap_audit import audit_geometry

    values, layout, _ = simulate_grouped("complementary", 32, 79501)
    layout = json.loads(json.dumps(layout))
    run(source, tmp_path)
    for a in model_arms().arms:
        d = tmp_path / "analysis/arms" / a.name
        if a.kind == "predictive":
            assert audit_screen(d, values, layout, a) == 6
        else:
            audit_geometry(PointGeometryPaths(d), values, layout, np.arange(32), a.model)
            assert audit_winner(d, a.model) > 0
    for case in CASES:
        task = dict(case=case, n=32, data_seed=79501)
        x, _, _ = generate(task)
        y, _, _ = independent_raw(task)
        for key in x:
            np.testing.assert_allclose(x[key], y[key], atol=4e-14, rtol=4e-14)


def test_raw_child_tampering_is_rejected(source, tmp_path, monkeypatch):
    native = api.run_grouped_arm

    def tamper(arm, *args):
        result = native(arm, *args)
        if arm.name == "shared_bic":
            path = args[-1] / "view_000/source_values.npy"
            a = np.load(path)
            a[0, 0] += 1
            np.save(path, a)
        return result

    monkeypatch.setattr(api, "run_grouped_arm", tamper)
    c = model_arms()
    summary = run(source, tmp_path, GroupedAnalysisConfig(c.arms[:2]))
    assert summary["models"][1]["status"] == "failed"
    assert "source values" in summary["models"][1]["error"]


def test_partial_cli_exit_preserves_other_arm(source, tmp_path, monkeypatch, capsys):
    native = api.run_grouped_arm

    def fail(arm, *args):
        if arm.name == "shared_bic":
            raise ValueError("retained failed point")
        return native(arm, *args)

    monkeypatch.setattr(api, "run_grouped_arm", fail)
    c = model_arms()
    c = GroupedAnalysisConfig(c.arms[:2])
    (tmp_path / "analysis.json").write_text(json.dumps(c.to_dict()))
    assert (
        main(
            [
                "analyze-grouped-tabular",
                str(tmp_path / "raw/dataset.json"),
                "--analysis-config",
                str(tmp_path / "analysis.json"),
                "--output-dir",
                str(tmp_path / "analysis"),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["models"][0]["status"] == "complete"


def test_complete_retained_draw_audit_without_refits(tmp_path):
    from super_deli.benchmarking.grouped_analysis_audit import audit_observation
    from super_deli.benchmarking.grouped_analysis_study import _observation

    task = dict(
        case="unequal_variance",
        n=32,
        data_seed=79513,
        draws=2,
        replay=True,
        key="test",
        phase="test",
        dataset=0,
    )
    directory = tmp_path / "evidence"
    result = _observation({**task, "output": str(directory)})
    assert result["status"] == "complete"
    audit = audit_observation(directory, task)
    assert audit["searches"] == 44 and not audit["failures"]
    candidate = directory / "replays/full_aic/gmm_candidates.npz"
    original_bytes = candidate.read_bytes()
    with np.load(candidate) as pool:
        values = {k: pool[k].copy() for k in pool.files}
    values["k4_means"][0, 0] += 1
    np.savez(candidate, **values)
    arm = model_arms().arms[2]
    with pytest.raises(AssertionError):
        compare_replay(arm, directory / "analysis/arms/full_aic", directory / "replays/full_aic")
    candidate.write_bytes(original_bytes)
    (directory / "analysis/models.csv").write_text("tampered")
    with pytest.raises(AssertionError):
        audit_observation(directory, task)
