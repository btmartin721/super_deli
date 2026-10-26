"""Paired native parity, immutable source cohorts, failure accounting and reports."""

# ruff: noqa: E402
import json
from dataclasses import replace
from itertools import combinations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.density_study import generate_raw, write_raw
from super_deli.cli import main
from super_deli.core.data import Observation
from super_deli.core.partition_comparison import compare_partitions
from super_deli.io.config import read_json
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.tabular_analysis_config import TabularAnalysisArm, TabularAnalysisConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.workflows import tabular_analysis as api
from super_deli.workflows.tabular_comparison_report import write_comparison_report

pytestmark = pytest.mark.baseline


def configuration(draws=0):
    names = ("trait_1", "trait_2")
    return TabularAnalysisConfig(
        (
            TabularAnalysisArm(
                "gmm",
                TabularClusterConfig(
                    True, feature_names=names, n_bootstrap=draws, random_seed=78211
                ),
            ),
            TabularAnalysisArm(
                "density",
                HDBSCANConfig(True, feature_names=names, n_subsamples=draws, random_seed=78211),
            ),
        )
    )


@pytest.fixture
def source(tmp_path):
    x, _ = generate_raw("four_spherical", 64, 78233)
    return write_raw(x, tmp_path / "source")


def run(dataset, out, c=None):
    return api.run_tabular_analysis(
        dataset,
        analysis_config=c or configuration(),
        base_dir=out.parent / "source",
        output_dir=out,
    )


def test_direct_workflow_parity_and_hash_bound_outputs(source, tmp_path):
    result = run(source, tmp_path / "analysis", configuration(3))
    assert result["status"] == "complete"
    assert result["inference"]["p_value"] is None
    for arm in configuration(3).arms:
        api._native(arm, source, tmp_path / "source", tmp_path / arm.name, "traits")
        native = tmp_path / "analysis/arms" / arm.name
        for filename in (
            "assignments.csv",
            "bootstrap_replicates.json" if arm.family == "gmm" else "subsamples.json",
        ):
            assert (native / filename).read_bytes() == (tmp_path / arm.name / filename).read_bytes()
        for filename in (
            ("mixture.npz", "preprocessing.npz")
            if arm.family == "gmm"
            else ("point.npz", "stability.npz")
        ):
            with np.load(native / filename) as a, np.load(tmp_path / arm.name / filename) as b:
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_allclose(a[key], b[key], equal_nan=True)
    gmm, density = result["models"]
    assert gmm["stability"]["requested"] == density["stability"]["requested"] == 3
    assert "oob" in gmm["stability"] and "denominator" in density["stability"]
    assert result["comparisons"][0]["joint_pairs"] <= 64 * 63 // 2
    html = (tmp_path / "analysis/report.html").read_text()
    assert "no automatic" in " ".join(result["limitations"]).lower()
    assert "noise" in html and "p-value" in html
    assert not any(p.name == "truth.json" for p in (tmp_path / "analysis").rglob("*"))
    with pytest.raises(FileExistsError):
        run(source, tmp_path / "analysis")


def test_source_mutation_between_arms_cannot_change_frozen_cohort(source, tmp_path, monkeypatch):
    native = api._native

    def mutate(*args):
        result = native(*args)
        (tmp_path / "source/traits.tsv").write_text("changed original after snapshot\n")
        return result

    monkeypatch.setattr(api, "_native", mutate)
    result = run(source, tmp_path / "analysis")
    assert result["status"] == "complete"
    assert result["comparisons"][0]["status"] == "available"


def test_failed_arm_does_not_become_k1_or_block_independent_arm(source, tmp_path, monkeypatch):
    native = api._native

    def fail(arm, *args):
        if arm.family == "gmm":
            raise ValueError("retained deliberate failure")
        return native(arm, *args)

    monkeypatch.setattr(api, "_native", fail)
    result = run(source, tmp_path / "analysis")
    assert result["status"] == "partial"
    assert result["models"][0]["selected_k"] is None
    assert result["models"][1]["status"] == "complete"
    assert result["comparisons"][0]["status"] == "unavailable"


def test_all_noise_is_success_and_agreement_is_undefined(source, tmp_path):
    c = configuration()
    c = replace(
        c,
        arms=(
            c.arms[0],
            replace(
                c.arms[1],
                model=replace(c.arms[1].model, min_cluster_size=100, allow_single_cluster=False),
            ),
        ),
    )
    result = run(source, tmp_path / "analysis", c)
    assert result["status"] == "complete" and result["models"][1]["selected_k"] == 0
    assert result["models"][1]["noise_count"] == 64
    assert result["comparisons"][0]["joint_resolved"] == 0
    assert result["comparisons"][0]["adjusted_rand"] is None


def test_resampling_failure_keeps_point_and_native_failure_bounds(source, tmp_path, monkeypatch):
    from super_deli.models import tabular_hdbscan

    original = tabular_hdbscan.fit_density_once
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("one retained failed draw")
        return original(*args, **kwargs)

    monkeypatch.setattr(tabular_hdbscan, "fit_density_once", fail)
    result = run(source, tmp_path / "analysis", configuration(3))
    assert result["status"] == "partial"
    density = result["models"][1]
    assert density["status"] == "partial" and density["selected_k"] is not None
    assert density["stability"]["failed"] == 1 and calls == 4
    assert result["comparisons"][0]["status"] == "available"


def test_mismatched_child_identity_or_preprocessing_withholds_comparison(
    source, tmp_path, monkeypatch
):
    original = api._read_point

    def mismatch(arm, *args):
        summary, prep, point = original(arm, *args)
        if arm.family == "hdbscan":
            prep["means"] = prep["means"] + 1
        return summary, prep, point

    monkeypatch.setattr(api, "_read_point", mismatch)
    result = run(source, tmp_path / "analysis")
    assert result["models"][1]["status"] == "failed"
    assert result["comparisons"][0]["status"] == "unavailable"


@pytest.mark.parametrize("mode", ["count", "categorical", "missing", "repeated"])
def test_unsupported_input_never_runs_an_arm(source, tmp_path, monkeypatch, mode):
    m = source.modalities[0]
    if mode in {"count", "categorical"}:
        features = (
            replace(
                m.features[0], dtype=mode, categories=("a", "b") if mode == "categorical" else ()
            ),
            m.features[1],
        )
        source = replace(source, modalities=(replace(m, features=features),))
    elif mode == "missing":
        lines = (tmp_path / "source/traits.tsv").read_text().splitlines()
        fields = lines[1].split("\t")
        fields[1] = ""
        lines[1] = "\t".join(fields)
        (tmp_path / "source/traits.tsv").write_text("\n".join(lines) + "\n")
    else:
        obs = (
            m.observations[0],
            Observation(m.observations[1].observation_id, m.observations[0].unit_id),
            *m.observations[2:],
        )
        source = replace(source, modalities=(replace(m, observations=obs),))

    def forbidden(*args):
        raise AssertionError("No arm should run")

    monkeypatch.setattr(api, "_native", forbidden)
    with pytest.raises(ValueError):
        run(source, tmp_path / "analysis")
    assert (tmp_path / "analysis/failure.json").exists()


def test_ecological_locality_table_and_explicit_selection(source, tmp_path):
    m = source.modalities[0]
    ids = tuple(s.specimen_id for s in source.specimens)
    source = replace(
        source, localities=ids, modalities=(replace(m, kind="ecological", level="locality"),)
    )
    result = run(source, tmp_path / "ecology")
    assert result["kind"] == "ecological" and result["level"] == "locality"
    with pytest.raises(ValueError, match="exactly one"):
        api.run_tabular_analysis(
            source,
            analysis_config=configuration(),
            base_dir=tmp_path / "source",
            output_dir=tmp_path / "wrong",
            modality_name="unknown",
        )
    assert not (tmp_path / "wrong").exists()


def test_comparison_matches_sklearn_and_explicit_pairs():
    rng = np.random.default_rng(78301)
    for _ in range(50):
        a, b = rng.integers(-1, 4, (2, 30)).tolist()
        m, n = (rng.random((2, 30)) > 0.1).tolist()
        r = compare_partitions(a, b, m, n)
        ids = [i for i in range(30) if m[i] and n[i] and a[i] >= 0 and b[i] >= 0]
        assert r["adjusted_rand"] == pytest.approx(
            adjusted_rand_score([a[i] for i in ids], [b[i] for i in ids])
        )
        disagreement = sum((a[i] == a[j]) != (b[i] == b[j]) for i, j in combinations(ids, 2))
        assert r["discordant_pairs"] == disagreement


def test_cli_and_escaped_offline_report(source, tmp_path, capsys):
    (tmp_path / "analysis.json").write_text(json.dumps(configuration().to_dict()))
    assert (
        main(
            [
                "analyze-tabular",
                str(tmp_path / "source/dataset.json"),
                "--analysis-config",
                str(tmp_path / "analysis.json"),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    result["dataset_id"] = '<script>alert("x")</script>'
    write_comparison_report(result, {}, tmp_path / "escaped.html")
    html = (tmp_path / "escaped.html").read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "https://" not in html and "<script src" not in html
    assert read_json(tmp_path / "cli/summary.json")["status"] == "complete"


def test_independent_artifact_audit_and_missing_hash_detection(source, tmp_path):
    from super_deli.benchmarking.tabular_comparison_audit import audit_analysis

    run(source, tmp_path / "analysis", configuration(2))
    assert audit_analysis(tmp_path / "analysis") == {"models": 2, "draws": 4, "comparisons": 1}
    (tmp_path / "analysis/models.csv").write_text("changed")
    with pytest.raises(AssertionError):
        audit_analysis(tmp_path / "analysis")


def test_gmm_unresolved_mask_and_single_arm_analysis(source, tmp_path):
    c = configuration()
    arm = replace(c.arms[0], model=replace(c.arms[0].model, assignment_threshold=1))
    result = run(source, tmp_path / "single", replace(c, arms=(arm,)))
    assert result["status"] == "complete" and result["comparisons"] == []
    point = read_json(tmp_path / "single/arms/gmm/comparison-point.json")
    assert result["models"][0]["unresolved_count"] == sum(not x for x in point["resolved"])
    assert result["models"][0]["noise_count"] == 0


def test_all_arm_failures_and_child_identity_mismatch(source, tmp_path, monkeypatch):
    native = api._native

    def mismatch(arm, *args):
        native(arm, *args)
        path = args[2] / "summary.json"
        summary = read_json(path)
        summary["unit_ids"] = list(reversed(summary["unit_ids"]))
        path.write_text(json.dumps(summary))

    monkeypatch.setattr(api, "_native", mismatch)
    result = run(source, tmp_path / "failed")
    assert result["status"] == "failed"
    assert all(row["selected_k"] is None for row in result["models"])
    assert result["comparisons"][0]["status"] == "unavailable"
