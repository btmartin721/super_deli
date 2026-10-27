"""Training-only imputation, raw-cohort accounting and native stability parity."""

# ruff: noqa: E402
import csv
import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli.cli import main
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.tabular_analysis_config import TabularAnalysisArm, TabularAnalysisConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.models.tabular_gmm import fit_tabular_pca, project_tabular_pca
from super_deli.models.tabular_missing import TrainingContext
from super_deli.models.tabular_missing_config import TabularMissingAnalysisConfig
from super_deli.workflows.tabular_analysis import run_tabular_analysis

NAMES = tuple(f"trait_{i}" for i in range(4))


def write_data(path, x, *, bad=False):
    path.mkdir()
    ids = [f"unit_{i:04}" for i in range(len(x))]
    with (path / "traits.tsv").open("w", newline="") as h:
        writer = csv.writer(h, delimiter="\t", lineterminator="\n")
        writer.writerow(["observation_id", *NAMES])
        writer.writerows(
            [u, *("" if np.isnan(v) else v for v in row)] for u, row in zip(ids, x, strict=True)
        )
    if bad:
        (path / "traits.tsv").write_text(
            (path / "traits.tsv").read_text().replace("\t", "\tNANA", 1)
        )
    d = config_from_dict(
        dict(
            schema_version=1,
            dataset_id="invented_missing",
            specimens=[dict(specimen_id=u) for u in ids],
            modalities=[
                dict(
                    name="traits",
                    kind="morphological",
                    level="specimen",
                    format="tsv",
                    path="traits.tsv",
                    features=[dict(name=f, dtype="continuous") for f in NAMES],
                    observations=[dict(observation_id=u, unit_id=u) for u in ids],
                )
            ],
        )
    )
    (path / "dataset.json").write_text(json.dumps(d.to_dict()))
    return d


def configuration(draws=0, policies=("complete_case", "mean")):
    arms = (
        TabularAnalysisArm(
            "gmm",
            TabularClusterConfig(True, feature_names=NAMES, n_bootstrap=draws, random_seed=9941),
        ),
        TabularAnalysisArm(
            "density",
            HDBSCANConfig(True, feature_names=NAMES, n_subsamples=draws, random_seed=9941),
        ),
    )
    return TabularMissingAnalysisConfig(arms, policies)


def run(tmp_path, x, *, draws=0, policies=("complete_case", "mean"), name="analysis"):
    raw = tmp_path / (name + "-raw")
    d = write_data(raw, x)
    return run_tabular_analysis(
        d, analysis_config=configuration(draws, policies), base_dir=raw, output_dir=tmp_path / name
    )


def test_complete_data_matches_legacy_native_arrays(tmp_path):
    x = np.random.default_rng(991).normal(size=(32, 4))
    d = write_data(tmp_path / "raw", x)
    c = configuration()
    run_tabular_analysis(
        d,
        analysis_config=TabularAnalysisConfig(c.arms),
        base_dir=tmp_path / "raw",
        output_dir=tmp_path / "legacy",
    )
    s = run_tabular_analysis(
        d, analysis_config=c, base_dir=tmp_path / "raw", output_dir=tmp_path / "new"
    )
    assert s["status"] == "complete"
    for policy in c.policies:
        for arm in c.arms:
            f = "preprocessing.npz" if arm.family == "gmm" else "point.npz"
            with (
                np.load(tmp_path / "legacy/arms" / arm.name / f) as a,
                np.load(tmp_path / "new/policies" / policy / "analysis/arms" / arm.name / f) as b,
            ):
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


def test_policies_preserve_raw_masks_and_exclude_unobserved_not_noise(tmp_path):
    x = np.random.default_rng(993).normal(size=(32, 4))
    x[0] = np.nan
    x[1, 0] = np.nan
    x[2, 1] = np.nan
    s = run(tmp_path, x, draws=3)
    assert s["policies"]["complete_case"]["eligible_units"] == 29
    assert s["policies"]["mean"]["eligible_units"] == 31
    records = read_json(tmp_path / "analysis/policies/mean/eligibility.json")
    assert records[0]["exclusion_reason"] == "all_selected_missing"
    assert records[1]["values"][0] is None
    assert s["inference"]["p_value"] is None
    for arm in ("gmm", "density"):
        root = tmp_path / "analysis/policies/mean/analysis/arms" / arm
        fits = read_json(root / "training-context.json")["fits"]
        assert len(fits) == 4
        with np.load(root / "training-context.npz") as a:
            for i, fit in enumerate(fits):
                raw = a[f"fit_{i}_raw"]
                keep = a[f"fit_{i}_kept_indices"]
                np.testing.assert_allclose(a[f"fit_{i}_means"], np.nanmean(raw, axis=0)[keep])
                assert fit["missing_cells"] == int(np.isnan(raw).sum())
                if i:
                    assert not np.array_equal(a[f"fit_{i}_means"], a["fit_0_means"])
    assert all(v["n_units"] == 32 for v in s["comparisons"])


def test_all_missing_training_feature_is_new_context_failure_not_legacy_change():
    x = np.random.default_rng(997).normal(size=(24, 4))
    x[:, 3] = np.nan
    c = TabularClusterConfig(True, feature_names=NAMES, n_bootstrap=0, missing_policy="mean")
    old, _ = fit_tabular_pca(x, NAMES, ("continuous",) * 4, c)
    assert 3 not in old.kept_indices
    context = TrainingContext("mean")
    with pytest.raises(ValueError, match="no observed"):
        context.fit_pca(x, NAMES, ("continuous",) * 4, c)
    assert context.records[0]["preprocessing_status"] == "failed"


@pytest.mark.parametrize("kind", ["empty", "all_missing_feature", "too_few"])
def test_failed_cohorts_are_not_k1_or_noise(tmp_path, kind):
    x = np.random.default_rng(998).normal(size=(24, 4))
    if kind == "empty":
        x[:] = np.nan
    elif kind == "all_missing_feature":
        x[:, 0] = np.nan
    else:
        x[2:] = np.nan
    s = run(tmp_path, x, policies=("mean",))
    assert s["status"] == "failed" and all(m["selected_k"] is None for m in s["models"])


def test_projection_uses_training_means_and_rejects_all_unobserved():
    x = np.random.default_rng(999).normal(size=(24, 4))
    x[0, 0] = np.nan
    c = TabularClusterConfig(True, feature_names=NAMES, n_bootstrap=0, missing_policy="mean")
    model, _ = TrainingContext("mean").fit_pca(x, NAMES, ("continuous",) * 4, c)
    new = np.array([[np.nan, 100.0, 101.0, 102.0]])
    expected = np.where(np.isnan(new), model.means, new)
    np.testing.assert_allclose(
        project_tabular_pca(new, model=model),
        model.pca.transform((expected - model.means) / model.scales),
    )
    with pytest.raises(ValueError):
        project_tabular_pca(np.full((1, 4), np.nan), model=model)


def test_cli_versioned_dispatch(tmp_path, capsys):
    x = np.random.default_rng(9913).normal(size=(24, 4))
    x[0, 0] = np.nan
    write_data(tmp_path / "raw", x)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(configuration().to_dict()))
    assert (
        main(
            [
                "analyze-tabular",
                str(tmp_path / "raw/dataset.json"),
                "--analysis-config",
                str(p),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["schema_version"] == 2


def test_independent_training_and_support_audit(tmp_path):
    from super_deli.benchmarking.tabular_missing_audit import audit_policy

    x = np.random.default_rng(9931).normal(size=(32, 4))
    mask = np.zeros(x.shape, dtype=bool)
    mask[0, :] = True
    mask[1, 0] = True
    hidden = x.copy()
    hidden[mask] = np.nan
    run(tmp_path, hidden, draws=3)
    for policy in ("complete_case", "mean"):
        metrics, checked, draws = audit_policy(
            tmp_path / "analysis/policies" / policy, x, mask, policy, np.zeros(32, dtype=int)
        )
        assert checked == 8 and draws == 6
        assert all(m["status"] == "complete" for m in metrics)


@pytest.mark.parametrize("token", ["NANA", "0.2?", "nan", "inf"])
def test_malformed_tokens_do_not_become_missing(tmp_path, token):
    x = np.random.default_rng(9937).normal(size=(24, 4))
    d = write_data(tmp_path / "raw", x)
    p = tmp_path / "raw/traits.tsv"
    lines = p.read_text().splitlines()
    row = lines[1].split("\t")
    row[1] = token
    lines[1] = "\t".join(row)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        run_tabular_analysis(
            d,
            analysis_config=configuration(),
            base_dir=tmp_path / "raw",
            output_dir=tmp_path / "bad",
        )
    assert (tmp_path / "bad/failure.json").exists()
    assert not (tmp_path / "bad/summary.json").exists()


def test_study_budget_and_disjoint_seed_namespaces():
    from super_deli.benchmarking.tabular_missing_study import tasks

    for phase, n, fits in (("preflight", 6, 408), ("main", 60, 1836)):
        plan = tasks(phase)
        assert len(plan) == n
        assert sum(22 + 40 * t["stability"] + 6 * t["replay"] for t in plan) == fits
        assert len({tuple(t["data_seed"]) for t in plan}) == n
    assert {tuple(t["data_seed"]) for t in tasks("main")}.isdisjoint(
        {tuple(t["data_seed"]) for t in tasks("preflight")}
    )


def test_per_fit_retention_budget_is_checked_before_resampling():
    with pytest.raises(ValueError, match="max_working_bytes"):
        TrainingContext("mean").check_budget(np.zeros((32, 4)), 100000, 1024 * 1024)


def test_metadata_and_reference_labels_do_not_affect_cohort_selection():
    from super_deli.core.data import Feature
    from super_deli.core.tabular import TabularTable
    from super_deli.workflows.tabular_missing import cohort_rows

    features = tuple(Feature(f, "continuous") for f in NAMES) + (
        Feature("label", "continuous", role="covariate"),
    )
    t = TabularTable(
        "traits",
        "morphological",
        "specimen",
        features,
        ("a", "b"),
        ("a", "b"),
        ((1.0, 2.0, 3.0, 4.0, None), (None, 2.0, 3.0, 4.0, 123.0)),
    )
    other = replace(t, values=((1.0, 2.0, 3.0, 4.0, 999.0), (None, 2.0, 3.0, 4.0, None)))
    for policy in ("complete_case", "mean"):
        assert cohort_rows(t, NAMES, policy) == cohort_rows(other, NAMES, policy)
    with pytest.raises(ValueError):
        cohort_rows(t, ("label",), "mean")
    with pytest.raises(ValueError):
        cohort_rows(t, NAMES, "unknown")


def test_late_source_change_withholds_completed_analysis(tmp_path, monkeypatch):
    import super_deli.workflows.tabular_analysis as native

    x = np.random.default_rng(9943).normal(size=(24, 4))
    d = write_data(tmp_path / "raw", x)
    original = native._native

    def changed(*args, **kwargs):
        r = original(*args, **kwargs)
        p = tmp_path / "raw/traits.tsv"
        p.write_text(p.read_text() + "\n")
        return r

    monkeypatch.setattr(native, "_native", changed)
    with pytest.raises(ValueError, match="input changed"):
        run_tabular_analysis(
            d,
            analysis_config=configuration(),
            base_dir=tmp_path / "raw",
            output_dir=tmp_path / "changed",
        )
    assert (tmp_path / "changed/failure.json").exists()
    assert not (tmp_path / "changed/summary.json").exists()


def test_failed_refit_keeps_point_and_native_failure_bounds(tmp_path, monkeypatch):
    original = TrainingContext.fit_pca

    def failed(self, *args):
        result = original(self, *args)
        if len(self.records) == 2:
            raise ValueError("deliberate refit failure")
        return result

    monkeypatch.setattr(TrainingContext, "fit_pca", failed)
    x = np.random.default_rng(9949).normal(size=(32, 4))
    x[0, 0] = np.nan
    s = run(tmp_path, x, draws=3, policies=("mean",))
    assert s["status"] == "partial"
    assert all(m["selected_k"] is not None for m in s["models"])
    assert all(m["stability"]["failed"] == 1 for m in s["models"])
