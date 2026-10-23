"""Full raw refits, independent PCA audits and retained failures for grouped means."""

# ruff: noqa: E402
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.grouped_analysis import audit_grouped_study, point, verify_model
from super_deli.benchmarking.grouped_study import run_grouped_study, simulate_grouped
from super_deli.benchmarking.tabular_pca_audit import audit_tabular_views
from super_deli.io.config import read_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.grouped_config import (
    GroupedCalibrationConfig,
    GroupedStudyConfig,
    study_cases,
)
from super_deli.models.grouped_null import fit_null, simulate_null, write_raw_views
from super_deli.workflows.grouped_calibration import run_grouped_calibration

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.baseline


def policy(level="locality", family="tied"):
    config = GroupedCalibrationConfig.from_dict(
        read_json(ROOT / "examples/groupmean-calibration/calibration.json")
    )
    return replace(
        config,
        n_simulations=2,
        model=replace(config.model, target_level=level, covariance_type=family, n_init=1, max_k=2),
    )


@pytest.mark.parametrize("family", ["full", "tied"])
@pytest.mark.parametrize("level", ["population", "locality"])
@pytest.mark.parametrize("reverse", [False, True])
def test_groupmeans_and_all_preprocessing_refit_for_every_raw_draw(
    tmp_path, monkeypatch, level, reverse, family
):
    from super_deli.workflows import integrative

    c = policy(level, family)
    if reverse:
        c = replace(c, model=replace(c.model, views=c.model.views[::-1]))
    values, layout, _ = simulate_grouped("correlated_gaussian", 64, 193, target_level=level)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    before = {p: p.read_bytes() for p in (tmp_path / "raw").iterdir()}
    runner, calls = integrative.run_integrative, []

    def counted(*args, **kwargs):
        calls.append(kwargs["model_config"])
        return runner(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", counted)
    out = tmp_path / "calibration"
    result = run_grouped_calibration(dataset, config=c, base_dir=tmp_path / "raw", output_dir=out)
    assert result["successful_fits"] == 3 and len(calls) == 3
    assert all(m == c.model and m.n_bootstrap == 0 for m in calls)
    assert "equally weighted raw group means" in result["scope"]
    assert "group-mean morphology PCA" in read_json(out / "evidence.json")["method"]
    assert "nonregular" in result["inference"]["p_value_kind"]
    # The fitted layout follows the declared model view order.
    ordered_layout = {**layout, "views": layout["views"][::-1]} if reverse else layout
    fitted, draw, _, _ = verify_model(tmp_path, out, values, ordered_layout, c)
    means, components = [], []
    with ZipFile(out / "pipeline_artifacts.zip") as archive:
        for record in [result["point"], *read_json(out / "null_replicates.json")]:
            raw = values if record is result["point"] else draw(record["simulation_seed"])
            point(archive, record, c, raw, fitted)
            index = next(i for i, v in enumerate(c.model.views) if v.kind == "morphological")
            name = record["archive_prefix"] + f"/run/view_{index:03d}/parameters/pca.npz"
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as arrays:
                means.append(arrays["feature_means"].copy())
                components.append(arrays["components"].copy())
    assert all(not np.array_equal(means[0], m) for m in means[1:])
    assert all(not np.array_equal(components[0], m) for m in components[1:])
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("family", ["full", "tied"])
def test_raw_law_and_same_seed_draws_are_independent_of_pca_policy(tmp_path, family):
    values, layout, _ = simulate_grouped("zero_conditional_gaussian", 64, 193)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    tables = ingest_tabular(
        dataset, base_dir=tmp_path / "raw", output_dir=tmp_path / "ingestion"
    ).tables
    target = policy(family=family)
    source = replace(
        target,
        model=replace(
            target.model, views=tuple(replace(v, pca_fit="source_rows") for v in target.model.views)
        ),
    )
    a, b = (fit_null(dataset, tables, c) for c in (source, target))
    assert a.layout == b.layout
    for name, array in a.model_arrays().items():
        np.testing.assert_array_equal(array, b.model_arrays()[name])
    for name, array in simulate_null(a, 239).items():
        np.testing.assert_array_equal(array, simulate_null(b, 239)[name])


@pytest.mark.parametrize("family", ["full", "tied"])
@pytest.mark.parametrize("failure_call", [1, 2])
def test_failed_groupmean_point_or_draw_withholds_p_without_replacement(
    tmp_path, monkeypatch, failure_call, family
):
    from super_deli.workflows import integrative

    values, layout, _ = simulate_grouped("independent_gaussian", 64, 193)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    runner, calls = integrative.run_integrative, []

    def injected(*args, **kwargs):
        calls.append(None)
        if len(calls) == failure_call:
            raise ValueError("injected target-mean failure")
        return runner(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", injected)
    result = run_grouped_calibration(
        dataset,
        config=policy(family=family),
        base_dir=tmp_path / "raw",
        output_dir=tmp_path / "out",
    )
    assert result["inference"]["p_value"] is None
    assert result["inference"]["reject_grouped_raw_gaussian"] is None
    assert result["failed_fits"] == 1 and len(calls) == (1 if failure_call == 1 else 3)
    assert result["unattempted_fits"] == (2 if failure_call == 1 else 0)


@pytest.mark.parametrize("family", ["full", "tied"])
def test_seven_case_audit_never_refits_and_reconstructs_pca(tmp_path, monkeypatch, family):
    from super_deli.models import grouped_conditional, grouped_gaussian, mixture, tabular_gmm
    from super_deli.workflows import integrative

    study = GroupedStudyConfig.from_dict(
        read_json(ROOT / "examples/groupmean-calibration/smoke.json")
    )
    study = replace(
        study,
        calibration=replace(
            study.calibration,
            n_simulations=1,
            model=replace(study.calibration.model, covariance_type=family),
        ),
    )
    root = tmp_path / "study"
    root.mkdir()
    for case in study_cases(study):
        run_grouped_study(study, case=case, output_dir=root / case)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def refuse(*args, **kwargs):
        pytest.fail("Audit must not fit")

    for module, name in [
        (grouped_conditional, "minimize"),
        (grouped_gaussian, "minimize"),
        (integrative, "run_integrative"),
        (mixture, "select_gaussian_mixture"),
        (tabular_gmm, "fit_tabular_pca"),
    ]:
        monkeypatch.setattr(module, name, refuse)
    result = audit_grouped_study(root, output_dir=tmp_path / "audit")
    assert result["datasets"] == 7 and result["successful_fits"] == 14
    assert "PCA feature selection" in result["audit_scope"]
    assert before == {p: p.read_bytes() for p in before}
    # Bypass inventory checks deliberately: the numerical auditor must detect
    # changed parameters/projections rather than relying on their hashes alone.
    base = root / "correlated_gaussian/dataset_0000"
    c = study.calibration
    with np.load(base / "values.npz", allow_pickle=False) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    layout = read_json(base / "calibration/layout.json")
    with ZipFile(base / "calibration/pipeline_artifacts.zip") as archive:
        for corruption in ("feature_means", "components", "profiles", "counts"):

            class Corrupted:
                def read(self, name, corruption=corruption):
                    data = archive.read(name)
                    if name == "observed/run/view_000/parameters/pca.npz" and corruption in {
                        "feature_means",
                        "components",
                    }:
                        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                            p = {k: arrays[k] for k in arrays.files}
                        p[corruption] = p[corruption] + 0.1
                        buffer = io.BytesIO()
                        np.savez(buffer, **p)
                        return buffer.getvalue()
                    if (
                        name == "observed/run/view_000/parameters/profiles.npz"
                        and corruption == "profiles"
                    ):
                        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                            p = {k: arrays[k] for k in arrays.files}
                        p["profiles"] = p["profiles"] + 1
                        buffer = io.BytesIO()
                        np.savez(buffer, **p)
                        return buffer.getvalue()
                    if (
                        name == "observed/run/view_000/parameters/summary.json"
                        and corruption == "counts"
                    ):
                        p = json.loads(data)
                        p["diagnostics"]["pca_training_counts"][0] += 1
                        return json.dumps(p).encode()
                    return data

            with pytest.raises((AssertionError, ValueError)):
                audit_tabular_views(Corrupted(), "observed/run/", c.model, values, layout)


def test_full_component_covariances_have_independent_density_and_bic_audit(tmp_path):
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    from super_deli.benchmarking.pipeline_analysis import audit_pipeline_point

    c = policy(family="full")
    c = replace(c, n_simulations=1, model=replace(c.model, n_init=3))
    values, layout, _ = simulate_grouped("complementary", 128, 193)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    out = tmp_path / "calibration"
    result = run_grouped_calibration(dataset, config=c, base_dir=tmp_path / "raw", output_dir=out)
    assert result["successful_fits"] == 2 and result["point"]["selected_k"] == 2
    with ZipFile(out / "pipeline_artifacts.zip") as archive:
        name = "observed/run/mixture.npz"
        with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as arrays:
            model = {k: arrays[k] for k in arrays.files}
        assert not np.allclose(model["covariances"][0], model["covariances"][1])
        scores = model["scores"]
        components = np.column_stack(
            [
                np.log(w) + multivariate_normal.logpdf(scores, mean=m, cov=v)
                for w, m, v in zip(
                    model["weights"], model["means"], model["covariances"], strict=True
                )
            ]
        )
        log_density = logsumexp(components, axis=1)
        np.testing.assert_allclose(
            model["membership"], np.exp(components - log_density[:, None]), atol=1e-10
        )
        n, d = scores.shape
        k = len(model["weights"])
        # Count a separate symmetric covariance for every component directly.
        parameter_count = (k - 1) + k * d + k * d * (d + 1) // 2
        candidate = next(x for x in result["point"]["candidates"] if x["k"] == k)
        np.testing.assert_allclose(
            candidate["bic"], -2 * log_density.sum() + parameter_count * np.log(n), atol=1e-8
        )
        audit_pipeline_point(archive, result["point"], c)
        wrong = replace(c, model=replace(c.model, covariance_type="tied"))
        with pytest.raises(ValueError, match="configuration mismatch"):
            audit_pipeline_point(archive, result["point"], wrong)
        # Update both hashes deliberately so the independent density check,
        # rather than inventory integrity alone, must reject changed covariance.
        model["covariances"][0] *= 1.2
        handle = io.BytesIO()
        np.savez(handle, **model)
        raw = handle.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        record = {**result["point"], "mixture_sha256": digest}
        summary = json.loads(archive.read("observed/run/summary.json"))
        summary["provenance"]["mixture_sha256"] = digest

        class Corrupt:
            def read(self, path):
                if path == name:
                    return raw
                if path == "observed/run/summary.json":
                    return json.dumps(summary).encode()
                return archive.read(path)

        with pytest.raises(AssertionError):
            audit_pipeline_point(Corrupt(), record, c)
