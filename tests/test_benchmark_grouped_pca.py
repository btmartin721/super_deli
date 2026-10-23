"""Frozen grid, no-refit audit and retained failures for grouped PCA comparisons."""

# ruff: noqa: E402
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking import grouped_study
from super_deli.benchmarking.grouped_pca_analysis import audit_grouped_pca_study
from super_deli.benchmarking.grouped_pca_study import run_grouped_pca_study, simulate
from super_deli.benchmarking.grouped_recovery import run_grouped_recovery
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_config import GroupedStudyConfig
from super_deli.models.grouped_pca_study_config import (
    ARMS,
    COHORTS,
    PINS,
    GroupedPCAStudyConfig,
    study_model,
)
from super_deli.models.grouped_recovery_config import GroupedRecoveryConfig

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.baseline


@pytest.fixture
def source(tmp_path):
    study = GroupedStudyConfig.from_dict(
        read_json(ROOT / "examples/shared-grouped-calibration/smoke.json")
    )
    study = replace(
        study,
        n_units=24,
        calibration=replace(
            study.calibration,
            n_simulations=1,
            model=replace(study.calibration.model, n_init=10, max_k=4),
        ),
    )
    origin = tmp_path / "origin"
    (origin / "study").mkdir(parents=True)
    write_json(study.to_dict(), origin / "study.json")
    grouped_study.run_grouped_study(
        study, case="complementary", output_dir=origin / "study/complementary"
    )
    (origin / "study-audit").mkdir()
    write_json(inventory(origin / "study"), origin / "study-audit/files.json")
    modules = {
        "benchmarking/grouped_study.py": hashlib.sha256(
            Path(grouped_study.__file__).read_bytes()
        ).hexdigest()
    }
    write_json(modules, origin / "runtime_sources.json")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    diagnostics = GroupedRecoveryConfig(
        *(
            digest(origin / name)
            for name in ("study.json", "study-audit/files.json", "runtime_sources.json")
        ),
        n_datasets=1,
        small_n=24,
        large_n=32,
    )
    root = tmp_path / "source"
    root.mkdir()
    run_grouped_recovery(diagnostics, source_artifacts=origin, output_dir=root / "study")
    write_json(modules, root / "runtime_sources.json")
    config = GroupedPCAStudyConfig(
        model=study.calibration.model,
        **{field: digest(root / name) for field, name in PINS.items()},
        n_datasets=1,
        small_n=24,
        large_n=32,
        n_stability=1,
        n_bootstrap=2,
    )
    return root, config


def test_grid_preserves_all_other_settings(source):
    _, config = source
    for arm in ARMS:
        m = study_model(config, "archived", 0, arm)
        assert (
            m.n_bootstrap == 0
            and m.n_pcs == int(arm[-1])
            and m.views[0].pca_fit
            == ("target_means" if arm.startswith("target") else "source_rows")
        )
        assert (
            m.covariance_type == "tied"
            and m.min_cluster_size == 5
            and m.n_init == 10
            and m.max_k == 4
        )
        assert study_model(config, "fresh_64", 0, arm).n_bootstrap == 2
    for field, value in (
        ("schema_version", True),
        ("n_stability", 2),
        ("random_seed", -1),
        ("small_n", 25),
    ):
        with pytest.raises(ValueError):
            replace(config, **{field: value})


def test_end_to_end_audit_never_refits_and_source_is_preserved(source, tmp_path, monkeypatch):
    from super_deli.models import mixture

    root, config = source
    before = inventory(root)
    result = run_grouped_pca_study(
        config, source_artifacts=root, output_dir=tmp_path / "study", workers=1
    )
    assert (
        result["successful_points"] == 28
        and result["successful_bootstrap"] == 16
        and result["reproduced_archived"] == 1
    )
    assert inventory(root) == before
    monkeypatch.setattr(
        mixture, "select_gaussian_mixture", lambda *a, **kw: pytest.fail("Audit must not fit")
    )
    audited = audit_grouped_pca_study(tmp_path / "study", output_dir=tmp_path / "audit")
    assert audited["successful_points"] == 28 and len(audited["summaries"]) == 28
    assert len(audited["paired"]) == 28 and audited["significance_fits"] == 0
    with pytest.raises(FileExistsError):
        run_grouped_pca_study(config, source_artifacts=root, output_dir=tmp_path / "study")
    (tmp_path / "study/points.json").write_text("[]\n")
    with pytest.raises(ValueError, match="inventory"):
        audit_grouped_pca_study(tmp_path / "study", output_dir=tmp_path / "bad")


def test_failed_point_and_bootstrap_are_not_replaced(source, tmp_path, monkeypatch):
    from super_deli.models import integrative
    from super_deli.workflows import integrative as workflow

    root, config = source
    original = workflow.run_integrative
    calls = []

    def fail_point(*a, **kw):
        calls.append(kw["model_config"])
        if len(calls) == 6:
            raise ValueError("Injected point failure")
        return original(*a, **kw)

    monkeypatch.setattr(workflow, "run_integrative", fail_point)
    fitter = integrative._fit_once
    injected = []

    def fail_bootstrap(inputs, units, model, seed, draw):
        if (
            model.views[0].pca_fit == "target_means"
            and model.n_bootstrap
            and not np.array_equal(draw, np.arange(len(units)))
            and not injected
        ):
            injected.append(None)
            raise ValueError("Injected bootstrap failure")
        return fitter(inputs, units, model, seed, draw)

    monkeypatch.setattr(integrative, "_fit_once", fail_bootstrap)
    result = run_grouped_pca_study(config, source_artifacts=root, output_dir=tmp_path / "study")
    assert len(calls) == 28 and len(injected) == 1 and result["failed_points"] == 1
    assert result["unattempted_bootstrap"] == 2 and result["failed_bootstrap"] == 1
    audited = audit_grouped_pca_study(tmp_path / "study", output_dir=tmp_path / "audit")
    assert audited["successful_points"] == 27 and audited["failed_bootstrap"] == 1


def test_source_corruption_is_rejected_before_fresh_data(source, tmp_path):
    root, config = source
    (root / "study/archived/dataset_0000/raw/view_000.csv").write_text("altered\n")
    with pytest.raises(ValueError, match="inventory"):
        run_grouped_pca_study(config, source_artifacts=root, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_fresh_generators_are_reproducible_with_explicit_truth_scope():
    for cohort in COHORTS:
        a = simulate(cohort, 64, 810, target_level="population")
        b = simulate(cohort, 64, 810, target_level="population")
        for key in a[0]:
            np.testing.assert_array_equal(a[0][key], b[0][key])
        assert a[1]["counts"] == b[1]["counts"] and sum(a[1]["counts"]) == len(
            a[0]["morphological"]
        )
        np.testing.assert_array_equal(a[2], b[2])
        assert len(np.unique(a[2])) == (
            1 if cohort in {"independent_gaussian", "correlated_gaussian", "cline"} else 4
        )
