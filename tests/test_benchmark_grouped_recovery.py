"""Point-grid, truth separation, failure accounting and independent audit checks."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_config import GroupedStudyConfig
from super_deli.models.grouped_recovery_config import GroupedRecoveryConfig, variants

ROOT = Path(__file__).resolve().parents[1]


def config(**kwargs):
    return GroupedRecoveryConfig("a" * 64, "b" * 64, "c" * 64, **kwargs)


@pytest.mark.parametrize(
    "options",
    [
        {"n_datasets": True},
        {"small_n": 25},
        {"large_n": 24},
        {"random_seed": -1},
        {"schema_version": True},
    ],
)
def test_invalid_design(options):
    with pytest.raises(ValueError):
        config(**options)


def test_grid_changes_only_declared_controls():
    model = GroupedStudyConfig.from_dict(
        read_json(ROOT / "examples/shared-grouped-calibration/study.json")
    ).calibration.model
    grid = variants(model)
    assert list(grid) == [
        "baseline",
        "pcs_3",
        "pcs_4",
        "ecology_0.5",
        "ecology_2",
        "ecology_4",
        "starts_100",
        "views_4_joint_2",
        "views_4_joint_4",
        "first_specimen",
    ]
    assert grid["baseline"] == model
    assert grid["first_specimen"] == model
    assert grid["views_4_joint_4"] == replace(
        model, n_pcs=4, views=tuple(replace(v, n_pcs=4) for v in model.views)
    )
    assert all(m.covariance_type == "tied" for m in grid.values())
    assert grid["pcs_4"] == replace(model, n_pcs=4)
    assert grid["starts_100"] == replace(model, n_init=100)
    for weight in (0.5, 2, 4):
        assert grid[f"ecology_{weight}"] == replace(
            model,
            views=tuple(
                replace(v, weight=weight) if v.kind == "ecological" else v for v in model.views
            ),
        )
    assert all(m.n_bootstrap == 0 and m.min_cluster_size == 5 for m in grid.values())
    with pytest.raises(ValueError, match="baseline"):
        variants(replace(model, n_pcs=3))


def test_dimension_specific_bic_decomposition():
    np = pytest.importorskip("numpy")
    from super_deli.benchmarking.recovery_metrics import bic_terms

    candidates = [
        {"k": 2, "bic": 10, "status": "eligible"},
        {"k": 4, "bic": 20, "status": "small_component"},
    ]
    rows = bic_terms(candidates, 2, 64, 2)
    assert [r["parameters"] for r in rows] == [11, 23]
    assert rows[1]["penalty_minus_selected"] == pytest.approx(12 * np.log(64))
    assert rows[1]["bic_minus_selected"] == pytest.approx(
        rows[1]["penalty_minus_selected"] - rows[1]["twice_log_likelihood_gain"]
    )
    assert bic_terms(candidates, 2, 64, 4)[1]["parameters"] == 59


SCIENTIFIC = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="Install clustering extra",
)


@pytest.fixture
def source(tmp_path):
    import super_deli.benchmarking.grouped_study as generator
    from super_deli.benchmarking.grouped_study import run_grouped_study
    from super_deli.benchmarking.recovery_inputs import inventory

    study = GroupedStudyConfig.from_dict(
        read_json(ROOT / "examples/shared-grouped-calibration/smoke.json")
    )
    study = replace(
        study,
        n_units=24,
        calibration=replace(
            study.calibration,
            n_simulations=1,
            model=replace(study.calibration.model, max_k=4, n_init=10),
        ),
    )
    root = tmp_path / "source"
    (root / "study").mkdir(parents=True)
    sparse = study.to_dict()
    sparse["calibration"]["model"].pop("tol")
    write_json(sparse, root / "study.json")
    run_grouped_study(study, case="complementary", output_dir=root / "study/complementary")
    (root / "study-audit").mkdir()
    write_json(inventory(root / "study"), root / "study-audit/files.json")
    write_json(
        {
            "super_deli/benchmarking/grouped_study.py": hashlib.sha256(
                Path(generator.__file__).read_bytes()
            ).hexdigest()
        },
        root / "runtime_sources.json",
    )
    digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()  # noqa: E731
    declaration = GroupedRecoveryConfig(
        digest("study.json"),
        digest("study-audit/files.json"),
        digest("runtime_sources.json"),
        n_datasets=1,
        small_n=24,
        large_n=32,
    )
    return root, declaration


@SCIENTIFIC
@pytest.mark.baseline
def test_end_to_end_and_no_refit_audit(source, tmp_path, monkeypatch):
    from super_deli.benchmarking.grouped_recovery import run_grouped_recovery
    from super_deli.benchmarking.grouped_recovery_analysis import audit_grouped_recovery
    from super_deli.benchmarking.recovery_inputs import inventory
    from super_deli.cli import main
    from super_deli.models import mixture

    source_root, declaration = source
    before = inventory(source_root)
    output = tmp_path / "diagnostics"
    result = run_grouped_recovery(declaration, source_artifacts=source_root, output_dir=output)
    assert result["successful_points"] == 30 and result["successful_truth_initialized"] == 3
    assert result["replayed_archived"] == 1 and result["bootstrap_or_calibration_fits"] == 0
    assert inventory(source_root) == before
    assert len(read_json(output / "points.json")) == 30
    assert "labels" not in json.dumps(read_json(output / "fresh_64/dataset_0000/raw/dataset.json"))
    monkeypatch.setattr(
        mixture, "select_gaussian_mixture", lambda *a, **kw: pytest.fail("Audit must not refit")
    )
    result = audit_grouped_recovery(output, output_dir=tmp_path / "audit")
    assert result["successful_points"] == 30 and len(result["summaries"]) == 30
    assert (
        main(
            [
                "analyze-grouped-recovery",
                str(output),
                "--output-dir",
                str(tmp_path / "cli-audit"),
            ]
        )
        == 0
    )
    with pytest.raises(FileExistsError):
        run_grouped_recovery(declaration, source_artifacts=source_root, output_dir=output)
    (output / "points.json").write_text("[]\n")
    with pytest.raises(ValueError, match="inventory"):
        audit_grouped_recovery(output, output_dir=tmp_path / "corrupt")


@SCIENTIFIC
@pytest.mark.baseline
def test_source_corruption_is_rejected_before_fresh_data(source, tmp_path):
    from super_deli.benchmarking.grouped_recovery import run_grouped_recovery

    root, declaration = source
    path = root / "study/complementary/dataset_0000/observed/view_000.csv"
    path.write_text(path.read_text() + "\n")
    output = tmp_path / "bad-source"
    with pytest.raises(ValueError, match="source file changed"):
        run_grouped_recovery(declaration, source_artifacts=root, output_dir=output)
    assert not (output / "fresh_64/dataset_0000").exists()


@SCIENTIFIC
@pytest.mark.baseline
@pytest.mark.parametrize("failure_call", [2, 11])
def test_one_failed_variant_is_not_replaced(source, tmp_path, monkeypatch, failure_call):
    from super_deli.benchmarking.grouped_recovery import run_grouped_recovery
    from super_deli.benchmarking.grouped_recovery_analysis import audit_grouped_recovery
    from super_deli.workflows import integrative

    original = integrative.run_integrative
    calls = []

    def run(*args, **kwargs):
        calls.append(None)
        if len(calls) == failure_call:
            raise ValueError("Injected diagnostic failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", run)
    root, declaration = source
    output = tmp_path / "failure"
    result = run_grouped_recovery(declaration, source_artifacts=root, output_dir=output)
    assert len(calls) == 30 and result["failed_points"] == 1
    assert result["successful_points"] == 29
    assert result["successful_truth_initialized"] == (2 if failure_call == 11 else 3)
    assert result["unattempted_truth_initialized"] == (1 if failure_call == 11 else 0)
    audited = audit_grouped_recovery(output, output_dir=tmp_path / "audit-failure")
    assert audited["failed_points"] == 1


def test_first_control_retains_original_identities_and_no_truth():
    np = pytest.importorskip("numpy")
    from super_deli.benchmarking.grouped_recovery_inputs import first_specimen
    from super_deli.benchmarking.grouped_study import simulate_grouped

    values, layout, labels = simulate_grouped("complementary", 64, 299)
    original = json.dumps(layout)
    control, design = first_specimen(values, layout)
    indices = np.cumsum([0, *layout["counts"][:-1]])
    np.testing.assert_array_equal(control["morphological"], values["morphological"][indices])
    np.testing.assert_array_equal(control["ecological"], values["ecological"])
    assert design["counts"] == [1] * 64 and json.dumps(layout) == original
    assert design["views"][0]["unit_ids"] == [layout["views"][0]["unit_ids"][i] for i in indices]
    assert "labels" not in json.dumps(design) and len(labels) == 64


@SCIENTIFIC
@pytest.mark.baseline
def test_audit_detects_tampered_source_pca_even_after_inventory_update(source, tmp_path):
    np = pytest.importorskip("numpy")
    from super_deli.benchmarking.grouped_recovery import run_grouped_recovery
    from super_deli.benchmarking.grouped_recovery_analysis import audit_grouped_recovery
    from super_deli.benchmarking.recovery_inputs import inventory

    root, declaration = source
    output = tmp_path / "tampered"
    run_grouped_recovery(declaration, source_artifacts=root, output_dir=output)
    path = output / "fresh_64/dataset_0000/baseline/run/view_000/parameters/pca.npz"
    with np.load(path, allow_pickle=False) as arrays:
        data = {k: arrays[k] for k in arrays.files}
    data["explained_variance"] = data["explained_variance"] + 0.1
    np.savez(path, **data)
    files = inventory(output)
    files.pop("files.json")
    (output / "files.json").write_text(json.dumps(files, indent=2) + "\n")
    with pytest.raises((AssertionError, ValueError)):
        audit_grouped_recovery(output, output_dir=tmp_path / "bad-audit")
