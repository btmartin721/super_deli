"""Fit-free algebra, leakage/identity guards and explicit development decisions."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


def fixture(path):
    source = Path(__file__).resolve().parents[1] / "examples/genomic-control-diagnosis/synthetic.py"
    spec = importlib.util.spec_from_file_location("invented_genomic_control", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture(path)


def test_no_fit_workflow_and_derivatives(tmp_path, monkeypatch):
    import builtins

    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    from super_deli.benchmarking.genomic_diagnosis import run_genomic_diagnosis

    root = tmp_path / "source"
    config = fixture(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("A fit-free workflow called a training/random API")

    monkeypatch.setattr(PCA, "fit", forbidden)
    monkeypatch.setattr(PCA, "fit_transform", forbidden)
    monkeypatch.setattr(GaussianMixture, "fit", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.split(".")[0] == "torch":
            raise AssertionError("Fit-free diagnosis imported Torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    result = run_genomic_diagnosis(root, config=config, output_dir=tmp_path / "out")
    assert (
        result["point_models"] == result["subsample_models"] == 5
        and result["archived_neural_maps"] == 6
    )
    assert (
        result["new_training_fits"] == result["new_gmm_searches"] == result["new_random_draws"] == 0
    )
    assert result["decision"]["status"] == "invented_fixture_no_scientific_decision"
    records = list((tmp_path / "out/records").glob("*.json"))
    assert len(records) == 10
    neural = [json.loads(p.read_text()) for p in records if "neural" in p.name]
    assert all(r["jacobian_maximum_error"] < 2e-7 for r in neural)
    with pytest.raises(FileExistsError):
        run_genomic_diagnosis(root, config=config, output_dir=tmp_path / "out")


@pytest.mark.parametrize("failure", ["corrupt", "receipt", "missing", "cap", "selection"])
def test_withheld_on_invalid_source(tmp_path, failure):
    from super_deli.benchmarking.genomic_diagnosis import run_genomic_diagnosis

    root = tmp_path / "source"
    config = fixture(root)
    extra = {}
    if failure == "corrupt":
        p = root / "verification-evidence/main/invented-main/analysis/inputs/dosages.npy"
        p.write_bytes(p.read_bytes() + b"bad")
    elif failure == "receipt":
        (root / "completion.json").write_text("{}")
    elif failure == "missing":
        (root / "verification-evidence/study-freeze.json").unlink()
    elif failure == "cap":
        config = replace(config, max_output_bytes=1)
    else:
        extra["case_ids"] = ["unknown"]
    with pytest.raises((ValueError, FileNotFoundError)):
        run_genomic_diagnosis(root, config=config, output_dir=tmp_path / "out", **extra)
    result = json.loads((tmp_path / "out/failure.json").read_text())
    assert result["status"] == "withheld" and result["decision"] == "unavailable"
    assert not (tmp_path / "out/completion.json").exists()


def test_no_path_escape(tmp_path):
    from super_deli.benchmarking.genomic_diagnosis_archive import Archive

    root = tmp_path / "source"
    source = Archive(root, fixture(root))
    with pytest.raises(ValueError):
        source.path("../completion.json")
    with pytest.raises(ValueError):
        source.path("/etc/passwd")


def test_costs_and_rank_correlation():
    from super_deli.benchmarking.genomic_diagnosis_math import candidate_costs, correlation

    records = [
        dict(k=1, status="eligible", bic=100.0, converged=True, hard_cluster_sizes=[20]),
        dict(k=2, status="eligible", bic=90.0, converged=True, hard_cluster_sizes=[10, 10]),
    ]
    rows = candidate_costs(records, 20, False, 3)
    np.testing.assert_allclose(rows[1]["parameter_cost"], 3 * np.log(20))
    np.testing.assert_allclose(rows[1]["twice_likelihood_gain"], 10 + 3 * np.log(20))
    assert correlation(np.ones(10), np.arange(10)) is None
    np.testing.assert_allclose(correlation(np.arange(10), -np.arange(10)), -1.0, atol=1e-15)


def test_label_invariant_agreement():
    from super_deli.benchmarking.genomic_diagnosis_summary import adjusted_rand

    assert adjusted_rand(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == 1
    np.testing.assert_allclose(adjusted_rand(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])), -0.5)


def test_development_decision_uses_paired_evidence():
    from super_deli.benchmarking.genomic_diagnosis_summary import summarize

    template = dict(
        phase="main",
        draw=None,
        n=64,
        selected_bic_improvement=0.0,
        twice_likelihood_gain=0.0,
        parameter_cost=0.0,
        rare_sensitivity_share=0.2,
        hidden_saturation=[],
        burden_associations={"energy": dict(radius_spearman=0.1, label_variance_fraction=0.2)},
    )
    rows = []
    for law in ("common_hwe", "four_groups"):
        for arm in ("pca_shared", "neural_seed_0", "neural_seed_1", "neural_seed_2"):
            neural = arm.startswith("neural")
            rows.append(
                dict(
                    template,
                    case_id=law,
                    law=law,
                    arm=arm,
                    family="categorical_ae" if neural else "pca_shared",
                    selected_k=4 if law == "four_groups" else 2 if neural else 1,
                    ari=1.0 if law == "four_groups" else None,
                )
            )
    result = summarize(rows, "completed_study")
    assert (
        result["decision"]["status"] == "no_go_for_promotion_unguided_tuning_or_large_calibration"
    )
    assert result["primary_independent_datasets"] == 2
    assert all(
        r["control_datasets"] == 1
        and r["neural_only_control_splits"] == 1
        and r["neural_ari_better"] == 0
        for r in result["paired_main_comparisons"]
    )
    for row in rows:
        if row["law"] == "four_groups" and row["arm"] == "pca_shared":
            row["ari"] = 0.5
    assert (
        summarize(rows, "completed_study")["decision"]["status"]
        == "insufficient_basis_for_promotion_review_before_new_work"
    )


def test_coherently_resealed_wrong_projection_withholds(tmp_path):
    from super_deli.benchmarking.genomic_diagnosis import run_genomic_diagnosis
    from super_deli.benchmarking.genomic_diagnosis_archive import digest

    root = tmp_path / "source"
    config = fixture(root)
    base = root / "verification-evidence"
    path = base / "main/invented-main/analysis/arms/neural_seed_0/model.npz"
    with np.load(path) as source:
        arrays = {k: source[k] for k in source.files}
    arrays["scores"] = arrays["scores"] + 0.2
    np.savez_compressed(path, **arrays)
    meta = path.with_name("model.json")
    document = json.loads(meta.read_text())
    document["model_sha256"] = digest(path)
    meta.write_text(json.dumps(document))
    manifest = root / "verification-manifest.json"
    inventory = json.loads(manifest.read_text())
    for p in (path, meta):
        inventory[p.relative_to(base).as_posix()] = dict(bytes=p.stat().st_size, sha256=digest(p))
    manifest.write_text(json.dumps(inventory))
    config = replace(config, source_manifest_sha256=digest(manifest))
    with pytest.raises(ValueError, match="withheld"):
        run_genomic_diagnosis(root, config=config, output_dir=tmp_path / "out")
    assert json.loads((tmp_path / "out/failure.json").read_text())["decision"] == "unavailable"
