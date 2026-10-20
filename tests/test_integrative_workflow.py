"""Real SNPio integration, parameter reconstruction, source alignment and CLI artifacts."""

# ruff: noqa: E402

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from super_deli import load_config
from super_deli.cli import main
from super_deli.core.data import Specimen
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json
from super_deli.io.shape_model import read_shape_model
from super_deli.models import integrative
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.shape_pca import project_shape_pca
from super_deli.workflows.integrative import run_integrative

pytestmark = pytest.mark.baseline
EXAMPLES = Path(__file__).resolve().parents[1] / "examples/integrative"


def snapshot(directory):
    return {
        str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    shutil.copytree(EXAMPLES, raw)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return raw


def configuration(raw, mode="matched", genomic=False, **changes):
    config = IntegrativeConfig.from_dict(read_json(raw / f"{mode}_model.json"))
    if not genomic:
        config = replace(config, views=tuple(v for v in config.views if v.kind != "genomic"))
    return replace(config, **changes)


def run(raw, output, mode="matched", genomic=False, **changes):
    if genomic:
        pytest.importorskip("snpio")
    return run_integrative(
        load_config(raw / f"{mode}_dataset.json"),
        model_config=configuration(raw, mode, genomic, **changes),
        base_dir=raw,
        output_dir=output,
    )


def reconstruct_profiles(output, i, summary):
    directory = output / f"view_{i:03d}"
    raw = np.load(directory / "source_values.npy", allow_pickle=False)
    rows = read_json(directory / "source_rows.json")
    params = directory / "parameters"
    if summary["kind"] == "morphometric":
        model, _, _ = read_shape_model(params)
        scores = project_shape_pca(raw, model=model).scores
    else:
        with np.load(params / "pca.npz", allow_pickle=False) as p:
            matrix = raw[:, p["kept_indices"]]
            matrix = np.where(np.isnan(matrix), p["feature_means"], matrix)
            scores = ((matrix - p["feature_means"]) / p["feature_scales"] - p["mean"]) @ p[
                "components"
            ].T
    profiles = np.stack(
        [
            scores[
                [j for j, target in enumerate(rows["target_ids"]) if target == member["target_id"]]
            ].mean(axis=0)
            for member in summary["members"]
        ]
    )
    with np.load(params / "profiles.npz", allow_pickle=False) as p:
        np.testing.assert_allclose(profiles, p["profiles"], atol=1e-9)
        return (profiles - p["mean"]) / p["scale"] * np.sqrt(p["normalized_weight"])


@pytest.mark.parametrize(
    "mode,genomic", [("matched", False), ("grouped", False), ("matched", True), ("grouped", True)]
)
def test_four_group_truth_reconstruction_and_source_safety(source, tmp_path, mode, genomic):
    if genomic:
        pytest.importorskip("snpio")
    before = snapshot(source)
    output = tmp_path / "fit"
    summary = run(source, output, mode, genomic, n_bootstrap=3)
    assert summary["selected_k"] == 4 and summary["bootstrap"]["successful"] == 3
    assert summary["level"] == ("locality" if mode == "grouped" else "specimen")
    assert len(summary["unit_ids"]) == 64
    assert len(summary["excluded_units"]) == (64 if mode == "matched" else 0)
    blocks = [reconstruct_profiles(output, i, view) for i, view in enumerate(summary["views"])]
    fused = np.concatenate(blocks, axis=1)
    with np.load(output / "fusion.npz", allow_pickle=False) as p:
        np.testing.assert_allclose(fused, p["fused"], atol=1e-9)
        projected = (fused - p["mean"]) @ p["components"].T
    with np.load(output / "mixture.npz", allow_pickle=False) as p:
        np.testing.assert_allclose(projected, p["scores"], atol=1e-9)
        model = GaussianMixture(n_components=4, covariance_type="full")
        model.weights_, model.means_ = p["weights"], p["means"]
        model.covariances_, model.precisions_cholesky_ = p["covariances"], p["precisions_cholesky"]
        np.testing.assert_allclose(model.predict_proba(projected), p["membership"], atol=1e-9)
        truth = [u["latent_group"] for u in read_json(source / "truth.json")["units"]]
        assert adjusted_rand_score(truth, p["labels"]) == 1
    with (output / "assignments.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["unit_id"] for r in rows] == summary["unit_ids"]
    with (output / "scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    np.testing.assert_allclose(
        [[float(r["PC1"]), float(r["PC2"])] for r in rows], projected, atol=1e-9
    )
    provenance = summary["provenance"]
    assert provenance["configuration_sha256"] == configuration_hash(provenance["configuration"])
    assert provenance["fusion_sha256"] == fingerprint_file(output / "fusion.npz").sha256
    assert all(v["normalized_weight"] == pytest.approx(1 / len(blocks)) for v in summary["views"])
    result = read_json(output / "result.json")
    assert result["level"] == summary["level"]
    assert next(e for e in result["evidence"] if e["kind"] == "p_value")["status"] == "not_computed"
    assert not list(output.rglob("truth.json"))
    if not genomic:
        assert not list(output.rglob("genotypes.phy"))
    if mode == "matched":
        # Original matrices contain 128 rows; retained point-fit rows are even.
        first = summary["views"][0]
        assert first["retained_source_indices"] == list(range(0, 128, 2))
        assert first["training_source_indices"] == list(range(0, 128, 2))
    else:
        for view in summary["views"]:
            expected = 1 if view["kind"] == "ecological" else 2
            assert all(len(m["source_unit_ids"]) == expected for m in view["members"])
    assert snapshot(source) == before and not list(Path.cwd().iterdir())
    existing = snapshot(output)
    with pytest.raises(FileExistsError):
        run(source, output, mode, genomic, n_bootstrap=0)
    assert snapshot(output) == existing


def test_no_downward_ecology_broadcast_or_unlinked_groups(source, tmp_path):
    dataset = load_config(source / "grouped_dataset.json")
    config = configuration(source, "grouped", False)
    config = replace(
        config,
        target_level="specimen",
        views=tuple(replace(v, aggregation="none") for v in config.views),
    )
    with pytest.raises(ValueError, match="downward"):
        run_integrative(dataset, model_config=config, base_dir=source, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    specs = tuple(replace(s, locality_id=None) for s in dataset.specimens)
    with pytest.raises(ValueError, match="locality_id"):
        run_integrative(
            replace(dataset, specimens=specs),
            model_config=configuration(source, "grouped", False),
            base_dir=source,
            output_dir=tmp_path / "unlinked",
        )


def test_cli_json_without_genomic_extra_and_cwd_independence(source, tmp_path, capsys):
    model = configuration(source, n_bootstrap=2)
    (source / "nongeneric.json").write_text(json.dumps(model.to_dict()))
    args = [
        "cluster-integrative",
        str(source / "matched_dataset.json"),
        "--model-config",
        str(source / "nongeneric.json"),
        "--output-dir",
        str(tmp_path / "cli"),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["selected_k"] == 4
    result = subprocess.run(
        [sys.executable, "-m", "super_deli", *args[:-1], str(tmp_path / "outside")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "complete"


def test_failed_bootstrap_retains_point_result_and_requested_denominator(
    source, tmp_path, monkeypatch
):
    original, calls = integrative._fit_once, 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("deliberate failure")
        return original(*args)

    monkeypatch.setattr(integrative, "_fit_once", fail)
    output = tmp_path / "partial"
    summary = run(source, output, n_bootstrap=3)
    assert summary["status"] == "complete" and summary["bootstrap"]["status"] == "unavailable"
    assert summary["bootstrap"]["successful"] == 2 and calls == 4
    for cluster in read_json(output / "clusters.json"):
        assert cluster["frequency"] is None
        assert cluster["failure_upper"] - cluster["failure_lower"] == pytest.approx(1 / 3)
    assert not (output / "failure.json").exists()


def test_disabled_stability_and_point_failure_are_explicit(source, tmp_path):
    output = tmp_path / "disabled"
    summary = run(source, output, n_bootstrap=0, assignment_threshold=1)
    assert summary["bootstrap"]["status"] == "not_computed"
    assert not (output / "stability.npz").exists()
    config = configuration(source, n_bootstrap=0)
    config = replace(config, views=(replace(config.views[0], n_pcs=4), *config.views[1:]))
    with pytest.raises(ValueError, match="Too few variable"):
        run_integrative(
            load_config(source / "matched_dataset.json"),
            model_config=config,
            base_dir=source,
            output_dir=tmp_path / "rank",
        )
    assert (tmp_path / "rank/failure.json").exists()
    assert not (tmp_path / "rank/summary.json").exists()


def test_shape_missing_observation_policy_is_auditable(source, tmp_path):
    lines = (source / "landmarks.csv").read_text().splitlines()
    fields = lines[1].split(",")
    fields[2] = ""
    lines[1] = ",".join(fields)
    (source / "landmarks.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        run(source, tmp_path / "missing", n_bootstrap=0)
    config = configuration(source, n_bootstrap=0)
    views = tuple(
        replace(v, shape=replace(v.shape, missing_policy="drop_observations"))
        if v.kind == "morphometric"
        else v
        for v in config.views
    )
    summary = run_integrative(
        load_config(source / "matched_dataset.json"),
        model_config=replace(config, views=views),
        base_dir=source,
        output_dir=tmp_path / "drop",
    )
    assert len(summary["unit_ids"]) == 63
    assert len(summary["excluded_observations"]) == 1
    assert len(summary["excluded_units"]) == 65


def test_declared_targets_without_any_view_are_reported(source, tmp_path):
    dataset = load_config(source / "matched_dataset.json")
    dataset = replace(dataset, specimens=(*dataset.specimens, Specimen("unobserved")))
    summary = run_integrative(
        dataset,
        model_config=configuration(source, n_bootstrap=0),
        base_dir=source,
        output_dir=tmp_path / "unobserved",
    )
    assert summary["unobserved_target_units"] == ["unobserved"]
    assert len(summary["unit_ids"]) == 64
