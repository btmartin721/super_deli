"""CLI, canonical parameter export, input safety, failure artifacts and evidence semantics."""

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
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.results import DelimitationResult
from super_deli.io.config import read_json
from super_deli.io.shape_model import read_shape_model
from super_deli.models import shape_gmm
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.models.shape_pca import project_shape_pca
from super_deli.workflows.shape_gmm import run_shape_gmm

pytestmark = pytest.mark.baseline
EXAMPLES = Path(__file__).resolve().parents[1] / "examples/shape/clustering"


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
    return raw, load_config(raw / "dataset.json")


def run(source, output, **changes):
    raw, dataset = source
    config = replace(ShapeClusterConfig.from_dict(read_json(raw / "model.json")), **changes)
    return run_shape_gmm(dataset, model_config=config, base_dir=raw, output_dir=output)


def test_fixture_truth_parameters_evidence_and_source_safety(source, tmp_path):
    raw, _ = source
    before = snapshot(raw)
    output = tmp_path / "fit"
    summary = run(source, output, n_bootstrap=5)
    assert summary["selected_k"] == 2
    assert summary["bootstrap"]["status"] == "available"
    truth = [u["latent_group"] for u in read_json(raw / "truth.json")["units"]]
    with np.load(output / "mixture.npz", allow_pickle=False) as arrays:
        assert adjusted_rand_score(truth, arrays["labels"]) == 1
        # Reconstruct from canonical saved numerical parameters, without pickle.
        model = GaussianMixture(n_components=2, covariance_type="full")
        model.weights_, model.means_ = arrays["weights"], arrays["means"]
        model.covariances_ = arrays["covariances"]
        model.precisions_cholesky_ = arrays["precisions_cholesky"]
        np.testing.assert_allclose(model.predict_proba(arrays["scores"]), arrays["membership"])
        with (output / "assignments.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [r["unit_id"] for r in rows] == summary["unit_ids"]
        np.testing.assert_allclose(
            [float(r["membership_cluster_1"]) for r in rows], arrays["membership"][:, 0]
        )
    point_shape, metadata, _ = read_shape_model(output / "shape")
    assert point_shape.config.n_components == 2
    assert metadata["training_unit_ids"] == summary["unit_ids"]
    p = summary["provenance"]
    assert p["configuration_sha256"] == configuration_hash(p["configuration"])
    assert p["mixture_array_sha256"] == fingerprint_file(output / "mixture.npz").sha256
    evidence = read_json(output / "result.json")["evidence"]
    assert {e["kind"] for e in evidence} == {
        "membership_probability",
        "bootstrap_frequency",
        "p_value",
    }
    assert next(e for e in evidence if e["kind"] == "p_value")["status"] == "not_computed"
    clusters = read_json(output / "clusters.json")
    assert all(c["failure_lower"] == c["frequency"] == c["failure_upper"] for c in clusters)
    assert snapshot(raw) == before and not list(Path.cwd().iterdir())
    assert not list(output.rglob("truth.json"))
    assert summary["model"] == "shape_pca_full_gmm"
    assert len(read_json(output / "bootstrap_replicates.json")) == 5
    before_output = snapshot(output)
    with pytest.raises(FileExistsError):
        run(source, output)
    assert snapshot(output) == before_output


def test_cli_json_and_working_directory_independence(source, tmp_path, capsys):
    raw, _ = source
    args = [
        "cluster-shape",
        str(raw / "dataset.json"),
        "--model-config",
        str(raw / "model.json"),
        "--output-dir",
        str(tmp_path / "cli"),
    ]
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["selected_k"] == 2 and summary["bootstrap"]["requested"] == 20
    command = subprocess.run(
        [sys.executable, "-m", "super_deli", *args[:-1], str(tmp_path / "subprocess")],
        capture_output=True,
        text=True,
    )
    assert command.returncode == 0, command.stderr
    assert json.loads(command.stdout)["status"] == "complete"


def test_tps_and_csv_have_equivalent_memberships(source, tmp_path):
    raw, dataset = source
    original = run(source, tmp_path / "csv", n_bootstrap=0)
    with (raw / "coordinates.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    lines = []
    for unit in original["unit_ids"]:
        points = [row for row in rows if row["observation_id"] == unit]
        lines += ["LM=6", *(f"{row['x']} {row['y']}" for row in points), f"ID={unit}"]
    (raw / "coordinates.tps").write_text("\n".join(lines) + "\n")
    modality = dataset.modalities[0]
    tps = replace(dataset, modalities=(replace(modality, format="tps", path="coordinates.tps"),))
    run_shape_gmm(
        tps,
        model_config=ShapeClusterConfig(True, n_bootstrap=0, max_k=3, random_seed=71),
        base_dir=raw,
        output_dir=tmp_path / "tps",
    )
    with np.load(tmp_path / "csv/mixture.npz") as a, np.load(tmp_path / "tps/mixture.npz") as b:
        np.testing.assert_allclose(a["membership"], b["membership"], atol=1e-8)


def test_disabled_bootstrap_and_membership_threshold(source, tmp_path, monkeypatch):
    # Capture the validated result contract directly to check unresolved assignments.
    import super_deli.workflows.shape_gmm as workflow

    captured = []

    def capture(*args, **kwargs):
        result = DelimitationResult(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(workflow, "DelimitationResult", capture)
    summary = run(source, tmp_path / "disabled", n_bootstrap=0, assignment_threshold=1)
    assert summary["bootstrap"]["status"] == "not_computed"
    assert not (tmp_path / "disabled/stability.npz").exists()
    evidence = captured[0].evidence
    assert all(e.status == "not_computed" for e in evidence if e.kind == "bootstrap_frequency")
    # The threshold is applied to membership; it never modifies candidate labels.
    for a, e in zip(captured[0].assignments, evidence[:64], strict=True):
        assert (a.label is not None) == (e.value >= 1)


def test_partial_bootstrap_reports_point_result_and_bounds(source, tmp_path, monkeypatch):
    original = shape_gmm.fit_shape_pca
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:  # workflow point fit is reused, only replicates call this function
            raise ValueError("deliberate failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(shape_gmm, "fit_shape_pca", fail_once)
    summary = run(source, tmp_path / "partial", n_bootstrap=3)
    assert summary["status"] == "complete" and summary["bootstrap"]["successful"] == 2
    assert summary["bootstrap"]["status"] == "unavailable"
    result = read_json(tmp_path / "partial/result.json")
    assert all(e["value"] is None for e in result["evidence"] if e["kind"] == "bootstrap_frequency")
    for c in read_json(tmp_path / "partial/clusters.json"):
        assert c["frequency"] is None
        assert c["failure_upper"] - c["failure_lower"] == pytest.approx(1 / 3)
    assert not (tmp_path / "partial/failure.json").exists()


def test_rank_failure_never_writes_completion_summary(source, tmp_path):
    output = tmp_path / "rank_failure"
    with pytest.raises(ValueError, match="rank deficient"):
        run(source, output, n_pcs=10, min_cluster_size=11)
    assert (output / "failure.json").exists()
    assert not (output / "summary.json").exists()


def test_repeat_units_and_missing_landmarks_follow_explicit_policies(source, tmp_path):
    raw, dataset = source
    modality = dataset.modalities[0]
    repeated = replace(modality.observations[1], unit_id=modality.observations[0].unit_id)
    repeat_dataset = replace(
        dataset,
        modalities=(
            replace(
                modality,
                observations=(modality.observations[0], repeated, *modality.observations[2:]),
            ),
        ),
    )
    with pytest.raises(ValueError, match="Repeated"):
        run((raw, repeat_dataset), tmp_path / "repeats", n_bootstrap=0)
    contents = (raw / "coordinates.csv").read_text().splitlines()
    fields = contents[1].split(",")
    fields[2] = ""
    contents[1] = ",".join(fields)
    (raw / "coordinates.csv").write_text("\n".join(contents) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        run(source, tmp_path / "missing", n_bootstrap=0)
    config = ShapeClusterConfig(True, n_bootstrap=0)
    summary = run(
        source,
        tmp_path / "drop",
        n_bootstrap=0,
        shape=replace(config.shape, missing_policy="drop_observations"),
    )
    assert len(summary["unit_ids"]) == 63
    assert len(summary["excluded_observations"]) == 1


def test_saved_shape_and_raw_coordinates_project_to_exported_scores(source, tmp_path):
    raw, _ = source
    output = tmp_path / "roundtrip"
    summary = run(source, output, n_bootstrap=0)
    with (raw / "coordinates.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    x = np.array(
        [
            [[float(r["x"]), float(r["y"])] for r in rows if r["observation_id"] == u]
            for u in summary["unit_ids"]
        ]
    )
    model, _, _ = read_shape_model(output / "shape")
    projection = project_shape_pca(x, model=model)
    with np.load(output / "mixture.npz") as arrays:
        np.testing.assert_allclose(projection.scores, arrays["scores"], atol=1e-12)
