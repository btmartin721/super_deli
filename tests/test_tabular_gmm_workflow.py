"""Tabular clustering CLI, parameter reconstruction, inference levels and input safety."""

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
from super_deli.io.config import read_json
from super_deli.models import tabular_gmm
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.workflows.tabular_gmm import run_tabular_gmm

pytestmark = pytest.mark.baseline
EXAMPLES = Path(__file__).resolve().parents[1] / "examples/tabular/clustering"


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


def run(source, output, modality="traits", **changes):
    raw, dataset = source
    filename = "morphology_model.json" if modality == "traits" else "ecology_model.json"
    config = replace(TabularClusterConfig.from_dict(read_json(raw / filename)), **changes)
    return run_tabular_gmm(
        dataset, model_config=config, base_dir=raw, output_dir=output, modality_name=modality
    )


@pytest.mark.parametrize("modality,level", [("traits", "specimen"), ("environment", "locality")])
def test_truth_parameter_reconstruction_identity_and_safety(source, tmp_path, modality, level):
    raw, _ = source
    before = snapshot(raw)
    output = tmp_path / modality
    summary = run(source, output, modality, n_bootstrap=5)
    assert summary["selected_k"] == 2 and summary["level"] == level
    assert summary["bootstrap"]["successful"] == 5
    typed = read_json(output / "ingestion/modality_000/table.json")
    prep = read_json(output / "preprocessing.json")
    indices = {f["name"]: i for i, f in enumerate(typed["features"])}
    raw_values = np.array(
        [[r[indices[name]] for name in prep["feature_order"]] for r in typed["values"]], dtype=float
    )
    for i, dtype in enumerate(prep["feature_types"]):
        if dtype == "count":
            raw_values[:, i] = np.log1p(raw_values[:, i])
    with np.load(output / "preprocessing.npz", allow_pickle=False) as params:
        scaled = (raw_values[:, params["kept_indices"]] - params["means"]) / params["scales"]
        projected = (scaled - params["pca_mean"]) @ params["components"].T
        assert params["components"].shape[0] == 2
    with np.load(output / "mixture.npz", allow_pickle=False) as params:
        np.testing.assert_allclose(params["scores"], projected)
        model = GaussianMixture(n_components=2, covariance_type="full")
        model.weights_, model.means_ = params["weights"], params["means"]
        model.covariances_ = params["covariances"]
        model.precisions_cholesky_ = params["precisions_cholesky"]
        np.testing.assert_allclose(model.predict_proba(projected), params["membership"])
        truth = [u["latent_group"] for u in read_json(raw / "truth.json")["units"]]
        assert adjusted_rand_score(truth, params["labels"]) == 1
    with (output / "assignments.csv").open() as handle:
        assignments = list(csv.DictReader(handle))
    assert [r["observation_id"] for r in assignments] == summary["observation_ids"]
    assert [r["unit_id"] for r in assignments] == summary["unit_ids"]
    with (output / "scores.csv").open() as handle:
        scores = list(csv.DictReader(handle))
    assert [r["observation_id"] for r in scores] == summary["observation_ids"]
    np.testing.assert_allclose([[float(r["PC1"]), float(r["PC2"])] for r in scores], projected)
    result = read_json(output / "result.json")
    assert result["level"] == level
    assert [r["unit_id"] for r in result["assignments"]] == summary["unit_ids"]
    assert next(e for e in result["evidence"] if e["kind"] == "p_value")["status"] == "not_computed"
    provenance = summary["provenance"]
    assert provenance["configuration_sha256"] == configuration_hash(provenance["configuration"])
    assert prep["array_sha256"] == fingerprint_file(output / "preprocessing.npz").sha256
    assert provenance["mixture_array_sha256"] == fingerprint_file(output / "mixture.npz").sha256
    if modality == "traits":
        assert prep["nonselected_columns"] == ["pigmentation", "sex"]
    assert not list(output.rglob("truth.json"))
    assert snapshot(raw) == before and not list(Path.cwd().iterdir())
    before_output = snapshot(output)
    with pytest.raises(FileExistsError):
        run(source, output, modality)
    assert snapshot(output) == before_output


def test_cli_json_and_outside_checkout_execution(source, tmp_path, capsys):
    raw, _ = source
    args = [
        "cluster-tabular",
        str(raw / "dataset.json"),
        "--model-config",
        str(raw / "ecology_model.json"),
        "--modality",
        "environment",
        "--output-dir",
        str(tmp_path / "cli"),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["level"] == "locality"
    result = subprocess.run(
        [sys.executable, "-m", "super_deli", *args[:-1], str(tmp_path / "subprocess")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "complete"


def test_multiple_tables_require_selection_and_do_not_fuse(source, tmp_path):
    raw, dataset = source
    for name in (None, "unknown"):
        with pytest.raises(ValueError, match="exactly one"):
            run_tabular_gmm(
                dataset,
                model_config=TabularClusterConfig(True),
                base_dir=raw,
                output_dir=tmp_path / "ambiguous",
                modality_name=name,
            )
    assert not (tmp_path / "ambiguous").exists()


def test_nonselected_categories_and_covariates_do_not_change_fits(source, tmp_path):
    raw, _ = source
    first = tmp_path / "first"
    run(source, first, n_bootstrap=0)
    with (raw / "morphology.csv").open() as handle:
        reader = csv.DictReader(handle)
        rows, columns = list(reader), reader.fieldnames
    for i, row in enumerate(rows):
        row["sex"] = "male" if i % 2 else "female"
        row["pigmentation"] = "plain" if i % 3 else "striped"
    with (raw / "morphology.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    second = tmp_path / "second"
    run(source, second, n_bootstrap=0)
    with np.load(first / "mixture.npz") as a, np.load(second / "mixture.npz") as b:
        np.testing.assert_array_equal(a["membership"], b["membership"])


def test_unavailable_bootstrap_evidence_keeps_point_result(source, tmp_path, monkeypatch):
    original, calls = tabular_gmm._fit_once, 0

    def fail_once(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("deliberate numerical failure")
        return original(*args)

    monkeypatch.setattr(tabular_gmm, "_fit_once", fail_once)
    output = tmp_path / "partial"
    summary = run(source, output, n_bootstrap=3)
    assert summary["status"] == "complete" and summary["bootstrap"]["successful"] == 2
    assert summary["bootstrap"]["status"] == "unavailable" and calls == 4
    evidence = read_json(output / "result.json")["evidence"]
    assert all(e["value"] is None for e in evidence if e["kind"] == "bootstrap_frequency")
    for cluster in read_json(output / "clusters.json"):
        assert cluster["frequency"] is None
        assert cluster["failure_upper"] - cluster["failure_lower"] == pytest.approx(1 / 3)
    assert not (output / "failure.json").exists()


def test_bootstrap_disabled_and_assignment_threshold(source, tmp_path):
    output = tmp_path / "disabled"
    summary = run(source, output, n_bootstrap=0, assignment_threshold=1)
    assert summary["bootstrap"]["status"] == "not_computed"
    assert not (output / "stability.npz").exists()
    result = read_json(output / "result.json")
    for assignment, evidence in zip(result["assignments"], result["evidence"][:64], strict=True):
        assert (assignment["label"] is not None) == (evidence["value"] >= 1)
    assert all(
        e["status"] == "not_computed"
        for e in result["evidence"]
        if e["kind"] == "bootstrap_frequency"
    )


def test_missingness_and_bad_feature_selection_leave_failure_diagnostics(source, tmp_path):
    raw, _ = source
    for label, changes in (
        ("mixed", {"feature_names": ()}),
        ("covariate", {"feature_names": ("length_mm", "sex")}),
    ):
        with pytest.raises(ValueError):
            run(source, tmp_path / label, n_bootstrap=0, **changes)
        assert (tmp_path / label / "failure.json").exists()
        assert not (tmp_path / label / "summary.json").exists()
    lines = (raw / "morphology.csv").read_text().splitlines()
    row = lines[1].split(",")
    row[1] = ""
    lines[1] = ",".join(row)
    (raw / "morphology.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="missing_policy"):
        run(source, tmp_path / "missing", n_bootstrap=0)
    summary = run(source, tmp_path / "mean", n_bootstrap=0, missing_policy="mean")
    assert summary["preprocessing"]["imputed_cells"] == 1


def test_repeated_units_are_rejected_before_model_fitting(source, tmp_path):
    raw, dataset = source
    modality = dataset.modalities[0]
    observations = (
        modality.observations[0],
        replace(modality.observations[1], unit_id=modality.observations[0].unit_id),
        *modality.observations[2:],
    )
    repeated = replace(dataset, modalities=(replace(modality, observations=observations),))
    with pytest.raises(ValueError, match="Repeated"):
        run((raw, repeated), tmp_path / "repeat", n_bootstrap=0)
    assert not (tmp_path / "repeat/mixture.npz").exists()
