"""Artifact identity, policy enforcement, frozen projection and safe parameter loading."""

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from super_deli import load_config  # noqa: E402
from super_deli.cli import main  # noqa: E402
from super_deli.core.provenance import configuration_hash, fingerprint_file  # noqa: E402
from super_deli.io.config import read_json  # noqa: E402
from super_deli.io.morphometric import ingest_morphometric  # noqa: E402
from super_deli.io.shape_model import read_shape_model  # noqa: E402
from super_deli.models.shape_config import ShapeConfig  # noqa: E402
from super_deli.workflows.shape_pca import run_shape_pca  # noqa: E402

pytestmark = pytest.mark.shape
EXAMPLES = Path(__file__).resolve().parents[1] / "examples/shape"


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
    return raw, load_config(raw / "train_dataset.json")


def test_known_coordinate_truth_artifacts_and_roundtrip(source, tmp_path):
    raw, config = source
    before = snapshot(raw)
    run = run_shape_pca(
        config, model_config=ShapeConfig(), base_dir=raw, output_dir=tmp_path / "run"
    )
    expected = read_json(raw / "expected.json")
    np.testing.assert_allclose(run.projection.centroids, expected["centroids"], atol=1e-12)
    np.testing.assert_allclose(
        run.projection.centroid_sizes, expected["centroid_sizes"], atol=1e-12
    )
    np.testing.assert_allclose(run.model.consensus, expected["consensus"], atol=1e-12)
    assert run.summary["pca"]["retained_components"] == 1
    assert run.summary["uncertainty"]["status"] == "not_computed"
    assert run.summary["gpa"]["reflection"] == "forbid"
    p = run.summary["provenance"]
    assert p["configuration_sha256"] == configuration_hash(p["configuration"])
    assert p["ingestion"]["inputs"][0]["sha256"] == fingerprint_file(raw / "train.csv").sha256
    fitted, metadata, _ = read_shape_model(run.output_dir)
    np.testing.assert_array_equal(fitted.components, run.model.components)
    assert metadata["array_sha256"] == fingerprint_file(run.output_dir / "model.npz").sha256
    aligned_config = load_config(run.output_dir / "aligned_dataset.json")
    result = ingest_morphometric(
        aligned_config, base_dir=run.output_dir, output_dir=tmp_path / "roundtrip"
    )
    assert result.collections[0].coordinate_unit == "unit_centroid_size"
    np.testing.assert_allclose(
        [r.coordinates for r in result.collections[0].records], run.projection.aligned
    )
    with (run.output_dir / "scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["observation_id"] for r in rows] == run.summary["observation_ids"]
    np.testing.assert_allclose([float(r["PC1"]) for r in rows], run.projection.scores[:, 0])
    assert snapshot(raw) == before and not list(Path.cwd().iterdir())
    assert (run.output_dir / "summary.json").exists()
    assert not (run.output_dir / "failure.json").exists()


def test_projection_reuses_training_parameters_and_has_no_unit_leakage(source, tmp_path):
    raw, config = source
    train = run_shape_pca(
        config, model_config=ShapeConfig(), base_dir=raw, output_dir=tmp_path / "train"
    )
    original = snapshot(train.output_dir)
    result = run_shape_pca(
        load_config(raw / "projection_dataset.json"),
        model_config=ShapeConfig(),
        base_dir=raw,
        output_dir=tmp_path / "projection",
        reference_dir=train.output_dir,
    )
    assert result.summary["mode"] == "project"
    assert result.summary["gpa"]["iterations"] == 0
    assert not result.summary["diagnostics"]["training_unit_overlap"]
    np.testing.assert_array_equal(result.model.components, train.model.components)
    np.testing.assert_array_equal(result.model.consensus, train.model.consensus)
    assert len(result.summary["provenance"]["configuration"]["reference_inputs"]) == 2
    saved, metadata, _ = read_shape_model(result.output_dir)
    assert metadata["training_unit_ids"] == train.summary["unit_ids"]
    np.testing.assert_array_equal(saved.components, train.model.components)
    assert snapshot(train.output_dir) == original
    # A deliberate training-data reprojection remains identifiable as overlapping.
    same = run_shape_pca(
        config,
        model_config=ShapeConfig(),
        base_dir=raw,
        output_dir=tmp_path / "same",
        reference_dir=train.output_dir,
    )
    assert len(same.summary["diagnostics"]["training_unit_overlap"]) == 6


@pytest.mark.parametrize("zero_variance", [False, True])
def test_3d_exports_and_zero_rank_model_roundtrip(source, tmp_path, zero_variance):
    raw, config = source
    modality = config.modalities[0]
    observations = modality.observations[:4]
    points = np.array([[0, 0, 0], [3, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    with (raw / "3d.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("observation_id", "landmark_id", "x", "y", "z"))
        for i, observation in enumerate(observations):
            sample = points.copy()
            if not zero_variance:
                sample[-1, -1] += i * 0.1
            sample = sample * (i + 1) + [i * 3, -i * 2, i]
            for name, row in zip(modality.morphometric_options.landmark_ids, sample, strict=True):
                writer.writerow((observation.observation_id, name, *row))
    config = replace(
        config,
        modalities=(
            replace(
                modality,
                path="3d.csv",
                observations=observations,
                morphometric_options=replace(modality.morphometric_options, dimensions=3),
            ),
        ),
    )
    run = run_shape_pca(
        config, model_config=ShapeConfig(), base_dir=raw, output_dir=tmp_path / "3d"
    )
    model, _, _ = read_shape_model(run.output_dir)
    assert (model.numerical_rank == 0) == zero_variance
    assert run.projection.scores.shape[1] == model.numerical_rank
    assert (run.projection.scores.shape[1] == 0) == zero_variance
    exported = load_config(run.output_dir / "aligned_dataset.json")
    roundtrip = ingest_morphometric(
        exported, base_dir=run.output_dir, output_dir=tmp_path / "roundtrip3d"
    )
    np.testing.assert_allclose(
        [r.coordinates for r in roundtrip.collections[0].records], run.projection.aligned
    )


def test_missingness_and_repeat_weighting_are_explicit(source, tmp_path):
    raw, config = source
    path = raw / "train.csv"
    with path.open() as handle:
        rows = list(csv.reader(handle))
    rows[1][2] = ""
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="incomplete"):
        run_shape_pca(
            config, model_config=ShapeConfig(), base_dir=raw, output_dir=tmp_path / "missing"
        )
    assert not (tmp_path / "missing/summary.json").exists()
    assert (tmp_path / "missing/failure.json").exists()
    dropped = run_shape_pca(
        config,
        model_config=ShapeConfig(missing_policy="drop_observations"),
        base_dir=raw,
        output_dir=tmp_path / "dropped",
    )
    assert len(dropped.aligned.records) == 5 and len(dropped.summary["excluded_observations"]) == 1
    assert (
        len(load_config(dropped.output_dir / "aligned_dataset.json").modalities[0].observations)
        == 5
    )
    m = config.modalities[0]
    obs = list(m.observations)
    obs[2] = replace(obs[2], unit_id=obs[1].unit_id)
    repeated = replace(config, modalities=(replace(m, observations=tuple(obs)),))
    policy = ShapeConfig(missing_policy="drop_observations")
    with pytest.raises(ValueError, match="repeated_policy"):
        run_shape_pca(repeated, model_config=policy, base_dir=raw, output_dir=tmp_path / "repeats")
    allowed = run_shape_pca(
        repeated,
        model_config=replace(policy, repeated_policy="allow"),
        base_dir=raw,
        output_dir=tmp_path / "allowed",
    )
    assert allowed.summary["repeated_units"] == {obs[1].unit_id: 2}


def test_reference_schema_and_existing_output_safety(source, tmp_path):
    raw, config = source
    fitted = run_shape_pca(
        config, model_config=ShapeConfig(), base_dir=raw, output_dir=tmp_path / "fit"
    )
    before = snapshot(fitted.output_dir)
    with pytest.raises(FileExistsError):
        run_shape_pca(
            config, model_config=ShapeConfig(), base_dir=raw, output_dir=fitted.output_dir
        )
    assert snapshot(fitted.output_dir) == before
    m = config.modalities[0]
    changed = replace(
        config,
        modalities=(
            replace(
                m,
                morphometric_options=replace(
                    m.morphometric_options,
                    landmark_ids=tuple(reversed(m.morphometric_options.landmark_ids)),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="identities"):
        run_shape_pca(
            changed,
            model_config=ShapeConfig(),
            base_dir=raw,
            output_dir=tmp_path / "wrong_order",
            reference_dir=fitted.output_dir,
        )
    assert not (tmp_path / "wrong_order").exists()
    with pytest.raises(ValueError, match="configuration"):
        run_shape_pca(
            config,
            model_config=ShapeConfig(reflection="allow"),
            base_dir=raw,
            output_dir=tmp_path / "wrong_policy",
            reference_dir=fitted.output_dir,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("dimensions", 4),
        ("numerical_rank", 999),
        ("shape_dimension", 999),
        ("total_variance", -1),
        ("array_sha256", "bad"),
        ("array_path", "../elsewhere.npz"),
        ("training_unit_ids", []),
    ],
)
def test_model_metadata_corruption_is_rejected(source, tmp_path, field, value):
    raw, config = source
    run = run_shape_pca(
        config, model_config=ShapeConfig(), base_dir=raw, output_dir=tmp_path / "fit"
    )
    path = run.output_dir / "model.json"
    metadata = read_json(path)
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        read_shape_model(run.output_dir)


@pytest.mark.parametrize(
    "array", ["components", "consensus", "tangent_mean", "object", "precision"]
)
def test_invalid_model_arrays_are_never_executed_or_accepted(source, tmp_path, array):
    raw, config = source
    run = run_shape_pca(
        config, model_config=ShapeConfig(), base_dir=raw, output_dir=tmp_path / "fit"
    )
    path = run.output_dir / "model.npz"
    with np.load(path, allow_pickle=False) as loaded:
        values = {name: loaded[name].copy() for name in loaded.files}
    if array == "components":
        values[array] = np.array(3.0)
    elif array == "object":
        values["consensus"] = np.array([None], dtype=object)
    elif array == "precision":
        values = {name: value.astype(np.float32) for name, value in values.items()}
    else:
        values[array] += 5
    np.savez(path, **values)
    metadata = read_json(run.output_dir / "model.json")
    metadata["array_sha256"] = fingerprint_file(path).sha256
    (run.output_dir / "model.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        read_shape_model(run.output_dir)


def test_convergence_failure_keeps_diagnostics_without_completion(source, tmp_path):
    raw, config = source
    with pytest.raises(ValueError, match="converge"):
        run_shape_pca(
            config, model_config=ShapeConfig(max_iter=1), base_dir=raw, output_dir=tmp_path / "run"
        )
    assert (tmp_path / "run/failure.json").exists()
    assert len(read_json(tmp_path / "run/gpa_history.json")) == 1
    assert not (tmp_path / "run/summary.json").exists()


@pytest.mark.parametrize("level", ["population", "locality"])
def test_population_and_locality_levels_are_preserved(source, tmp_path, level):
    raw, config = source
    units = tuple(r.specimen_id for r in config.specimens)
    config = replace(
        config,
        specimens=(),
        populations=units if level == "population" else (),
        localities=units if level == "locality" else (),
        modalities=(replace(config.modalities[0], level=level),),
    )
    run = run_shape_pca(
        config, model_config=ShapeConfig(), base_dir=raw, output_dir=tmp_path / "run"
    )
    assert run.aligned.level == level and run.summary["level"] == level


def test_cli_and_no_image_snpio_or_sklearn_imports(source, tmp_path, capsys):
    raw, _ = source
    assert (
        main(
            [
                "shape-pca",
                str(raw / "train_dataset.json"),
                "--model-config",
                str(raw / "model.json"),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["pca"]["retained_components"] == 1
    code = """
import sys
from pathlib import Path
from super_deli import load_config
from super_deli.models.shape_config import ShapeConfig
from super_deli.workflows.shape_pca import run_shape_pca
p=Path(sys.argv[1]);run_shape_pca(load_config(p),model_config=ShapeConfig(),base_dir=p.parent,output_dir=sys.argv[2])
assert not {'snpio','sklearn','PIL','scipy'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(raw / "train_dataset.json"), str(tmp_path / "child")],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
