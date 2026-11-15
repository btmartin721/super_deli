"""Matched source bindings, native parity and shape-specific failures."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.workflows.genomic_analysis_adapters import run_genomic_arm
from super_deli.workflows.multimodal_analysis import _read_arm, run_multimodal_analysis

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")


@pytest.fixture
def generator():
    path = Path(__file__).resolve().parents[1] / "examples/multimodal/synthetic.py"
    spec = importlib.util.spec_from_file_location("multimodal_synthetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dimensions,format", [(2, "tps"), (2, "csv"), (3, "csv")])
def test_without_genomics_and_native_parity(generator, tmp_path, dimensions, format):
    from super_deli.core.schema import config_from_dict

    d, c = generator.invented(tmp_path / "source", dimensions=dimensions, format=format)
    d, c = generator.select_views(d, c, {"outline", "traits"})
    out = tmp_path / "analysis"
    s = run_multimodal_analysis(d, analysis_config=c, base_dir=tmp_path / "source", output_dir=out)
    assert s["status"] == "complete" and len(s["models"]) == 3
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["cohort"]["source_locus_count"] is None
    frozen = config_from_dict(manifest["frozen_dataset"])
    for a in c.arms:
        replay = tmp_path / ("native-" + a.name)
        run_genomic_arm(a, frozen, out / "inputs", replay)
        assert _read_arm(a, replay, manifest["cohort"], frozen) == _read_arm(
            a, out / "arms" / a.name, manifest["cohort"], frozen
        )
    assert (out / "report.html").read_text().count("<circle ") == 64 * 3
    with pytest.raises(FileExistsError):
        run_multimodal_analysis(d, analysis_config=c, base_dir=tmp_path / "source", output_dir=out)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_coordinate",
        "missing_landmark",
        "duplicate_specimen",
        "image_source",
        "missing_specimen",
    ],
)
def test_prefit_guards(generator, tmp_path, fault):
    from super_deli.core.data import Specimen

    d, c = generator.invented(tmp_path / "source", format="csv")
    d, c = generator.select_views(d, c, {"outline", "traits"})
    shape = next(m for m in d.modalities if m.kind == "morphometric")
    path = tmp_path / "source" / shape.path
    if fault in {"missing_coordinate", "missing_landmark"}:
        lines = path.read_text().splitlines()
        if fault == "missing_landmark":
            lines.pop(1)
        else:
            row = lines[1].split(",")
            row[-1] = ""
            lines[1] = ",".join(row)
        path.write_text("\n".join(lines) + "\n")
    elif fault == "duplicate_specimen":
        obs = list(shape.observations)
        obs[1] = replace(obs[1], unit_id=obs[0].unit_id)
        d = replace(
            d,
            modalities=tuple(
                replace(m, observations=tuple(obs)) if m.name == shape.name else m
                for m in d.modalities
            ),
        )
    elif fault == "image_source":
        d = replace(
            d,
            modalities=tuple(
                replace(m, format="image_manifest", morphometric_options=None)
                if m.name == shape.name
                else m
                for m in d.modalities
            ),
        )
    else:
        d = replace(d, specimens=(*d.specimens, Specimen("absent")))
    out = tmp_path / "failed"
    with pytest.raises((ValueError, RuntimeError)):
        run_multimodal_analysis(d, analysis_config=c, base_dir=tmp_path / "source", output_dir=out)
    assert not (out / "arms").exists() and (out / "failure.json").exists()


def test_observation_binding_tamper(generator, tmp_path):
    from super_deli.core.schema import config_from_dict

    d, c = generator.invented(tmp_path / "source")
    out = tmp_path / "analysis"
    s = run_multimodal_analysis(d, analysis_config=c, base_dir=tmp_path / "source", output_dir=out)
    assert s["status"] == "complete"
    manifest = json.loads((out / "manifest.json").read_text())
    arm = next(a for a in c.arms if a.family == "shape")
    path = out / "arms" / arm.name / "summary.json"
    native = json.loads(path.read_text())
    native["observation_ids"][0], native["observation_ids"][1] = (
        native["observation_ids"][1],
        native["observation_ids"][0],
    )
    path.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="binding"):
        _read_arm(
            arm, path.parent, manifest["cohort"], config_from_dict(manifest["frozen_dataset"])
        )


def test_bootstrap_gpa_refits_and_failures(generator, tmp_path, monkeypatch):
    from super_deli.models import integrative, shape_gmm, shape_pca

    seen = []
    for module in (shape_pca, shape_gmm, integrative):
        native = module.fit_shape_pca

        def observe(x, *, config=None, native=native):
            seen.append(np.asarray(x).copy())
            return native(x, config=config)

        monkeypatch.setattr(module, "fit_shape_pca", observe)
    d, c = generator.invented(tmp_path / "source", draws=2)
    d, c = generator.select_views(d, c, {"outline", "traits"})
    s = run_multimodal_analysis(
        d, analysis_config=c, base_dir=tmp_path / "source", output_dir=tmp_path / "analysis"
    )
    assert s["status"] == "complete" and len(seen) == 6
    assert all(x.shape == (64, 6, 2) for x in seen)
    assert all(r["stability"]["requested"] == 2 for r in s["models"])

    def fail(*args, **kwargs):
        raise RuntimeError("invented shape fitting failure")

    monkeypatch.setattr(shape_gmm, "_cluster_fitted_shape", fail)
    s = run_multimodal_analysis(
        d, analysis_config=c, base_dir=tmp_path / "source", output_dir=tmp_path / "partial"
    )
    assert s["status"] == "partial"
    row = next(r for r in s["models"] if r["family"] == "shape")
    assert row["status"] == "failed" and row["selected_k"] is None
    assert all(
        r["status"] == "unavailable" for r in s["comparisons"] if "shape" in (r["left"], r["right"])
    )


def test_export_failure_does_not_leave_a_point(generator, tmp_path, monkeypatch):
    from super_deli.workflows import matched_comparison as engine

    d, c = generator.invented(tmp_path / "source")
    write = engine.write_json

    def export(value, path):
        if str(path).endswith("/shape/comparison-point.json"):
            raise OSError("invented export failure")
        return write(value, path)

    monkeypatch.setattr(engine, "write_json", export)
    s = run_multimodal_analysis(
        d, analysis_config=c, base_dir=tmp_path / "source", output_dir=tmp_path / "failed-export"
    )
    row = next(r for r in s["models"] if r["family"] == "shape")
    assert s["status"] == "partial" and row["status"] == "failed" and row["selected_k"] is None
