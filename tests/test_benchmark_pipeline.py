"""Real four-view ingestion/refits, safe archives and retained failed attempts."""

# ruff: noqa: E402

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("snpio")

from super_deli.benchmarking.simulation import simulate, write_simulation
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.models.pipeline_config import PipelineCalibrationConfig
from super_deli.workflows.pipeline_calibration import run_reference_calibration

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/pipeline-calibration/calibration.json"
pytestmark = pytest.mark.genomic


def policy(grouped=False, **changes):
    config = PipelineCalibrationConfig.from_dict(read_json(EXAMPLE)).for_linkage(grouped)
    return replace(
        config,
        reference=replace(config.reference, n_units=20, n_loci=16),
        model=replace(config.model, max_k=2, n_init=1),
        n_simulations=2,
        **changes,
    )


def observed(config, directory, pattern="null"):
    data = simulate(config.reference.scenario(pattern), config.reference, 193)
    write_simulation(data, directory)
    return config_from_dict(read_json(directory / "observed/dataset.json"))


@pytest.mark.parametrize("grouped", [False, True])
def test_real_ingestion_repeated_for_every_draw_and_source_safety(tmp_path, monkeypatch, grouped):
    from super_deli.io import genomic
    from super_deli.workflows import integrative

    config = policy(grouped)
    source = tmp_path / "raw"
    dataset = observed(config, source)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file()
    }
    ingestion, fitting = [], []
    reader, fitter = genomic.ingest_genomic, integrative.run_integrative

    def read(*args, **kwargs):
        ingestion.append(kwargs["output_dir"])
        return reader(*args, **kwargs)

    def fit(*args, **kwargs):
        fitting.append(kwargs["model_config"])
        return fitter(*args, **kwargs)

    monkeypatch.setattr(genomic, "ingest_genomic", read)
    monkeypatch.setattr(integrative, "run_integrative", fit)
    output = tmp_path / "calibration"
    result = run_reference_calibration(
        dataset, config=config, base_dir=source / "observed", output_dir=output
    )
    assert len(ingestion) == len(fitting) == 3
    assert len(set(ingestion)) == 3
    assert all(m.random_seed == config.model.random_seed and m.n_bootstrap == 0 for m in fitting)
    assert result["successful_fits"] == result["attempted_fits"] == 3
    assert result["inference"]["p_value"] >= 1 / 3
    inventory = read_json(output / "archive_inventory.json")
    with ZipFile(output / "pipeline_artifacts.zip") as archive:
        assert set(archive.namelist()) == set(inventory)
        for name, fingerprint in inventory.items():
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == fingerprint["sha256"]
            assert len(content) == fingerprint["size_bytes"]
        paths = ["observed", "null_0000", "null_0001"]
        means = []
        for prefix in paths:
            summary = json.loads(archive.read(f"{prefix}/run/summary.json"))
            assert summary["bootstrap"]["status"] == "not_computed"
            assert summary["point_seed"] == result["point"]["point_seed"]
            assert len(summary["views"]) == 4
            assert not any(
                name.startswith(f"{prefix}/run/") and name.endswith("truth.json")
                for name in archive.namelist()
            )
            import io

            with np.load(
                io.BytesIO(archive.read(f"{prefix}/run/view_001/parameters/pca.npz")),
                allow_pickle=False,
            ) as arrays:
                means.append(arrays["feature_means"])
        assert not np.array_equal(means[0], means[1])
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file()
    }
    assert not list(output.glob("null-*")) and not list(output.glob("observed-*"))
    with pytest.raises(FileExistsError):
        run_reference_calibration(
            dataset, config=config, base_dir=source / "observed", output_dir=output
        )


@pytest.mark.parametrize("failed_call", [1, 2])
def test_failed_pipelines_preserve_requested_denominators(tmp_path, monkeypatch, failed_call):
    from super_deli.workflows import integrative

    config = policy()
    source = tmp_path / "raw"
    dataset = observed(config, source)
    original, calls = integrative.run_integrative, []

    def fail(*args, **kwargs):
        calls.append(None)
        if len(calls) == failed_call:
            raise ValueError("injected full-pipeline failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", fail)
    result = run_reference_calibration(
        dataset, config=config, base_dir=source / "observed", output_dir=tmp_path / "test"
    )
    assert result["inference"]["p_value"] is None
    assert result["inference"]["reject_reference_population"] is None
    assert result["planned_fits"] == 3 and result["failed_fits"] == 1
    assert result["attempted_fits"] == (1 if failed_call == 1 else 3)
    assert result["unattempted_fits"] == (2 if failed_call == 1 else 0)
    assert len(calls) == result["attempted_fits"]


def test_missing_cells_are_imputed_and_missing_view_alignment_is_refitted(tmp_path):
    config = policy()
    config = replace(
        config,
        reference=replace(
            config.reference, n_units=32, cell_missing_rate=0.1, view_missing_rate=0.1
        ),
        model=replace(config.model, missing_views="complete_cases"),
    )
    source = tmp_path / "raw"
    dataset = observed(config, source)
    result = run_reference_calibration(
        dataset, config=config, base_dir=source / "observed", output_dir=tmp_path / "test"
    )
    assert result["point"]["n_retained"] < 32
    assert result["attempted_fits"] == 3
