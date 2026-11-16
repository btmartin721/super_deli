"""Refits, source protection and failure-aware inclusion/omission evidence."""

# ruff: noqa: E402
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.io.config import read_json
from super_deli.models.individual_subsample_config import IndividualSubsampleConfig
from super_deli.models.individual_subsampling import draw_plan, summarize

pytestmark = pytest.mark.baseline


def test_draws_are_unique_paired_reproducible():
    c = IndividualSubsampleConfig(25, 1601)
    a = draw_plan(64, c, 301)
    b = draw_plan(64, c, 301)
    for i, ((x, seed), (y, seed2)) in enumerate(zip(a, b, strict=True)):
        np.testing.assert_array_equal(x, y)
        assert len(x) == len(set(x)) == 51 and seed == seed2
        child = np.random.SeedSequence(1601).spawn(25)[i]
        np.testing.assert_array_equal(
            x, np.sort(np.random.default_rng(child).choice(64, 51, replace=False))
        )
        assert seed == int(np.random.SeedSequence(301).spawn(26)[i + 1].generate_state(2)[1])


def test_omitted_denominators_and_failed_draw_bounds():
    source = SimpleNamespace(
        units=tuple(range(8)),
        labels=np.array([0] * 4 + [1] * 4),
        model=SimpleNamespace(min_cluster_size=3, max_k=2),
        summary={"selected_k": 2},
    )
    records = [
        dict(
            training_indices=[0, 1, 4, 5],
            omitted_indices=[2, 3, 6, 7],
            status="complete",
            labels=[0, 0, 1, 1],
            projected_labels=[0] * 4 + [1] * 4,
            selected_k=2,
            candidates=[dict(k=1, status="eligible"), dict(k=2, status="eligible")],
        ),
        dict(
            training_indices=[0, 2, 4, 6],
            omitted_indices=[1, 3, 5, 7],
            status="failed",
            error="invented",
        ),
    ]
    summary, a, support = summarize(source, records)
    assert summary["status"] == "incomplete" and summary["failed"] == 1
    assert a["omitted_requested"][3, 7] == 2 and a["omitted_successful"][3, 7] == 1
    assert (
        a["omitted_coassignment_lower"][3, 7] == 0 and a["omitted_coassignment_upper"][3, 7] == 0.5
    )
    assert np.isnan(a["omitted_coassignment"][3, 7])
    assert a["omitted_coassignment"][2, 3] == 1
    assert (
        a["included_coassignment_lower"][0, 2] == 0 and a["included_coassignment_upper"][0, 2] == 1
    )
    assert np.isnan(a["included_coassignment"][0, 2])
    assert summary["selected_k_frequencies"][0]["count"] == 0
    assert (
        support[0]["below_declared_minimum"] == 2
        and support[0]["best_jaccard_lower"] == 0.5
        and support[0]["best_jaccard_upper"] == 1
    )


def test_native_point_preserved_and_all_views_refitted(tmp_path):
    from super_deli.benchmarking.signal_loss_source import digest
    from super_deli.io.individual_subsample_source import load_point_source
    from super_deli.models.full_retention import full_retention_model
    from super_deli.workflows.individual_subsampling import run_individual_subsampling
    from super_deli.workflows.integrative import run_integrative

    path = Path(__file__).resolve().parents[1] / "examples/multimodal/synthetic.py"
    spec = importlib.util.spec_from_file_location("individual_subsample_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset, analysis = module.invented(
        tmp_path / "inputs", seed=17501, n=64, law="shape_only", draws=0
    )
    dataset, analysis = module.select_views(dataset, analysis, {"traits", "habitat", "outline"})
    model = full_retention_model(
        replace(analysis.arms[-1].model, n_init=100, n_bootstrap=0), min_cluster_size=7
    )
    point = tmp_path / "point"
    run_integrative(dataset, model_config=model, base_dir=tmp_path / "inputs", output_dir=point)
    before = {str(p.relative_to(point)): digest(p) for p in point.rglob("*") if p.is_file()}
    result = run_individual_subsampling(
        point,
        stability_config=IndividualSubsampleConfig(2, 1601),
        output_dir=tmp_path / "subsample",
    )
    assert (
        result["successful"] == 2
        and result["subsample_individuals"] == 51
        and result["point_analysis_changed"] is False
    )
    assert before == {str(p.relative_to(point)): digest(p) for p in point.rglob("*") if p.is_file()}
    records = read_json(tmp_path / "subsample/subsample_replicates.json")
    for r in records:
        for i in range(3):
            metadata = read_json(
                tmp_path
                / "subsample/subsamples"
                / r["parameters"]
                / f"view_{i:03d}/parameters/summary.json"
            )
            assert len(metadata["training_source_indices"]) == 51
            assert {m["target_id"] for m in metadata["members"]} == set(r["training_unit_ids"])
    assert "species probabilities" in (tmp_path / "subsample/report.html").read_text()
    rows = list(__import__("csv").DictReader((tmp_path / "subsample/cluster_support.csv").open()))
    assert len(rows) == result["source_point_k"]
    from unittest.mock import patch

    from super_deli.models import integrative

    with patch.object(integrative, "_fit_once", side_effect=ValueError("invented failed draw")):
        failed = run_individual_subsampling(
            point,
            stability_config=IndividualSubsampleConfig(2, 1601),
            output_dir=tmp_path / "failed",
        )
    assert failed["successful"] == 0 and failed["failed"] == 2
    assert all(r["count"] == 0 and r["upper"] == 1 for r in failed["selected_k_frequencies"])
    assert all(
        "selected_k" not in r for r in read_json(tmp_path / "failed/subsample_replicates.json")
    )
    with patch(
        "super_deli.models.individual_subsampling._save_parameters",
        side_effect=ValueError("invented export error"),
    ):
        with pytest.raises(ValueError, match="export error"):
            run_individual_subsampling(
                point,
                stability_config=IndividualSubsampleConfig(1, 1601),
                output_dir=tmp_path / "export-error",
            )
    assert (tmp_path / "export-error/failure.json").is_file() and not (
        tmp_path / "export-error/summary.json"
    ).exists()
    # Raw-source drift fail closed rather than creating a new analysis.
    raw = point / "view_000/source_values.npy"
    data = raw.read_bytes()
    raw.write_bytes(data + b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_point_source(point)
    raw.write_bytes(data)
    with pytest.raises(ValueError, match="outside"):
        run_individual_subsampling(
            point, stability_config=IndividualSubsampleConfig(1, 1601), output_dir=point / "nested"
        )
