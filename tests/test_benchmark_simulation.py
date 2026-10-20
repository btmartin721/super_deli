"""Generative truth, linkage, input provenance and source preservation checks."""

# ruff: noqa: E402

import json
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from super_deli.benchmarking.config import KINDS, BenchmarkConfig, Scenario
from super_deli.benchmarking.simulation import simulate, write_simulation
from super_deli.core.provenance import fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import validate_dataset
from super_deli.io.config import read_json

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/benchmarking/smoke.json"


def config():
    return BenchmarkConfig.from_dict(json.loads(EXAMPLE.read_text()))


@pytest.mark.parametrize("pattern", ["null", "concordant", "complementary", "conflicting"])
@pytest.mark.parametrize("linkage", ["matched", "grouped"])
def test_reproducible_simulation_linkage_and_truth_safety(tmp_path, pattern, linkage):
    scenario = Scenario(
        "test",
        pattern,
        linkage,
        strength=0 if pattern == "null" else 3,
        unequal_groups=linkage == "grouped",
        view_missing_rate=0.25,
    )
    first, second = (simulate(scenario, config(), 71) for _ in range(2))
    assert first.target_ids == second.target_ids and first.primary_labels == second.primary_labels
    assert len(set(first.primary_labels)) == (1 if pattern == "null" else 4)
    assert (first.alternative_labels is not None) == (pattern == "conflicting")
    for kind in KINDS:
        assert [(r["target"], r["source"]) for r in first.rows[kind]] == [
            (r["target"], r["source"]) for r in second.rows[kind]
        ]
        for a, b in zip(first.rows[kind], second.rows[kind], strict=True):
            np.testing.assert_array_equal(a["values"], b["values"])
    assert 0 < len(first.common_ids) < config().n_units
    if linkage == "grouped":
        source_sets = [{r["source"] for r in first.rows[k]} for k in KINDS if k != "ecological"]
        assert not set.intersection(*source_sets)
    output = tmp_path / "data"
    cohort = write_simulation(first, output)
    assert validate_dataset(
        config_from_dict(cohort), base_dir=output / "cohort", check_files=True
    ).valid
    assert not list((output / "cohort").rglob("truth*"))
    assert not list((output / "observed").rglob("truth*"))
    for modality in cohort["modalities"]:
        assert "primary" not in json.dumps(modality) and "alternative" not in json.dumps(modality)
    manifest = read_json(output / "manifest.json")
    for name, digest in manifest["input_hashes"].items():
        assert fingerprint_file(output / name).sha256 == digest
    before = {str(p): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        write_simulation(second, output)
    assert before == {str(p): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_missing_cells_and_unequal_members_are_explicit():
    scenario = Scenario("missing", linkage="grouped", unequal_groups=True, cell_missing_rate=0.3)
    data = simulate(scenario, config(), 42)
    from collections import Counter

    for kind in KINDS:
        values = np.stack([r["values"] for r in data.rows[kind]])
        if kind == "morphometric":
            assert np.isfinite(values).all()
        else:
            assert np.isnan(values).any()
        if kind != "ecological":
            counts = Counter(r["target"] for r in data.rows[kind])
            assert len(set(counts.values())) > 1 and min(counts.values()) == 1
    assert data.common_ids == data.target_ids
    changed = simulate(replace(scenario, cell_missing_rate=0), config(), 42)
    assert data.primary_labels == changed.primary_labels
