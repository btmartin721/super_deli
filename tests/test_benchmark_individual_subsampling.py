"""Independent end-to-end sampling and support guards on a tiny invented cohort."""

# ruff: noqa: E402
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("snpio")
from super_deli.benchmarking.individual_subsample_audit import audit_subsampling
from super_deli.benchmarking.individual_subsample_simulation import load_evaluation
from super_deli.benchmarking.individual_subsample_study import parent
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.models.integrative_config import IntegrativeConfig

pytestmark = pytest.mark.genomic


def test_sampling_and_support_corruption_are_detected(tmp_path):
    generator = Path(__file__).resolve().parents[1] / "examples/multimodal/synthetic.py"
    row = parent(
        dict(index=0, n=64, law="shape_unequal", seed=17571), tmp_path / "case", generator, draws=1
    )
    assert row["searches"] == row["attempted_searches"] == 3 and row["withheld"] == 0
    case = tmp_path / "case"
    base = case / "evaluation"
    model = IntegrativeConfig.from_dict(row["configuration"])
    evaluation = load_evaluation(
        config_from_dict(read_json(base / "dataset.json")), model, base, case / "audit-evaluation"
    )
    path = case / "subsample/subsample_replicates.json"
    before = path.read_bytes()
    changed = read_json(path)
    changed[0]["training_indices"][0] = changed[0]["training_indices"][1]
    path.write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        audit_subsampling(case / "subsample", evaluation, write=False)
    path.write_bytes(before)
    path = case / "subsample/cluster_support.json"
    before = path.read_bytes()
    changed = read_json(path)
    changed[0]["below_declared_minimum"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        audit_subsampling(case / "subsample", evaluation, write=False)
    path.write_bytes(before)
    assert audit_subsampling(case / "subsample", evaluation, write=False)["passed"]
