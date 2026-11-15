"""Tampering with selected initialization batches must fail independent auditing."""

# ruff: noqa: E402
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.initialization_audit import audit_initialization
from super_deli.benchmarking.initialization_study import native
from super_deli.io.config import read_json
from super_deli.models.full_retention import full_retention_model
from super_deli.models.initialization import truth_free_model

pytestmark = pytest.mark.baseline


def test_changed_batch_choice_and_draw_identity_are_detected(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples/multimodal/synthetic.py"
    spec = importlib.util.spec_from_file_location("init_audit_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    dataset, analysis = module.invented(source, seed=74403, law="complementary")
    dataset, analysis = module.select_views(dataset, analysis, {"traits", "habitat", "outline"})
    model = truth_free_model(
        full_retention_model(replace(analysis.arms[-1].model, n_bootstrap=1), min_cluster_size=7)
    )
    out = tmp_path / "native"
    row = native(
        dataset,
        model,
        source,
        out,
        "complementary",
        read_json(source / "truth-evaluation-only.json"),
    )
    assert row["status"] == "complete" and row["audit"]["draws"] == 1
    target = out / "observed-pipeline/events.json"
    before = target.read_bytes()
    import json

    changed = read_json(target)
    entry = next(c for c in changed[0]["candidates"] if "initialization" in c)
    entry["initialization"]["selected_batch"] = (entry["initialization"]["selected_batch"] + 1) % 3
    target.write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        audit_initialization(out)
    target.write_bytes(before)
    assert audit_initialization(out) == row["audit"]


def test_conflicting_stress_has_no_invented_recovery_or_contrast(tmp_path):
    import json

    from super_deli.benchmarking.initialization_report import aggregate

    path = tmp_path / "observed-pipeline"
    path.mkdir()
    (path / "events.json").write_text(
        json.dumps([{"candidates": [{"k": 4, "status": "eligible"}]}])
    )
    row = dict(
        status="complete",
        path=str(tmp_path),
        configuration={"n_bootstrap": 0},
        selected_k=2,
        target_k=None,
        ari=None,
        correct_recovery=None,
        control_split=None,
    )
    summary = dict(
        phase="invented",
        records=[
            dict(
                origin="fresh",
                task={"cell": 4, "n": 64, "law": "conflicting"},
                arms={name: row for name in ("legacy10", "legacy100", "diverse100")},
            )
        ],
        candidate_searches=0,
        independent={},
        exact_parity_arrays=0,
    )
    report = aggregate(summary)
    assert len(report["cells"]) == 3
    for cell in report["cells"]:
        assert cell["event"] == "descriptive_partition_only"
        assert cell["event_count"] is cell["event_interval"] is cell["correct_k_count"] is None
    for pair in report["paired_comparisons"]:
        assert pair["difference"] is pair["difference_interval"] is None
        assert pair["outcomes"] == {"descriptive_only": 1}


def test_unused_ties_do_not_relax_used_labels_or_densities(tmp_path):
    import json

    import numpy as np

    from super_deli.benchmarking.multimodal_validation import array_hash
    from super_deli.benchmarking.signal_loss_fit import audit_candidates

    x = np.zeros((8, 2))
    membership = np.tile([0.5 - np.finfo(float).eps, 0.5 + np.finfo(float).eps], (8, 1))
    values = dict(
        scores=x,
        weights=np.array([0.5, 0.5]),
        means=np.zeros((2, 2)),
        covariances=np.eye(2),
        membership=membership,
        labels=np.ones(8, dtype=int),
    )
    path = tmp_path / "candidate.npz"
    np.savez(path, **values)
    likelihood = -len(x) * np.log(2 * np.pi)
    record = dict(
        arrays=path.name,
        fit_status="complete",
        scores_sha256=array_hash(x),
        covariance_type="tied",
        k=2,
        dimension=2,
        log_likelihood=likelihood,
        bic=-2 * likelihood + 8 * np.log(len(x)),
    )
    (tmp_path / "index.json").write_text(json.dumps([record]))
    # Strict defaults and all used-batch checks still reject differing hard labels.
    with pytest.raises(AssertionError):
        audit_candidates(tmp_path)
    with pytest.raises(AssertionError):
        audit_candidates(tmp_path, required_label_arrays={path.name}, label_ambiguities=[])
    ambiguity = []
    assert audit_candidates(tmp_path, required_label_arrays=set(), label_ambiguities=ambiguity) == 1
    assert ambiguity[0]["rows"] == 8 and ambiguity[0]["largest_native_gap"] < 1e-14
    # Even an unused batch must retain its own exact posterior argmax and density.
    values["labels"] = np.zeros(8, dtype=int)
    np.savez(path, **values)
    with pytest.raises(AssertionError):
        audit_candidates(tmp_path, required_label_arrays=set(), label_ambiguities=[])
    values["labels"] = np.ones(8, dtype=int)
    values["membership"] = np.tile([0.49, 0.51], (8, 1))
    np.savez(path, **values)
    with pytest.raises(AssertionError):
        audit_candidates(tmp_path, required_label_arrays=set(), label_ambiguities=[])
    with pytest.raises(ValueError):
        audit_candidates(tmp_path, required_label_arrays=set())
