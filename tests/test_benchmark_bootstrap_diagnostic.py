# ruff: noqa: E402
"""Numerical diagnostic smoke and independent corruption checks."""

from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")


from super_deli.benchmarking.bootstrap_diagnostic_audit import audit_metrics, audit_oracles
from super_deli.benchmarking.bootstrap_diagnostic_metrics import (
    axis_information,
    diagnose_workflow,
    matched_changes,
    truth_diagnostics,
)
from super_deli.benchmarking.bootstrap_diagnostic_study import execute, observation
from super_deli.benchmarking.local_archives import materialize
from super_deli.benchmarking.predictive_study import simulate
from super_deli.io.config import read_json, write_json
from super_deli.models.bootstrap_diagnostic_config import BootstrapDiagnosticConfig
from super_deli.models.predictive_study_config import PredictiveCell

ROOT = Path(__file__).resolve().parents[1]


def test_axis_degenerate_and_label_invariance():
    labels = np.repeat(np.arange(4), 4)
    matrix = np.column_stack((labels // 2, labels % 2))
    assert axis_information(matrix, labels) == pytest.approx(dict(a=1, b=1))
    assert axis_information(matrix[:4], labels[:4]) == dict(a=None, b=None)
    assert axis_information(matrix, None) is None
    assert matched_changes([0, 0, 1, 1], [5, 5, 3, 3]) == 0
    assert matched_changes([0, 0, 1, 1], [5, 3, 3, 3]) == 1


def test_partial_failure_retains_denominators(tmp_path):
    target = tmp_path / "failure"
    (target / "full").mkdir(parents=True)
    write_json(dict(status="failed", error="retained"), target / "full/record.json")

    def fail():
        raise ValueError("stop without replacement")

    outcome = execute(fail, target)
    assert outcome["selection"]["screened_k"] is None
    assert outcome["attempted_fits"] == outcome["failed_fits"] == 1
    assert outcome["unattempted_fits"] == 5


def test_paired_observation_audits_and_oracle_original_ids(tmp_path):
    raw = read_json(ROOT / "examples/bootstrap-instability/smoke.json")
    c = BootstrapDiagnosticConfig.from_dict(raw)
    c = replace(c, n_draws=1, extended_draws=1, cells=(c.cells[0],), min_free_bytes=1024**2)
    declaration = c.plan()[0]
    root = tmp_path / "study"
    root.mkdir()
    row = observation((str(root), c.to_dict(), declaration))
    assert row["successful_fits"] == 30 and row["successful_oracles"] == 12
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    values, layout, truth = simulate(cell, cell.n_units, declaration["data_seed"])
    with materialize(root / row["archive"], row, scratch=tmp_path / "scratch") as base:
        assignments = read_json(base / "fold_assignments.json")
        assert audit_oracles(base, truth, c.model, assignments) == 12
        for item in read_json(base / "workflows.json"):
            name = "point" if item["replicate"] == -1 else "draw_0000"
            directory = base / item["prefix"]
            model = (
                replace(c.model, n_init=c.extended_starts) if item["arm"] == "extended" else c.model
            )
            report = read_json(base / "diagnostics" / item["arm"] / (name + ".json"))
            audit_metrics(directory, values, layout, item["indices"], model, truth, report)
        # A duplicated support of two IDs/group cannot satisfy five distinct IDs.
        indices = np.repeat(np.concatenate([np.flatnonzero(truth == k)[:2] for k in range(4)]), 8)
        scores = np.column_stack((truth[indices] // 2, truth[indices] % 2)) + np.random.default_rng(
            2
        ).normal(0, 0.01, (len(indices), 2))
        fake = tmp_path / "fake/full/run"
        fake.mkdir(parents=True)
        np.savez(fake / "gmm_candidates.npz", scores=scores)
        result = truth_diagnostics(
            tmp_path / "fake", truth, indices, c.model, [], tmp_path / "oracle"
        )
        assert result[0]["status"] == "complete"
        assert not result[0]["eligible_original_ids"]
        assert sum(result[0]["distinct_unit_sizes"]) == 8
        assert all(r["status"] == "unattempted" for r in result[1:])
        corrupted = diagnose_workflow(
            base / "baseline/point", values, layout, list(range(cell.n_units)), c.model, truth
        )
        corrupted["gate"]["candidates"][-1]["full_distinct_sizes"] = [999] * 4
        with pytest.raises(ValueError, match="Size diagnostic"):
            audit_metrics(
                base / "baseline/point",
                values,
                layout,
                list(range(cell.n_units)),
                c.model,
                truth,
                corrupted,
            )
