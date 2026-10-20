"""Hand-computed label-invariant metrics and independent-dataset MCSE denominators."""

# ruff: noqa: E402

import json
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.config import BenchmarkConfig
from super_deli.benchmarking.metrics import score_run, summarize_runs, summarize_values
from super_deli.io.config import write_json


def artifacts(path, *, wrong=False, failed=False):
    path.mkdir()
    units = ["a", "b", "c", "d"]
    labels = np.array([0, 0, 0, 0] if wrong else [1, 1, 0, 0])
    same = np.equal.outer(labels, labels).astype(float)
    oob = np.full((4, 4), np.nan)
    oob[0, 1] = oob[1, 0] = 1
    requests = np.zeros((4, 4), dtype=int)
    requests[0, 1] = requests[1, 0] = 1
    write_json(
        {
            "unit_ids": units,
            "selected_k": len(set(labels)),
            "point_seed": 71,
            "bootstrap": {"requested": 2, "successful": 1 if failed else 2},
        },
        path / "summary.json",
    )
    np.savez(path / "mixture.npz", labels=labels)
    np.savez(
        path / "stability.npz",
        coassignment=np.full_like(same, np.nan) if failed else same,
        coassignment_lower=same / 2 if failed else same,
        coassignment_upper=(same + 1) / 2 if failed else same,
        oob_coassignment=oob,
        oob_requested=requests,
        oob_coassignment_lower=oob,
        oob_coassignment_upper=oob,
    )
    write_json(
        [
            {
                "unit_ids": [u for u, c in zip(units, labels, strict=True) if c == k],
                "frequency": None if failed else 1.0,
            }
            for k in sorted(set(labels))
        ],
        path / "clusters.json",
    )
    truth = {u: {"primary": i // 2, "alternative": None} for i, u in enumerate(units)}
    return score_run(path, truth)


def test_perfect_partition_off_diagonal_and_sparse_oob(tmp_path):
    score = artifacts(tmp_path / "fit")
    metrics = score["metrics"]
    assert metrics["ari"] == metrics["k_correct"] == 1
    assert metrics["pair_error"] == metrics["coassignment_squared_error"] == 0
    assert metrics["oob_pair_coverage"] == pytest.approx(1 / 6)
    assert metrics["oob_coassignment_squared_error"] == 0
    assert metrics["oob_coassignment_between_mean"] is None
    assert metrics["stable_cluster_exact_truth_fraction"] == 1
    assert sum(b["pairs"] for b in score["coassignment_bins"] if b["scope"] == "all_pairs") == 6


def test_stable_wrong_cluster_is_not_correctness(tmp_path):
    score = artifacts(tmp_path / "fit", wrong=True)
    metrics = score["metrics"]
    assert metrics["ari"] == 0 and metrics["k_correct"] == 0
    assert metrics["coassignment_squared_error"] == pytest.approx(4 / 6)
    assert metrics["pair_precision"] == pytest.approx(2 / 6)
    assert metrics["pair_recall"] == 1
    assert metrics["stable_cluster_exact_truth_fraction"] == 0
    assert score["stable_wrong_cluster_count"] == 1


def test_failed_bootstrap_preserves_unknowns_and_loss_bounds(tmp_path):
    score = artifacts(tmp_path / "fit", failed=True)
    metrics = score["metrics"]
    assert metrics["coassignment_squared_error"] is None
    assert metrics["coassignment_loss_lower"] == 0
    assert metrics["coassignment_loss_upper"] == pytest.approx(0.25)
    assert metrics["bootstrap_success_fraction"] == 0.5
    assert metrics["stable_cluster_exact_truth_fraction"] is None


def test_dataset_mcse_and_missingness():
    result = summarize_values([0, 1, None], 3)
    assert result["mean"] == result["median"] == 0.5
    assert result["mcse"] == 0.5 and result["n_available"] == 2 and result["n_missing"] == 1
    assert summarize_values([1], 3)["mcse"] is None
    assert summarize_values([], 3)["mean"] is None
    with pytest.raises(ValueError):
        summarize_values([float("nan")], 1)


def test_paired_differences_and_all_failed_arm_keep_denominators():
    options = json.loads(
        (Path(__file__).resolve().parents[1] / "examples/benchmarking/smoke.json").read_text()
    )
    config = BenchmarkConfig.from_dict(options)
    config = replace(config, scenarios=config.scenarios[:1], arms=config.arms[:2], n_datasets=3)
    records = []
    for repetition in range(3):
        for arm, value in (
            ("full", 0.9 - repetition * 0.1),
            ("genome_environment", 0.4 - repetition * 0.3),
        ):
            record = {"scenario": config.scenarios[0].name, "arm": arm, "repetition": repetition}
            record.update(
                status="complete",
                evaluation={
                    "metrics": {"ari": value},
                    "selected_k": 4,
                    "bootstrap": {"successful": 10},
                },
            )
            if repetition == 2:
                record.pop("evaluation")
                record["status"] = "failed"
            records.append(record)
    summary = summarize_runs(records, config)
    delta = summary["paired_comparisons"][0]["metrics"]["ari"]
    assert delta["mean"] == pytest.approx(0.6)
    assert delta["mcse"] == pytest.approx(0.1)
    assert delta["n_available"] == 2 and delta["n_missing"] == 1
    for record in records:
        record.pop("evaluation", None)
        record["status"] = "failed"
    summary = summarize_runs(records, config)
    assert all(g["metrics"]["ari"]["n_missing"] == 3 for g in summary["groups"])
    assert all(g["bootstrap_requested_including_failed_points"] == 30 for g in summary["groups"])
