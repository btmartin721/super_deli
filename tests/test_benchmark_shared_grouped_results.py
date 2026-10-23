"""Portable grouped evidence: fixed design, uncertainty, counts and numerical acceptance."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from super_deli.models.grouped_config import GroupedStudyConfig, study_cases

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/shared-grouped-calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_fixed_design_normalized_settings_and_attempt_counts():
    study, result = read("study.json"), read("summary.json")
    c = GroupedStudyConfig.from_dict(study)
    assert c == GroupedStudyConfig.from_dict(result["study"])
    assert c.random_seed == 20261102 and c.n_units == 64 and c.n_datasets == 25
    assert len(study_cases(c)) == 7 and result["datasets"] == 175
    assert c.calibration.schema_version == 2 and c.calibration.n_simulations == 99
    assert c.calibration.alpha == 0.05 and c.calibration.gradient_tol == 1e-5
    m = c.calibration.model
    assert m.covariance_type == "tied" and m.n_init == 10 and m.max_k == 4
    assert m.n_pcs == 2 and m.min_cluster_size == 5 and m.n_bootstrap == 0
    assert m.random_seed == 71323 and all(v.n_pcs == 2 and v.weight == 1 for v in m.views)
    assert c.calibration.solver.estimator == "conditional_psd_v1"
    assert c.calibration.solver.max_polish_iterations == 2000
    assert result["planned_fits"] == 17500
    data = rows("datasets.csv")
    for key in (
        "planned_fits",
        "attempted_fits",
        "successful_fits",
        "failed_fits",
        "unattempted_fits",
    ):
        assert result[key] == sum(int(row[key]) for row in data)
    assert result["successful_fits"] + result["failed_fits"] + result["unattempted_fits"] == 17500
    execution = read("execution.json")
    assert execution["workers"] == 4 and execution["threads_per_job"] == 1
    assert execution["replacement_datasets"] == execution["replacement_draws"] == 0


def test_public_integrity_private_paths_and_frozen_sources():
    manifest = read("manifest.json")
    assert manifest["local_artifact_id"] == "shared-grouped-null-20261022"
    for name, expected in manifest["published_artifacts"].items():
        assert Path(name).name == name
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest["publication_files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert all(
        "/Users/" not in p.read_text() and "/private/tmp" not in p.read_text()
        for p in RESULTS.iterdir()
    )
    sources = read("runtime_sources.json")
    assert len(sources) == 115 and all(
        not Path(n).is_absolute() and len(h) == 64 for n, h in sources.items()
    )
    assert read("runtime_versions.json")["libraries"]["super-deli"]["version"] == "0.25.0.dev0"


def test_independent_plus_one_and_conditional_binomial_tails():
    data = rows("diagnostics.csv")
    assert len(data) == 175
    for row in data:
        if row["status"] != "available":
            assert row["p_value"] == row["reject"] == ""
            continue
        n, x = int(row["successful_draws"]), int(row["exceedances"])
        assert n == 99 and int(row["failed_draws"]) == 0
        p = float(row["p_value"])
        assert p == (1 + x) / (1 + n)
        assert float(row["failed_p_lower"]) == float(row["failed_p_upper"]) == p
        assert (row["reject"] == "True") == (p <= 0.05)
        low, high = float(row["tail_lower_95"]), float(row["tail_upper_95"])
        if x:
            tail = sum(math.comb(n, j) * low**j * (1 - low) ** (n - j) for j in range(x, n + 1))
            assert tail == pytest.approx(0.025, abs=1e-9)
        else:
            assert low == 0
        if x < n:
            tail = sum(math.comb(n, j) * high**j * (1 - high) ** (n - j) for j in range(x + 1))
            assert tail == pytest.approx(0.025, abs=1e-9)
        else:
            assert high == 1


def test_counts_covariance_acceptance_and_independent_observation_rates():
    result = read("summary.json")
    data, diagnostics = rows("datasets.csv"), rows("diagnostics.csv")
    assert len(data) == len(diagnostics) == 175
    for row in diagnostics:
        if row["null_fit_available"] != "True":
            continue
        counts = [int(row[f"groups_with_{i}_specimens"]) for i in range(1, 5)]
        assert sum(counts) == int(row["n_groups"]) == int(row["n_ecology_rows"]) == 64
        assert sum((i + 1) * n for i, n in enumerate(counts)) == int(row["n_specimens"])
        assert 0 <= int(row["conditional_rank"]) <= 4
        assert float(row["optimality_maximum"]) <= 1e-5
        assert int(row["declared_starts"]) == 4 and 1 <= int(row["accepted_starts"]) <= 4
    for rate in result["rates"]:
        selected = [row for row in data if row["case"] == rate["case"]]
        available = [row for row in selected if row["p_value"]]
        events = sum(float(row["p_value"]) <= 0.05 for row in available)
        assert len(selected) == 25 and {int(row["repetition"]) for row in selected} == set(
            range(25)
        )
        assert rate["n_available"] == len(available) and rate["events"] == events
        assert rate["withheld_outcome_bounds"] == [
            events / 25,
            (events + 25 - len(available)) / 25,
        ]
        n, z = len(available), 1.959963984540054
        if n:
            p = events / n
            center = (p + z * z / (2 * n)) / (1 + z * z / n)
            radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            assert rate["wilson_95"] == pytest.approx(
                [max(0, center - radius), min(1, center + radius)]
            )
        for rank, count in result["conditional_rank_counts"][rate["case"]].items():
            assert count == sum(
                row["conditional_rank"] == rank
                for row in diagnostics
                if row["case"] == rate["case"]
            )
    report = (ROOT / "docs/shared-grouped-calibration-results.md").read_text().lower()
    for phrase in (
        "nonregular",
        "approximate",
        "no general",
        "not independently refitted",
        "unpaired",
        "17,500",
        "stability",
        "counts",
        "fresh",
    ):
        assert phrase in report
