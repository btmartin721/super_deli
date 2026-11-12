"""Independent nonlinear equations, fixed fit budgets and sealed fixture integration."""

# ruff: noqa: E402
import json
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")

from super_deli.benchmarking.residual_audit import reference_data
from super_deli.benchmarking.residual_config import CASES, ResidualStudy, counts, tasks
from super_deli.benchmarking.residual_data import generate


def test_approved_counts_and_disjoint_data_paths():
    c = ResidualStudy()
    assert len(tasks(c, "main")) == 80
    assert len(tasks(c, "preflight")) == 16
    assert counts(c, "main")["workflows"] == 580
    assert counts(c, "preflight")["workflows"] == 120
    assert counts(c, "main")["neural_fits"] == 678
    assert counts(c, "preflight")["neural_fits"] == 138
    assert counts(c, "main")["candidate_searches"] == 332
    assert counts(c, "preflight")["candidate_searches"] == 64
    paths = [tuple(t["seed_path"]) for phase in ("preflight", "main") for t in tasks(c, phase)]
    assert len(paths) == len(set(paths))


@pytest.mark.parametrize("case", CASES)
def test_independent_nonlinear_measurement_equations(case):
    # Explicit engineering seeds, disjoint from all registered research seeds.
    task = dict(
        case=case,
        n=16,
        fixture=True,
        seed_path=[834917, CASES.index(case)],
        labeled=case not in ("gaussian", "gradient"),
    )
    cohorts = (
        ["train", "iid"]
        + (["shifted"] if case == "confounded_size" else [])
        + (["unfamiliar"] if task["labeled"] else [])
    )
    for cohort in cohorts:
        a, b = generate(task, cohort), reference_data(task, cohort)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)
        if case in ("allometric", "combined", "confounded_size"):
            assert np.all(a["morphology"] > 0)
        if case in ("saturating", "combined"):
            assert np.all((a["ecology"] > 15) & (a["ecology"] < 25))
        if cohort == "unfamiliar":
            assert np.unique(a["truth"]).tolist() == [4]


def test_complete_sealed_fixture(tmp_path):
    pytest.importorskip("torch")
    code = (
        "from super_deli.benchmarking.residual_fixture import smoke;"
        "import sys,json;print(json.dumps(smoke(sys.argv[1],workers=2)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "fixture")], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-4000:]
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["observations"] == 1
    assert summary["workflows"] == 12
    assert summary["verified_optimizer_updates"] == 6000
    assert summary["failed_workflows"] == 0
