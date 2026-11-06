"""Invented full-lifecycle contracts and independent simulation laws."""

# ruff: noqa: E402

import json
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.benchmarking.separate_gmm_config import CASES
from super_deli.benchmarking.separate_gmm_data import generate, reference


@pytest.mark.parametrize("case", CASES)
def test_raw_laws_match_independent_references(case):
    task = dict(
        case=case,
        n=32,
        data_seed=[114001, 32],
        noise_seed=[114007, 32],
        permutation_seed=[114013, 32],
        ecology_order_seed=[114013, 32, 1],
    )
    a, b = generate(task), reference(task)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-11)


def test_invented_end_to_end_audited_lifecycle(tmp_path):
    from super_deli.benchmarking.separate_gmm_archive import bind
    from super_deli.benchmarking.separate_gmm_config import SharedEncoderStudy
    from super_deli.benchmarking.separate_gmm_fixture import create_fixture
    from super_deli.benchmarking.separate_gmm_study import freeze

    source = tmp_path / "source"
    create_fixture(source)
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering fixture")
    bind(source, root, fixture=True)
    freeze(root, SharedEncoderStudy(fixture=True))
    for action in ("preflight", "audit-preflight", "gate", "main", "audit-main", "report"):
        cmd = [
            sys.executable,
            "-m",
            "super_deli",
            "benchmark-separate-gmm",
            action,
            "--archive-root",
            str(source),
            "--output-dir",
            str(root),
        ]
        if action == "gate":
            cmd += ["--staging-bytes", "0"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        (root / (action + ".stderr.log")).write_text(p.stderr)
        assert p.returncode == 0, (action, p.stderr, p.stdout[-2000:])
        assert not (root / "STOP.json").exists()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["observations"] == 1 and summary["new_pvalues"] == 0
    assert summary["audits"]["preflight"]["verified_optimizer_updates"] == 6000
    assert summary["audits"]["main"]["verified_optimizer_updates"] == 7000
    assert summary["counts"]["main"]["workflows"] == 25
