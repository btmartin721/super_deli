"""Keep the published results snapshot, code provenance and denominators consistent."""

import csv
import hashlib
import json
import re
from pathlib import Path

from super_deli import __version__
from super_deli.benchmarking.config import BenchmarkConfig
from super_deli.core.provenance import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/benchmarking/study-results"


def test_published_snapshot_artifact_and_code_hashes():
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    for relative, digest in manifest["artifacts"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    for key, relative in (
        ("execution_script_sha256", "scripts/run_integrative_study.py"),
        ("analysis_module_sha256", "src/super_deli/benchmarking/study_analysis.py"),
        ("plot_script_sha256", "scripts/plot_integrative_study.py"),
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", manifest[key])
        # Later reporting versions may evolve while this historical snapshot stays frozen.
        if __version__ == manifest["reporting_checkpoint_version"]:
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == manifest[key]
    config = BenchmarkConfig.from_dict(json.loads((RESULTS / "study.json").read_text()))
    assert configuration_hash(config.to_dict()) == manifest["configuration_sha256"]


def test_published_snapshot_requested_and_attempted_denominators():
    with (RESULTS / "datasets.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2400
    assert len({(r["scenario"], r["arm"], r["repetition"]) for r in rows}) == 2400
    assert sum(r["status"] == "failed" for r in rows) == 2
    assert sum(int(r["bootstrap_requested"]) for r in rows if r["bootstrap_requested"]) == 239800
    assert sum(int(r["bootstrap_successful"]) for r in rows if r["bootstrap_successful"]) == 239800
    assert sum(int(r["bootstrap_failed"]) for r in rows if r["bootstrap_failed"]) == 0
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert manifest["bootstrap"]["expected_bootstrap_draws"] == 240000
    assert manifest["bootstrap"]["draws_not_logged_after_failed_points"] == 200


def test_published_snapshot_keeps_conditional_and_end_to_end_recovery_separate():
    with (RESULTS / "rates.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for arm in ("full", "genome_environment"):
        subset = {
            r["event"]: r for r in rows if r["scenario"] == "missing_cells" and r["arm"] == arm
        }
        assert float(subset["k_correct"]["rate"]) == 1
        assert int(subset["k_correct"]["n_available"]) == 99
        assert int(subset["k_correct"]["n_missing"]) == 1
        assert float(subset["fit_and_k_correct"]["rate"]) == 0.99
        assert int(subset["fit_and_k_correct"]["n_available"]) == 100
