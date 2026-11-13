"""Invented source classifiers and isolated installed benchmark checks only."""

import json
import os
import subprocess
import sys
from pathlib import Path


def make_source(source):
    from super_deli.benchmarking.local_reference_run import run_unit
    from super_deli.io.config import write_json

    source = Path(source)
    source.mkdir()
    (source / "main").mkdir()
    task = dict(
        key="invented_observed_reference",
        case="linear_standard",
        n=32,
        fixture=True,
        replicate=0,
        seed_path=[871431, 751],
    )
    result = run_unit(
        dict(
            kind="fresh",
            task=task,
            phase_root=str(source / "main"),
            stop_file=str(source / "STOP.json"),
        )
    )
    if result["failed_points"]:
        raise ValueError("Invented source reference failure")
    write_json(
        dict(kind="invented_observed_feasibility_source_v1", observations=["main/" + task["key"]]),
        source / "fixture.json",
    )
    return source


def smoke(output_dir, *, workers=1):
    from super_deli.benchmarking.observed_feasibility_config import ObservedFeasibilityStudy
    from super_deli.io.config import write_json

    out = Path(output_dir).resolve()
    out.mkdir()
    source = make_source(out / "invented-source")
    root = out / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text(
        "Invented observed-feature installed fixture; not scientific evidence.\n"
    )
    config = out / "study.json"
    write_json(ObservedFeasibilityStudy(fixture=True).to_dict(), config)
    env = {
        **os.environ,
        **{
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    prefix = [sys.executable, "-m", "super_deli", "benchmark-observed-feasibility"]
    common = ["--archive-root", str(source), "--output-dir", str(root), "--fixture"]
    commands = [
        ["bind"],
        ["freeze", "--study-config", str(config)],
        ["run", "--phase", "preflight", "--workers", str(workers)],
        ["audit", "--phase", "preflight"],
        ["gate", "--staging-bytes", "0"],
        ["run", "--phase", "main", "--workers", str(workers)],
        ["audit", "--phase", "main"],
        ["report"],
    ]
    summaries = []
    for command in commands:
        completed = subprocess.run(
            prefix + command + common, cwd=out, env=env, capture_output=True, text=True
        )
        (out / (command[0] + "_" + str(len(summaries)) + ".stdout.log")).write_text(
            completed.stdout
        )
        (out / (command[0] + "_" + str(len(summaries)) + ".stderr.log")).write_text(
            completed.stderr
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:])
        summaries.append(json.loads(completed.stdout))
    return dict(
        passed=True,
        invented_only=True,
        scientific_neural_training=0,
        isolated_no_torch_science=True,
        study=summaries[-1],
    )
