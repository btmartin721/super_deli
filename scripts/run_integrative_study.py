"""Run unchanged scenario shards using a chosen, preferably frozen, Python runtime."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from super_deli.benchmarking.config import BenchmarkConfig
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json


def scenario_configs(config):
    """Keep every study setting and repetition index; only partition scenario lists."""
    config = BenchmarkConfig.from_dict(config.to_dict())
    return {
        scenario.name: {**config.to_dict(), "scenarios": [config.to_dict()["scenarios"][i]]}
        for i, scenario in enumerate(config.scenarios)
    }


def run_study(config, *, output_dir, python=sys.executable, workers=4):
    config = BenchmarkConfig.from_dict(config.to_dict())
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("workers must be an integer in [1,64]")
    if workers > 1 and config.model.n_threads != 1:
        raise ValueError("Concurrent scenarios require one numerical thread per fit")
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    configurations = destination / "configurations"
    configurations.mkdir()
    shards = scenario_configs(config)
    write_json(config.to_dict(), destination / "study.json")
    write_json(
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "workers": workers,
            "python": str(Path(python).absolute()),
            "host_platform": platform.platform(),
            "configuration_sha256": configuration_hash(config.to_dict()),
            "script_sha256": fingerprint_file(__file__).sha256,
            "partition": "Scenario only; preserve all repetition indices and seed keys",
            "thread_environment": {
                k: "1"
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        destination / "execution.json",
    )
    for name, declaration in shards.items():
        write_json(declaration, configurations / f"{name}.json")

    def execute(name):
        job = destination / name
        job.mkdir()
        if shutil.disk_usage(destination).free < 8 * 1024**3:
            return {"scenario": name, "returncode": None, "error": "Less than 8 GiB free"}
        env = os.environ.copy()
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = "1"
        env["MPLCONFIGDIR"] = str(job / "cache/matplotlib")
        env["NUMBA_CACHE_DIR"] = str(job / "cache/numba")
        command = [
            str(python),
            "-m",
            "super_deli",
            "benchmark-integrative",
            str(configurations / f"{name}.json"),
            "--output-dir",
            str(job / "runs"),
        ]
        started = perf_counter()
        print(f"{datetime.now(UTC).isoformat()} start {name}", flush=True)
        with (job / "stdout.json").open("x") as out, (job / "stderr.log").open("x") as err:
            result = subprocess.run(command, cwd=job, env=env, stdout=out, stderr=err, check=False)
        record = {
            "scenario": name,
            "returncode": result.returncode,
            "elapsed_seconds": perf_counter() - started,
        }
        if result.returncode == 0:
            summary = read_json(job / "runs/summary.json")
            record.update(
                point_fits=summary["attempted"],
                point_failed=summary["point_failed"],
                bootstrap_incomplete_runs=summary["bootstrap_incomplete_runs"],
                summary_sha256=fingerprint_file(job / "runs/summary.json").sha256,
            )
        write_json(record, job / "outcome.json")
        print(f"{datetime.now(UTC).isoformat()} finish {name}: {result.returncode}", flush=True)
        return record

    outcomes = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(execute, name): name for name in shards}
            for future in as_completed(futures):
                outcomes.append(future.result())
        order = {name: i for i, name in enumerate(shards)}
        outcomes.sort(key=lambda item: order[item["scenario"]])
        result = {
            "schema_version": 1,
            "status": "complete" if all(r["returncode"] == 0 for r in outcomes) else "failed",
            "outcomes": outcomes,
        }
        write_json(result, destination / "execution-outcomes.json")
        if result["status"] != "complete":
            raise RuntimeError("Scenario execution failed; preserve outputs and inspect outcomes")
        return result
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    run_study(
        BenchmarkConfig.from_dict(read_json(args.config)),
        output_dir=args.output_dir,
        python=args.python,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
