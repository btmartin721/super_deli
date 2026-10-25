"""Exclusive bounded study execution with atomic progress and retained failure evidence."""

import json
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from super_deli.benchmarking.local_archives import write_progress
from super_deli.benchmarking.predictive_study import retained_bytes
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json

COUNTS = ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")


def run_bounded_study(config, *, output_dir, worker, workers=1, preflight=None):
    c = type(config).from_dict(config.to_dict())
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    output = Path(output_dir).resolve()
    if any(cell.n_datasets > 1 for cell in c.cells):
        if preflight is None:
            raise ValueError("Measured audited storage preflight required")
        gate = read_json(preflight)
        root = Path(gate["artifact_root"]).resolve()
        if (
            gate["status"] != "passed"
            or gate["configuration_sha256"] != configuration_hash(c.to_dict())
            or output == root
            or not output.is_relative_to(root)
        ):
            raise ValueError("Storage gate/configuration/output mismatch")
    else:
        root = output.parent
    if shutil.disk_usage(output.parent).free < c.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked free space")
    output.mkdir(parents=True, exist_ok=False)
    write_json(c.to_dict(), output / "configuration.json")
    plan = c.plan()
    write_json(plan, output / "plan.json")
    rows = []

    def report(status):
        result = {
            "status": status,
            "observations": len(rows),
            "requested": len(plan),
            "planned_fits": sum(r["planned_fits"] for r in plan),
            **{k: sum(r[k] for r in rows) for k in COUNTS},
            **{
                k: sum(r[k] for r in rows)
                for k in ("failed_draws", "successful_draws", "unattempted_draws")
            },
        }
        write_progress(result, output / "progress.json")
        return result

    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (output / "observations.jsonl").open("x") as ledger,
    ):
        tasks = iter(plan)
        pending = set()

        def submit():
            row = next(tasks, None)
            if row is not None:
                pending.add(pool.submit(worker, (str(output), c.to_dict(), row)))

        for _ in range(workers):
            submit()
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    rows.append(row)
                    ledger.write(json.dumps(row, sort_keys=True) + "\n")
                    ledger.flush()
                    report("running")
                    if (
                        retained_bytes(root) > c.max_artifact_bytes
                        or shutil.disk_usage(root).free < c.min_free_bytes
                    ):
                        raise RuntimeError(
                            "Runtime storage guard failed; preserve partial evidence"
                        )
                    submit()
        except BaseException as exc:
            for future in pending:
                future.cancel()
            write_json(
                {**report("interrupted"), "error": str(exc)}, output / "execution_failure.json"
            )
            raise
    result = report("complete")
    write_json(result, output / "summary.json")
    return result
