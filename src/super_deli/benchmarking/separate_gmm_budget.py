"""Measured per-N/representation point/refit costs with serial-only projections."""

import shutil

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.benchmarking.separate_gmm_config import ARMS, NATIVE, tasks
from super_deli.benchmarking.separate_gmm_observer import evaluation_path
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def project(root, frozen, config, audit, execution, design, staging_bytes):
    times = {}
    sizes = {}
    costs = {}
    overhead = {}
    source_seconds = {}
    source_bytes = {}

    def maximum(d, k, v):
        d[k] = max(d.get(k, 0), v)

    grid = {t["key"]: t for t in tasks(config, "preflight")}
    for job in execution["records"]:
        task = grid[job["key"]]
        n = task["n"]
        folder = root / "preflight" / job["key"]
        rows = read_json(folder / "fits.json")
        maximum(source_seconds, n, max(job["seconds"] - sum(r["seconds"] for r in rows), 0))
        maximum(
            source_bytes,
            n,
            retained_bytes(folder) - sum(retained_bytes(folder / r["arm"]) for r in rows),
        )
        for row in rows:
            arm = row["arm"]
            out = folder / arm
            draws = row["draws_requested"]
            maximum(sizes, (n, arm, bool(draws)), retained_bytes(out))
            if row["status"] != "available":
                continue
            if arm not in NATIVE:
                maximum(times, (n, arm, "point"), row["seconds"])
                continue
            consumed = 0
            for i in range(1 + draws):
                point = read_json(evaluation_path(out, i, "coordinates") / "execution.json")
                require(
                    point["status"] == "available", "Complete representative preflight encoding"
                )
                value = point["seconds"]
                for head in ("hdbscan", "gmm_full"):
                    h = read_json(evaluation_path(out, i, head) / "execution.json")
                    require(
                        h["status"] == "available", "Complete representative preflight clustering"
                    )
                    value += h["seconds"]
                consumed += value
                maximum(times, (n, arm, "point" if i == 0 else "refit"), value)
            maximum(overhead, (n, arm), max(row["seconds"] - consumed, 0) / (1 + draws))
    for record in audit["records"]:
        maximum(costs, (record["n"], record["arm"], bool(record["draws"])), record["seconds"])
    extra = max(audit["seconds"] - sum(r["seconds"] for r in audit["records"]), 0)
    serial = 0.0
    bytecount = 0
    audittime = 0.0
    missing = []

    def charge(n, arm, draws):
        nonlocal serial, bytecount, audittime
        timekey = (n, arm, "point")
        sizekey = (n, arm, bool(draws))
        if config.fixture and sizekey not in sizes:
            sizekey = (n, arm, True)
        if timekey not in times:
            missing.append([*timekey, "seconds"])
        if sizekey not in sizes or sizekey not in costs:
            missing.append([*sizekey, "storage/audit"])
        serial += times.get(timekey, 0) + (1 + draws) * overhead.get((n, arm), 0)
        if draws:
            if (n, arm, "refit") not in times:
                missing.append([n, arm, "refit_seconds"])
            serial += draws * times.get((n, arm, "refit"), 0)
        bytecount += sizes.get(sizekey, 0)
        audittime += costs.get(sizekey, 0)

    main = tasks(config, "main")
    binding = read_json(root / "replay-binding.json")
    for task in main:
        n = task["n"]
        serial += source_seconds[n]
        bytecount += source_bytes[n] + 8192
        for arm in ARMS:
            charge(n, arm, task["draws"] if arm in NATIVE else 0)
    for e in binding["replays"]:
        n = e["task"]["n"]
        serial += source_seconds[n]
        bytecount += source_bytes[n] + 8192
        charge(n, "separate_" + str(e["seed"]), 0)
    audittime += extra * (len(main) + len(binding["replays"])) / len(grid)
    seconds = 1.25 * (serial + audittime)
    projected = int(retained_bytes(root) + 2 * staging_bytes + 1.25 * bytecount + 128 * 1024**2)
    free = shutil.disk_usage(root).free
    result = dict(
        passed=not missing
        and seconds < config.max_projected_seconds
        and (projected <= config.artifact_cap_bytes)
        and (free >= config.min_free_bytes),
        projected_bytes=projected,
        projected_main_audit_seconds=seconds,
        staging_bytes=staging_bytes,
        staging_copies_counted=2,
        free_bytes=free,
        missing_timing_storage_strata=missing,
        accounting=(
            "Per-N/representation maxima, separated point/refit durations; "
            "all head/workflow/source/audit overhead; 25% allowance, 128MiB reserve; "
            "no parallel speed credit"
        ),
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        audit_sha256=fingerprint_file(root / "preflight-audit.json").sha256,
        execution_sha256=fingerprint_file(root / "preflight-execution.json").sha256,
    )
    write_json(result, root / "storage-runtime-gate.json")
    if not result["passed"]:
        write_json(dict(reason="Measured shared-encoder gate", **result), root / "STOP.json")
    return result
