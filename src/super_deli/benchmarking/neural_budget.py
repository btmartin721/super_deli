"""Measured per-arm projection for workflows with unequal numbers of encoders."""

import math
import shutil

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def project_gate(root, frozen, config, audit, execution, design, staging_bytes):
    timings = {}
    overheads = {}
    sizes = {}
    source_seconds = {}
    source_bytes = {}
    training = design["training_counts"]

    def maximum(mapping, key, value):
        mapping[key] = max(mapping.get(key, 0), value)

    grid = {t["key"]: t for t in design["tasks"](config, "preflight")}
    for job in execution["records"]:
        task = grid[job["key"]]
        n = task["n"]
        folder = root / "preflight" / job["key"]
        rows = read_json(folder / "fits.json")
        maximum(source_seconds, n, max(job["seconds"] - sum(r.get("seconds", 0) for r in rows), 0))
        maximum(
            source_bytes,
            n,
            retained_bytes(folder) - sum(retained_bytes(folder / r["arm"]) for r in rows),
        )
        for row in rows:
            arm = row["arm"]
            out = folder / arm
            draws = row["draws_requested"]
            k = training[arm]
            maximum(sizes, (n, arm, bool(draws)), retained_bytes(out))
            if row["status"] != "available":
                continue
            if not k:
                maximum(timings, (n, arm, "point"), row["seconds"])
                maximum(overheads, (n, arm), 0)
                continue
            events = read_json(out / "optimizer-checks.json")
            require(
                all(
                    isinstance(e.get("seconds"), (int, float))
                    and math.isfinite(e["seconds"])
                    and e["seconds"] >= 0
                    for e in events
                ),
                "Every attempted training has a measured duration",
            )
            maximum(overheads, (n, arm), max(row["seconds"] - sum(e["seconds"] for e in events), 0))
            if len(events) >= k and all(e["status"] == "complete" for e in events[:k]):
                maximum(timings, (n, arm, "point"), sum(e["seconds"] for e in events[:k]))
            records = read_json(out / "subsamples.json")
            if (
                draws
                and all(r["status"] == "complete" for r in records)
                and len(events) == k * (draws + 1)
            ):
                for i in range(1, draws + 1):
                    maximum(
                        timings,
                        (n, arm, "refit"),
                        sum(e["seconds"] for e in events[k * i : k * (i + 1)]),
                    )
    missing = []
    serial = 0.0
    projected = 0

    def point(n, arm, draws):
        nonlocal serial, projected
        key = (n, arm, "point")
        if key not in timings:
            missing.append([n, arm, "point_time"])
        serial += timings.get(key, 0) + overheads.get((n, arm), 0)
        if draws:
            key = (n, arm, "refit")
            if key not in timings:
                missing.append([n, arm, "refit_time"])
            serial += draws * timings.get(key, 0)
        size_key = (n, arm, bool(draws))
        if size_key not in sizes and config.fixture:
            size_key = (n, arm, True)
        if size_key not in sizes:
            missing.append([n, arm, "native" if draws else "point_bytes"])
        projected += sizes.get(size_key, 0)

    for task in design["tasks"](config, "main"):
        n = task["n"]
        serial += source_seconds[n]
        projected += source_bytes[n] + 4096
        for arm in design["arms"]:
            point(n, arm, task["draws"] if arm in design["native_arms"] else 0)
    binding = read_json(root / "replay-binding.json")
    for entry in binding["replays"]:
        n = entry["task"]["n"]
        serial += source_seconds[n]
        projected += 4096
        point(n, design["replay_prefix"] + str(entry["seed"]), 0)
    # Charge whole-observation audit cost to each replay as well. Times are sums
    # of measured training groups plus complete per-workflow overhead, never
    # divided by worker count. Native overhead includes its measured refits.
    ratio = (frozen["counts"]["main"]["observations"] + len(binding["replays"])) / frozen["counts"][
        "preflight"
    ]["observations"]
    seconds = 1.25 * (serial + audit["seconds"] * ratio)
    projected = int(retained_bytes(root) + 2 * staging_bytes + 1.25 * projected + 64 * 1024**2)
    free = shutil.disk_usage(root).free
    result = {
        "passed": not missing
        and seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        "projected_bytes": projected,
        "projected_main_audit_seconds": seconds,
        "staging_bytes": staging_bytes,
        "staging_copies_counted": 2,
        "free_bytes": free,
        "missing_timing_storage_strata": missing,
        "accounting": (
            "per-N/arm maxima; explicit encoder groups and native refits; full "
            "workflow/source overhead; 25% allowance; no parallel speed credit"
        ),
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
        "audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
        "execution_sha256": fingerprint_file(root / "preflight-execution.json").sha256,
    }
    write_json(result, root / "storage-runtime-gate.json")
    if not result["passed"]:
        write_json({"reason": "Measured preflight resource gate", **result}, root / "STOP.json")
    return result
