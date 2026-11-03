"""Exclusive-output, source-bound archival experiment and measured resource gate."""

import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_moons_study import retained_bytes, runtime, source_hashes
from super_deli.benchmarking.view_signal_archive import check_binding, file_hashes
from super_deli.benchmarking.view_signal_config import REPRESENTATIONS, ViewSignalConfig
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def selected(binding, phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Unknown diagnostic phase")
    return [e for e in binding["observations"] if e["phase"] == phase]


def guard(root, config):
    require(shutil.disk_usage(root).free >= config.min_free_bytes, "Free-space reserve failed")
    require(retained_bytes(root) <= config.artifact_cap_bytes, "Evidence cap exceeded")


def freeze(root, config):
    root = Path(root)
    binding = read_json(root / "archive-binding.json")
    require(
        binding["passed"] and config.fixture == (binding["kind"] == "invented_untrained_fixture"),
        "Source/configuration kind",
    )
    counts = {p: len(selected(binding, p)) * len(REPRESENTATIONS) for p in ("preflight", "main")}
    require(all(counts.values()), "Both diagnostic phases require observations")
    require(
        config.fixture or counts == {"preflight": 126, "main": 1260}, "Frozen evaluation budget"
    )
    guard(root, config)
    package = Path(__file__).parents[1]
    with tarfile.open(root / "frozen-sources.tar.gz", "x:gz") as archive:
        for name in source_hashes():
            archive.add(package / name, arcname=name)
    frozen = {
        "schema_version": 1,
        "config": config.to_dict(),
        "source_sha256": source_hashes(),
        "runtime": runtime(),
        "representations": list(REPRESENTATIONS),
        "counts": counts,
        "binding_sha256": fingerprint_file(root / "archive-binding.json").sha256,
        "plan_sha256": fingerprint_file(root / "approved-plan.md").sha256,
        "research_neural_training": 0,
        "research_gmm_refits": 0,
        "new_datasets": 0,
    }
    write_json(frozen, root / "frozen.json")
    return {"frozen": True, "counts": counts}


def check_frozen(root):
    root = Path(root)
    frozen = read_json(root / "frozen.json")
    require(
        frozen["source_sha256"] == source_hashes() and frozen["runtime"] == runtime(),
        "Source/runtime changed",
    )
    require(
        frozen["binding_sha256"] == fingerprint_file(root / "archive-binding.json").sha256
        and frozen["plan_sha256"] == fingerprint_file(root / "approved-plan.md").sha256,
        "Evidence binding changed",
    )
    return frozen, ViewSignalConfig.from_dict(frozen["config"])


def run_job(job):
    import numpy as np

    from super_deli.benchmarking.view_signal_coordinates import load_context, native_arm
    from super_deli.benchmarking.view_signal_fit import fit_coordinates

    start = time.monotonic()
    entry = job["entry"]
    out = Path(job["out"]) / Path(entry["path"]).name
    out.mkdir(exist_ok=False)
    write_json(entry, out / "entry.json")
    error, failed, unavailable = None, 0, 0
    try:
        context, decoder = load_context(job["source"], entry)
        with (out / "context.npz").open("xb") as f:
            np.savez_compressed(f, **context)
        write_json(decoder, out / "decoder.json")
    except Exception as exc:
        error = str(exc)
        write_json({"error": error}, out / "context-failure.json")
    for name in REPRESENTATIONS:
        target = out / name
        target.mkdir()
        try:
            if error is not None:
                raise ValueError(error)
            expected = (
                (context[name + "_labels"], context[name + "_strength"])
                if native_arm(name)
                else None
            )
            record, data = fit_coordinates(context[name], context, expected=expected)
            with (target / "arrays.npz").open("xb") as f:
                np.savez_compressed(f, **data)
            unavailable += record["hierarchy_status"] != "available"
        except Exception as exc:
            record = {"status": "failed", "error": str(exc)}
            failed += 1
        write_json(record, target / "result.json")
    write_json(file_hashes(out), out / "checksums.json")
    return {
        "key": Path(entry["path"]).name,
        "attempted": len(REPRESENTATIONS),
        "failed": failed,
        "unavailable": unavailable,
        "seconds": time.monotonic() - start,
    }


def run_phase(root, source, phase, *, workers=1):
    root, source = Path(root).resolve(), Path(source).resolve()
    frozen, config = check_frozen(root)
    require(type(workers) is int and 1 <= workers <= config.max_workers, "Worker limit")
    binding = read_json(root / "archive-binding.json")
    check_binding(source, binding)
    if phase == "main":
        gate_record = read_json(root / "storage-runtime-gate.json")
        require(
            gate_record["passed"]
            and gate_record["freeze_sha256"] == fingerprint_file(root / "frozen.json").sha256
            and gate_record["audit_sha256"]
            == fingerprint_file(root / "preflight-audit.json").sha256,
            "Bound passed preflight gate required",
        )
    entries = selected(binding, phase)
    guard(root, config)
    out = root / phase
    out.mkdir(exist_ok=False)
    jobs = [{"entry": e, "source": str(source), "out": str(out)} for e in entries]
    if workers == 1:
        records = [run_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(run_job, jobs))
    require(sum(r["attempted"] for r in records) == frozen["counts"][phase], "Execution budget")
    result = {
        "phase": phase,
        "workers": workers,
        "records": records,
        "attempted": sum(r["attempted"] for r in records),
        "failed": sum(r["failed"] for r in records),
        "unavailable": sum(r["unavailable"] for r in records),
        "serial_job_seconds": sum(r["seconds"] for r in records),
    }
    write_json(result, root / (phase + "-execution.json"))
    check_binding(source, binding)
    guard(root, config)
    return {k: v for k, v in result.items() if k != "records"}


def audit_phase(root, source, phase):
    from threadpoolctl import threadpool_limits

    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.benchmarking.view_signal_audit import audit_context, audit_fit
    from super_deli.benchmarking.view_signal_coordinates import native_arm

    root, source = Path(root).resolve(), Path(source).resolve()
    frozen, config = check_frozen(root)
    binding = read_json(root / "archive-binding.json")
    check_binding(source, binding)
    start = time.monotonic()
    attempted = failed = unavailable = replays = 0
    for entry in selected(binding, phase):
        out = root / phase / Path(entry["path"]).name
        require(read_json(out / "entry.json") == entry, "Planned source entry")
        checksums = read_json(out / "checksums.json")
        require(
            {k: v for k, v in file_hashes(out).items() if k != "checksums.json"} == checksums,
            "Diagnostic hash/coverage",
        )
        context = arrays(out / "context.npz") if (out / "context.npz").exists() else None
        if context is not None:
            with threadpool_limits(limits=1):
                gmm = audit_context(context, read_json(out / "decoder.json"), source, entry)
            write_json(gmm, root / f"{phase}-gmm-{out.name}.json")
        for name in REPRESENTATIONS:
            attempted += 1
            fit = out / name
            record = read_json(fit / "result.json")
            if record["status"] == "failed":
                require(
                    bool(record.get("error")) and not (fit / "arrays.npz").exists(),
                    "Failure retained without fallback",
                )
                failed += 1
                continue
            require(context is not None, "Successful fit requires context")
            with threadpool_limits(limits=1):
                audit_fit(context, name, record, arrays(fit / "arrays.npz"))
            unavailable += record["hierarchy_status"] != "available"
            replays += native_arm(name) is not None
    require(attempted == frozen["counts"][phase], "Audit budget")
    execution = read_json(root / (phase + "-execution.json"))
    for key, value in (("attempted", attempted), ("failed", failed), ("unavailable", unavailable)):
        require(execution[key] == value, "Execution/audit denominators")
    result = {
        "passed": True,
        "phase": phase,
        "attempted": attempted,
        "failed": failed,
        "unavailable": unavailable,
        "native_replays": replays,
        "seconds": time.monotonic() - start,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
        "limit": "Independent graph/condensation/frontier checks; not a second complete HDBSCAN",
    }
    write_json(result, root / (phase + "-audit.json"))
    check_binding(source, binding)
    guard(root, config)
    return result


def gate(root, *, staging_bytes):
    root = Path(root)
    frozen, config = check_frozen(root)
    audit = read_json(root / "preflight-audit.json")
    require(
        audit["passed"] and audit["freeze_sha256"] == fingerprint_file(root / "frozen.json").sha256,
        "Audited frozen preflight required",
    )
    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging bytes required")
    execution = read_json(root / "preflight-execution.json")
    ratio = frozen["counts"]["main"] / frozen["counts"]["preflight"]
    # Sum per-job wall times, then add the serial audit. No parallel speed credit.
    seconds = 1.25 * ratio * (execution["serial_job_seconds"] + audit["seconds"])
    projected = (
        retained_bytes(root)
        + staging_bytes
        + int(1.25 * ratio * retained_bytes(root / "preflight"))
        + 64 * 1024**2
    )
    free = shutil.disk_usage(root).free
    result = {
        "passed": projected <= config.artifact_cap_bytes
        and seconds < config.max_projected_seconds
        and free >= config.min_free_bytes
        and audit["failed"] == 0
        and audit["unavailable"] == 0,
        "projected_bytes": projected,
        "cap_bytes": config.artifact_cap_bytes,
        "projected_main_audit_seconds": seconds,
        "staging_bytes": staging_bytes,
        "free_bytes": free,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
        "audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
    }
    write_json(result, root / "storage-runtime-gate.json")
    return result
