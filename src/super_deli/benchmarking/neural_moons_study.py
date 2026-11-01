"""Gated archival experiment with exclusive outputs and complete attempt accounting."""

import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from super_deli import __version__
from super_deli.benchmarking.neural_moons_config import POLICIES, REPRESENTATIONS, NeuralMoonsConfig
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def source_hashes():
    package = Path(__file__).parents[1]
    return {
        str(p.relative_to(package)): fingerprint_file(p).sha256
        for p in sorted(package.rglob("*.py"))
    }


def runtime():
    return {
        "package": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            n: version(n) for n in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
    }


def check_free(root, config):
    if shutil.disk_usage(root).free < config.min_free_bytes:
        raise ValueError("Diagnostic free-space reserve is not satisfied")


def plan(binding, phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Unknown diagnostic phase")
    point = binding["observations"]
    replay = binding["replays"]
    if phase == "preflight":
        point = [r for r in point if r["replicate"] == 0]
        replay = [r for r in replay if r["replay_arm"] == "neural_0"]
    return [
        {"entry": r, "representations": list(REPRESENTATIONS), "replay_only": False} for r in point
    ] + [
        {
            "entry": r,
            "representations": [
                "pca_native" if r["replay_arm"] == "density" else r["replay_arm"] + "_standard"
            ],
            "replay_only": True,
        }
        for r in replay
    ]


def freeze(root, config):
    root = Path(root).resolve()
    config = NeuralMoonsConfig.from_dict(config.to_dict())
    binding = read_json(root / "archive-binding.json")
    if not binding["passed"] or config.fixture != (binding["kind"] == "invented_untrained_fixture"):
        raise ValueError("Diagnostic configuration/source kind mismatch")
    counts = {
        p: sum(len(j["representations"]) for j in plan(binding, p)) for p in ("preflight", "main")
    }
    if not config.fixture and counts != {"preflight": 56, "main": 548}:
        raise ValueError("Scientific diagnostic fit budget differs")
    check_free(root, config)
    frozen = {
        "config": config.to_dict(),
        "source_sha256": source_hashes(),
        "runtime": runtime(),
        "binding_sha256": fingerprint_file(root / "archive-binding.json").sha256,
        "plan_sha256": fingerprint_file(root / "approved-plan.md").sha256,
        "fits": counts,
        "representations": list(REPRESENTATIONS),
        "policies": POLICIES,
    }
    write_json(frozen, root / "frozen.json")
    return {"frozen": True, "fits": counts, "config": config.to_dict()}


def check_frozen(root):
    frozen = read_json(Path(root) / "frozen.json")
    if (
        frozen["source_sha256"] != source_hashes()
        or frozen["runtime"] != runtime()
        or frozen["binding_sha256"] != fingerprint_file(Path(root) / "archive-binding.json").sha256
        or frozen["plan_sha256"] != fingerprint_file(Path(root) / "approved-plan.md").sha256
    ):
        raise ValueError("Diagnostic source, runtime or evidence binding changed")
    return frozen, NeuralMoonsConfig.from_dict(frozen["config"])


def run_job(job):
    import numpy as np

    from super_deli.benchmarking.neural_moons_coordinates import load_context
    from super_deli.benchmarking.neural_moons_fit import fit_coordinates

    entry = job["entry"]
    out = Path(job["phase_root"]) / entry["key"]
    out.mkdir(exist_ok=False)
    write_json(
        {k: v for k, v in job.items() if k not in {"source_root", "phase_root"}}, out / "job.json"
    )
    config = NeuralMoonsConfig.from_dict(job["config"])
    setup_error = None
    try:
        context, diagnostics = load_context(job["source_root"], entry)
        with (out / "context.npz").open("xb") as f:
            np.savez_compressed(f, **context)
        write_json(diagnostics, out / "reconstruction.json")
    except Exception as exc:
        setup_error = str(exc)
        write_json({"status": "failed", "error": setup_error}, out / "context-failure.json")
    attempted, failed, unavailable = 0, 0, 0
    for representation in job["representations"]:
        attempted += 1
        target = out / representation
        target.mkdir()
        try:
            if setup_error is not None:
                raise ValueError(setup_error)
            arm = (
                "density"
                if representation == "pca_native"
                else representation.removesuffix("_standard")
            )
            expected = (
                (context[arm + "_labels"], context[arm + "_strength"])
                if representation == "pca_native"
                or representation.endswith("_standard")
                and representation.startswith("neural_")
                else None
            )
            record, data = fit_coordinates(
                context[representation],
                context,
                config=config,
                expected=expected,
                replay_only=job["replay_only"],
            )
            with (target / "arrays.npz").open("xb") as f:
                np.savez_compressed(f, **data)
            unavailable += record["hierarchy_status"] != "available"
        except Exception as exc:
            failed += 1
            record = {
                "status": "failed",
                "error": str(exc),
                "policies": {
                    p: {"status": "failed", "error": str(exc)}
                    for p in (["baseline"] if job["replay_only"] else POLICIES)
                },
            }
        write_json(record, target / "result.json")
    checksums = {
        str(p.relative_to(out)): fingerprint_file(p).sha256
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    write_json(checksums, out / "checksums.json")
    return {
        "key": entry["key"],
        "attempted": attempted,
        "failed": failed,
        "unavailable_hierarchies": unavailable,
    }


def run_phase(root, source_root, phase, *, workers=1):
    from super_deli.benchmarking.neural_moons_archive import check_binding

    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    frozen, config = check_frozen(root)
    if type(workers) is not int or not 1 <= workers <= config.max_workers:
        raise ValueError("Invalid diagnostic worker count")
    binding = read_json(root / "archive-binding.json")
    check_binding(source_root, binding)
    if phase == "main":
        gate = read_json(root / "storage-runtime-gate.json")
        if (
            not gate["passed"]
            or gate["freeze_sha256"] != fingerprint_file(root / "frozen.json").sha256
            or gate["preflight_audit_sha256"]
            != fingerprint_file(root / "preflight-audit.json").sha256
        ):
            raise ValueError("Passed bound preflight gate required")
    check_free(root, config)
    phase_root = root / phase
    phase_root.mkdir(exist_ok=False)
    jobs = [
        {
            **j,
            "source_root": str(source_root),
            "phase_root": str(phase_root),
            "config": config.to_dict(),
        }
        for j in plan(binding, phase)
    ]
    start = time.monotonic()
    if workers == 1:
        records = [run_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(run_job, jobs))
    total = sum(r["attempted"] for r in records)
    if total != frozen["fits"][phase]:
        raise ValueError("Diagnostic attempt accounting differs from freeze")
    execution = {
        "phase": phase,
        "attempted": total,
        "failed": sum(r["failed"] for r in records),
        "unavailable_hierarchies": sum(r["unavailable_hierarchies"] for r in records),
        "seconds": time.monotonic() - start,
        "workers": workers,
        "records": records,
    }
    write_json(execution, root / f"{phase}-execution.json")
    check_binding(source_root, binding)
    check_free(root, config)
    return {k: v for k, v in execution.items() if k != "records"}


def audit_phase(root, source_root, phase):
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.benchmarking.neural_moons_archive import check_binding
    from super_deli.benchmarking.neural_moons_audit import audit_context, audit_fit

    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    frozen, _ = check_frozen(root)
    binding = read_json(root / "archive-binding.json")
    check_binding(source_root, binding)
    attempted = failed = unavailable = replays = selections = 0
    for job in plan(binding, phase):
        folder = root / phase / job["entry"]["key"]
        original = read_json(folder / "job.json")
        for key in ("entry", "representations", "replay_only"):
            if original[key] != job[key]:
                raise ValueError("Diagnostic job differs from its frozen plan")
        checksums = read_json(folder / "checksums.json")
        actual = {
            str(p.relative_to(folder))
            for p in folder.rglob("*")
            if p.is_file() and p.name != "checksums.json"
        }
        if actual != set(checksums):
            raise ValueError("Diagnostic artifact inventory differs")
        for name, digest in checksums.items():
            if fingerprint_file(folder / name).sha256 != digest:
                raise ValueError("Diagnostic artifact hash mismatch")
        context = arrays(folder / "context.npz") if (folder / "context.npz").exists() else None
        if context is not None:
            audit_context(
                context, job["entry"], source_root, read_json(folder / "reconstruction.json")
            )
        for representation in job["representations"]:
            attempted += 1
            fit = folder / representation
            record = read_json(fit / "result.json")
            expected_policies = {"baseline"} if job["replay_only"] else set(POLICIES)
            if set(record["policies"]) != expected_policies:
                raise ValueError("Diagnostic selection budget differs")
            selections += len(expected_policies)
            if record["status"] == "failed":
                failed += 1
                if not record.get("error") or (fit / "arrays.npz").exists():
                    raise ValueError("Invalid retained failure")
                continue
            if context is None:
                raise ValueError("Successful fit without valid context")
            data = arrays(fit / "arrays.npz")
            audit_fit(context, representation, record, data)
            if record["archive_replay"]:
                arm = (
                    "density"
                    if representation == "pca_native"
                    else representation.removesuffix("_standard")
                )
                import numpy as np

                np.testing.assert_array_equal(data["baseline_labels"], context[arm + "_labels"])
                np.testing.assert_array_equal(data["baseline_strength"], context[arm + "_strength"])
                replays += 1
            unavailable += record["hierarchy_status"] != "available"
    if attempted != frozen["fits"][phase]:
        raise ValueError("Audited fit budget differs")
    execution = read_json(root / f"{phase}-execution.json")
    if any(
        execution[k] != v
        for k, v in (
            ("attempted", attempted),
            ("failed", failed),
            ("unavailable_hierarchies", unavailable),
        )
    ):
        raise ValueError("Execution/audit accounting mismatch")
    result = {
        "passed": True,
        "phase": phase,
        "attempted": attempted,
        "failed": failed,
        "unavailable_hierarchies": unavailable,
        "archive_replays": replays,
        "selection_evaluations": selections,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
        "limit": (
            "Independent graph/condensation/frontier checks; "
            "not a second complete HDBSCAN implementation"
        ),
    }
    write_json(result, root / f"{phase}-audit.json")
    return result


def retained_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def gate(root, *, staging_bytes):
    root = Path(root).resolve()
    frozen, config = check_frozen(root)
    audit = read_json(root / "preflight-audit.json")
    if (
        not audit["passed"]
        or audit["freeze_sha256"] != fingerprint_file(root / "frozen.json").sha256
    ):
        raise ValueError("Bound preflight audit required")
    if type(staging_bytes) is not int or staging_bytes < 0:
        raise ValueError("Actual nonnegative staging evidence bytes required")
    execution = read_json(root / "preflight-execution.json")
    ratio = frozen["fits"]["main"] / frozen["fits"]["preflight"]
    projected = (
        int(retained_bytes(root / "preflight") * ratio * 1.25)
        + retained_bytes(root)
        + staging_bytes
        + 128 * 1024**2
    )
    seconds = execution["seconds"] * execution["workers"] / config.max_workers * ratio * 1.25
    free = shutil.disk_usage(root).free
    result = {
        "passed": projected <= config.artifact_cap_bytes
        and seconds <= config.max_projected_seconds
        and free >= config.min_free_bytes
        and audit["failed"] == 0
        and audit["unavailable_hierarchies"] == 0,
        "projected_bytes": projected,
        "cap_bytes": config.artifact_cap_bytes,
        "staging_bytes": staging_bytes,
        "projected_main_seconds": seconds,
        "free_bytes": free,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
        "preflight_audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
        "failed_fits": audit["failed"],
        "unavailable_hierarchies": audit["unavailable_hierarchies"],
    }
    write_json(result, root / "storage-runtime-gate.json")
    return result
