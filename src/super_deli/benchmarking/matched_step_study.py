"""Gated deterministic training verification, separate from scientific candidates."""

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_reference import balanced_inputs
from super_deli.benchmarking.matched_steps import TOLERANCES, digest_arrays, verify_training
from super_deli.benchmarking.matched_study import check_frozen, guard, inventory
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def descriptors(root, phase):
    result = []
    execution = read_json(root / f"{phase}-execution.json")
    for dataset in execution["datasets"]:
        for row in dataset["records"]:
            if row["arm"].endswith(("gmm", "density")) or row["status"] != "available":
                continue
            folder = (
                root
                / phase
                / dataset["key"]
                / (row["arm"] + ("_replay" if row["kind"] == "replay" else ""))
            )
            subs = read_json(folder / "subsamples.json")
            complete = {0} | {i + 1 for i, s in enumerate(subs) if s["status"] == "complete"}
            for event in read_json(folder / "training-events.json"):
                i = event["fit"]
                if i in complete:
                    result.append(
                        {
                            "phase": phase,
                            "dataset": dataset["key"],
                            "arm": row["arm"],
                            "kind": row["kind"],
                            "fit_index": i,
                            "n": dataset["n"] if i == 0 else len(subs[i - 1]["training_indices"]),
                            "d": event["dimensions"],
                            "seed": row["configuration"]["random_seed"],
                        }
                    )
    require(len(result) == execution["neural_fits"], "Every original neural fit is verifiable")
    return result


def token(descriptor):
    return "__".join(str(descriptor[k]) for k in ("phase", "dataset", "arm", "kind", "fit_index"))


def original(root, descriptor):
    phase, key, arm, kind, i = (
        descriptor[k] for k in ("phase", "dataset", "arm", "kind", "fit_index")
    )
    folder = root / phase / key / (arm + ("_replay" if kind == "replay" else ""))
    data = arrays(root / phase / key / "generating-reference.npz")
    if i:
        subs = read_json(folder / "subsamples.json")
        indices = subs[i - 1]["training_indices"]
        prefix = f"draw_{i - 1}_"
        a = {
            k[len(prefix) :]: v
            for k, v in arrays(folder / "stability.npz").items()
            if k.startswith(prefix)
        }
    else:
        indices = np.arange(len(data["morphology"]))
        a = arrays(folder / "point.npz")
    paired = arm.startswith("paired_")
    x = (
        np.column_stack((data["morphology"][indices], data["ecology"][indices]))
        if paired
        else data[arm.split("_")[0]][indices]
    )
    x = balanced_inputs(x, a, paired)
    prefix = f"fit_{i}_"
    initial = {
        k[len(prefix) :]: v
        for k, v in arrays(folder / "initial-weights.npz").items()
        if k.startswith(prefix)
    }
    require(x.shape == (descriptor["n"], descriptor["d"]), "Original replay input dimensions")
    return x, initial, a


def approval(root):
    _, config = check_frozen(root)
    if not config.fixture:
        approved = read_json(root / "audit-amendment-approval.json")
        require(
            approved["approved"] is True
            and approved["plan_sha256"]
            == fingerprint_file(root / "approved-audit-amendment.md").sha256,
            "Explicit matching numerical-audit amendment approval",
        )
    for phase in ("preflight", "main"):
        require(
            inventory(root / phase) == read_json(root / f"{phase}-inventory.json"),
            "Scientific fits unchanged before verification",
        )
    return config


def probe_grid(root, config):
    points = [d for d in descriptors(root, "preflight") if d["kind"] == "point"]
    if config.fixture:
        return [
            next(d for d in points if d["arm"] == arm and d["fit_index"] == i)
            for arm in ("paired_0", "morphology_0", "ecology_0")
            for i in (0, 1)
        ]
    probes = []
    for n in (64, 256, 51, 204):
        for d in (4, 8, 12):
            for seed in (0, 101, 202) if n in (64, 256) else (0,):
                probes.append(
                    next(p for p in points if (p["n"], p["d"], p["seed"]) == (n, d, seed))
                )
    require(len(probes) == len({token(p) for p in probes}) == 24, "Frozen 24-replay preflight")
    return probes


def replay(job):
    root_string, output_string, descriptor = job
    root, output = Path(root_string), Path(output_string)
    folder = output / token(descriptor)
    folder.mkdir()
    start = time.monotonic()
    try:
        x, initial, expected = original(root, descriptor)
        metrics = verify_training(x, initial, expected, descriptor["seed"])
        np.savez_compressed(folder / "steps.npz", **metrics)
        result = {
            **descriptor,
            "passed": True,
            "updates": 500,
            "exact_initial_final_loss_parity": True,
            "input_sha256": digest_arrays({"input": x}),
            "initial_sha256": digest_arrays(initial),
            "final_sha256": digest_arrays({k: expected[k] for k in initial}),
            "loss_sha256": digest_arrays({"losses": expected["losses"]}),
            "max_errors": metrics["errors"].max(0).tolist(),
            "metrics_sha256": fingerprint_file(folder / "steps.npz").sha256,
        }
    except Exception as exc:
        result = {
            **descriptor,
            "passed": False,
            "error": str(exc),
            "partial_training_diagnostics": getattr(exc, "diagnostics", {}),
        }
    result["seconds"] = time.monotonic() - start
    write_json(result, folder / "verification.json")
    return result


def audit_results(root, name, expected):
    output = root / name
    rows = []
    for descriptor in expected:
        folder = output / token(descriptor)
        row = read_json(folder / "verification.json")
        require(all(row[k] == v for k, v in descriptor.items()), "Exact verification schedule")
        if row["passed"]:
            require(
                fingerprint_file(folder / "steps.npz").sha256 == row["metrics_sha256"],
                "Verified step evidence hash",
            )
            a = arrays(folder / "steps.npz")
            require(
                a["errors"].shape == (500, 4) and a["step_sha256"].shape == (500,),
                "Complete per-update evidence",
            )
            require(
                np.isfinite(a["errors"]).all() and np.all(a["errors"] >= 0),
                "Finite nonnegative numerical residuals",
            )
            np.testing.assert_array_equal(a["errors"].max(0), row["max_errors"])
            x, initial, final = original(root, descriptor)
            require(
                row["input_sha256"] == digest_arrays({"input": x})
                and row["initial_sha256"] == digest_arrays(initial)
                and row["final_sha256"] == digest_arrays({k: final[k] for k in initial})
                and row["loss_sha256"] == digest_arrays({"losses": final["losses"]}),
                "Exact replay anchored to retained inputs and numeric model",
            )
            require(
                row["updates"] == 500 and row["exact_initial_final_loss_parity"],
                "Exact initial/final/loss parity",
            )
        rows.append(row)
    result = {
        "passed": all(r["passed"] for r in rows),
        "verification_replays": len(rows),
        "updates_checked": sum(r.get("updates", 0) for r in rows),
        "failed": sum(not r["passed"] for r in rows),
        "records": rows,
        "tolerances": TOLERANCES,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
        "evidence_inventory": inventory(output),
    }
    write_json(result, root / f"{name}-audit.json")
    return result


def run(root, name, schedule, workers, config):
    require(type(workers) is int and 1 <= workers <= config.max_workers, "At most two workers")
    output = root / name
    output.mkdir()
    start = time.monotonic()
    staging = 0
    prior_seconds = 0.0
    if name != "step-preflight":
        passed = read_json(root / "step-storage-runtime-gate.json")
        staging = 2 * passed["staging_bytes"]
        prior_seconds = passed["prior_measured_seconds"] + sum(
            read_json(p)["seconds"] for p in root.glob("step-*-verification-execution.json")
        )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(schedule), workers):
            rows = list(
                pool.map(
                    replay,
                    [(str(root), str(output), d) for d in schedule[offset : offset + workers]],
                )
            )
            guard(root, config, staging)
            if any(not row["passed"] for row in rows):
                write_json({"passed": False, "records": rows}, root / f"{name}-failure.json")
                raise ValueError("Per-step numerical guard failed; retained without replacement")
            if prior_seconds + time.monotonic() - start > config.max_projected_seconds:
                raise ValueError("Verification runtime limit exceeded")
            if offset % 100 == 0:
                print(
                    {
                        "step_verification": name,
                        "completed": min(offset + workers, len(schedule)),
                        "total": len(schedule),
                    },
                    flush=True,
                )
    result = audit_results(root, name, schedule)
    result["seconds"] = time.monotonic() - start
    write_json(
        {"seconds": result["seconds"], "replays": len(schedule)}, root / f"{name}-execution.json"
    )
    return result


def gate(root, staging_bytes, prior_seconds):
    config = approval(root)
    require(
        type(staging_bytes) is int and staging_bytes >= 0 and prior_seconds >= 0,
        "Measured staging bytes and prior runtime required",
    )
    probe = read_json(root / "step-preflight-audit.json")
    require(
        probe["passed"] and probe["freeze_sha256"] == fingerprint_file(root / "frozen.json").sha256,
        "Matching passed step preflight",
    )
    require(
        inventory(root / "step-preflight") == probe["evidence_inventory"],
        "Unchanged step preflight evidence",
    )
    timing, sizes = {}, {}
    for row in probe["records"]:
        k = (row["n"], row["d"])
        timing[k] = max(timing.get(k, 0), row["seconds"])
        sizes[k] = max(sizes.get(k, 0), retained_bytes(root / "step-preflight" / token(row)))
    schedule = descriptors(root, "preflight") + descriptors(root, "main")
    # Existing full reference audit is a conservative allowance for final
    # source/PCA/density/support checks, even though serial training is skipped.
    audit_bound = read_json(root / "preflight-audit.json")["seconds"] * (
        read_json(root / "main-execution.json")["observations"]
        / read_json(root / "preflight-execution.json")["observations"]
        + 1
    )
    seconds = prior_seconds + 1.25 * (sum(timing[d["n"], d["d"]] for d in schedule) + audit_bound)
    projected = (
        retained_bytes(root)
        + 2 * staging_bytes
        + 128 * 1024**2
        + int(1.25 * sum(sizes[d["n"], d["d"]] for d in schedule))
    )
    result = {
        "passed": seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and shutil.disk_usage(root).free >= config.min_free_bytes,
        "projected_cumulative_seconds": seconds,
        "projected_retained_bytes": projected,
        "prior_measured_seconds": prior_seconds,
        "staging_bytes": staging_bytes,
        "verification_replays": len(schedule),
        "preflight_replays": len(probe["records"]),
        "preflight_audit_sha256": fingerprint_file(root / "step-preflight-audit.json").sha256,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
    }
    write_json(result, root / "step-storage-runtime-gate.json")
    return result


def verify_all(root, workers):
    config = approval(root)
    passed = read_json(root / "step-storage-runtime-gate.json")
    require(
        passed["passed"]
        and passed["freeze_sha256"] == fingerprint_file(root / "frozen.json").sha256
        and passed["preflight_audit_sha256"]
        == fingerprint_file(root / "step-preflight-audit.json").sha256,
        "Matching audited step resource gate",
    )
    for phase in ("preflight", "main"):
        run(root, f"step-{phase}-verification", descriptors(root, phase), workers, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "gate", "verify"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--staging-bytes", type=int, default=0)
    parser.add_argument("--prior-seconds", type=float, default=0)
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    config = approval(root)
    if args.action == "preflight":
        value = run(root, "step-preflight", probe_grid(root, config), args.workers, config)
        print(
            json.dumps(
                {k: v for k, v in value.items() if k not in {"records", "evidence_inventory"}},
                default=list,
            )
        )
    elif args.action == "gate":
        print(json.dumps(gate(root, args.staging_bytes, args.prior_seconds)))
    else:
        verify_all(root, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
