"""Measured storage gate and six-search archived workflow reproduction."""

import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.local_archives import compact_directory, materialize
from super_deli.benchmarking.predictive_study import retained_bytes
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import load_config, read_json, write_json
from super_deli.models.predictive_recovery_config import PredictiveRecoveryConfig
from super_deli.models.predictive_study_config import PredictiveStudyConfig

REPLAY_CELLS = (
    "four_n64_s1_5",
    "four_n64_s3_0",
    "four_n64_s4_5",
    "four_n256_s3_0",
    "independent_gaussian_n64",
)


def storage_preflight(config, *, probe_study, probe_audit, artifact_root, output_file):
    source = Path(probe_study)
    root = Path(artifact_root).resolve()
    probe = PredictiveRecoveryConfig.from_dict(read_json(source / "configuration.json"))
    if (
        config.probe(probe.study.random_seed) != probe
        or probe.study.random_seed == config.study.random_seed
    ):
        raise ValueError("Independent full-settings probe for every cell required")
    audit = read_json(Path(probe_audit) / "summary.json")
    counts = {k: sum(r[k] for r in probe.plan()) for k in ("planned_searches", "planned_oracles")}
    if (
        audit["successful_searches"] != counts["planned_searches"]
        or audit["successful_oracles"] != counts["planned_oracles"]
        or audit["withheld_parts"]
    ):
        raise ValueError("Preflight requires every planned fit and independent audit to succeed")
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    sizes = {r["cell"]: r["size_bytes"] for r in rows}
    if len(rows) != len(sizes) or set(sizes) != {c.name for c in config.study.cells}:
        raise ValueError("Exactly one probe per cell required")
    projected = sum(sizes[r["cell"]] for r in config.plan())
    replay_allowance = 25 * max(sizes.values())
    existing = retained_bytes(root)
    new = math.ceil(1.25 * (projected + replay_allowance)) + 128 * 1024**2
    free = shutil.disk_usage(root).free
    passed = (
        existing + new <= config.study.max_artifact_bytes
        and free >= config.study.min_free_bytes + new
    )
    result = {
        "status": "passed" if passed else "failed",
        "artifact_root": str(root),
        "configuration_sha256": configuration_hash(config.to_dict()),
        "projected_new_bytes": new,
        "projected_complete_bytes": existing + new,
        "max_artifact_bytes": config.study.max_artifact_bytes,
        "current_free_bytes": free,
        "min_free_bytes": config.study.min_free_bytes,
        "growth_factor": 1.25,
        "reserve_bytes": 128 * 1024**2,
        "probe_searches": counts["planned_searches"],
        "probe_single_k_fits": counts["planned_oracles"],
        "main_searches_including_replays": sum(r["planned_searches"] for r in config.plan()) + 150,
        "main_single_k_fits": sum(r["planned_oracles"] for r in config.plan()),
        "replay_archive_allowance": replay_allowance,
    }
    write_json(result, output_file)
    if not passed:
        raise RuntimeError(
            "Storage gate failed; obtain storage/design revision before launching main study"
        )
    return result


def _replay(task):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.workflows.predictive import run_predictive

    source, row, old_config, destination, scratch = task
    old = PredictiveStudyConfig.from_dict(old_config)
    root = Path(destination)
    base = root / ".working" / f"{row['cell']}_{row['repetition']:04d}"
    base.mkdir(parents=True)
    error = None
    with (
        threadpool_limits(limits=1),
        materialize(
            Path(source) / row["archive"], row, scratch=scratch, min_free_bytes=old.min_free_bytes
        ) as original,
    ):
        try:
            summary = run_predictive(
                load_config(original / "observed/dataset.json"),
                model_config=old.model,
                screen_config=replace(old.screening, random_seed=row["split_seed"]),
                base_dir=original / "observed",
                output_dir=base / "workflow",
            )
            if summary["selection"] != row["selection"]:
                raise ValueError("Replayed selection differs")
            if read_json(base / "workflow/fold_assignments.json") != read_json(
                original / "workflow/fold_assignments.json"
            ):
                raise ValueError("Replay fold identities differ")
            for prefix in ["full"] + [f"fold_{i:03d}" for i in range(5)]:
                a = original / "workflow" / prefix / "run"
                b = base / "workflow" / prefix / "run"
                if read_json(a / "candidates.json") != read_json(b / "candidates.json"):
                    raise ValueError("Replay candidates differ")
                for relative in (
                    "gmm_candidates.npz",
                    "mixture.npz",
                    "fusion.npz",
                    "view_000/parameters/pca.npz",
                    "view_001/parameters/pca.npz",
                    "view_000/parameters/profiles.npz",
                    "view_001/parameters/profiles.npz",
                ):
                    with (
                        np.load(a / relative, allow_pickle=False) as x,
                        np.load(b / relative, allow_pickle=False) as y,
                    ):
                        if set(x.files) != set(y.files):
                            raise ValueError("Replay parameter schema differs")
                        for key in x.files:
                            np.testing.assert_array_equal(x[key], y[key])
            verified = True
        except (
            ValueError,
            RuntimeError,
            AssertionError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            verified = False
            error = str(exc)
            write_json({"error": error}, base / "verification_failure.json")
    records = [read_json(p) for p in (base / "workflow").glob("*/record.json")]
    result = {
        "cell": row["cell"],
        "repetition": row["repetition"],
        "verified": verified,
        "verification_error": error,
        "planned_searches": 6,
        "attempted_searches": len(records),
        "successful_searches": sum(r["status"] == "complete" for r in records),
        "failed_searches": sum(r["status"] != "complete" for r in records),
        "unattempted_searches": 6 - len(records),
    }
    write_json(result, base / "record.json")
    write_json(row, base / "private-source-pin.json")
    relative = f"{row['cell']}_{row['repetition']:04d}.tar.gz"
    packed = compact_directory(base, root / relative)
    shutil.rmtree(base)
    return {**result, **packed}


def replay_workflows(config, *, source, output_dir, scratch, workers=1, per_cell=5):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four workers")
    if type(per_cell) is not int or not 1 <= per_cell <= 5:
        raise ValueError("One to five replays per cell")
    prior = PredictiveStudyConfig.from_dict(read_json(Path(source) / "configuration.json"))
    if json.loads(json.dumps(config.study.model.to_dict())) != json.loads(
        json.dumps(prior.model.to_dict())
    ):
        raise ValueError("Archive replay requires the unchanged original model")
    ledger = [json.loads(s) for s in (Path(source) / "observations.jsonl").read_text().splitlines()]
    rows = [
        r
        for name in REPLAY_CELLS
        for r in sorted([v for v in ledger if v["cell"] == name], key=lambda v: v["repetition"])[
            :per_cell
        ]
    ]
    if len(rows) != 5 * per_cell:
        raise ValueError("All declared replay observations required")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(rows, root / "private-source-plan.json")
    records = []
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (root / "observations.jsonl").open("x") as out,
    ):
        tasks = (
            (
                str(Path(source).resolve()),
                r,
                prior.to_dict(),
                str(root),
                str(Path(scratch).resolve()),
            )
            for r in rows
        )
        for r in pool.map(_replay, tasks, chunksize=1):
            records.append(r)
            out.write(json.dumps(r, sort_keys=True) + "\n")
            out.flush()
    result = {
        "requested_replays": len(rows),
        "verified_replays": sum(r["verified"] for r in records),
        **{
            key: sum(r[key] for r in records)
            for key in (
                "planned_searches",
                "attempted_searches",
                "successful_searches",
                "failed_searches",
                "unattempted_searches",
            )
        },
    }
    write_json(result, root / "summary.json")
    return result
