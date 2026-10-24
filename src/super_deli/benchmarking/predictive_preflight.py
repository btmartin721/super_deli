"""Measured local-evidence gate and independently verified archived point replays."""

import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.local_archives import compact_directory, materialize, retained_bytes
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_study_config import PredictiveStudyConfig


def storage_preflight(config, *, probe_study, probe_audit, artifact_root, output_file):
    root, base = Path(probe_study).resolve(), Path(artifact_root).resolve()
    probe = PredictiveStudyConfig.from_dict(read_json(root / "configuration.json"))
    if config.probe(probe.random_seed) != probe or probe.random_seed == config.random_seed:
        raise ValueError("Use independent full-settings probes for every declared cell")
    audited = read_json(Path(probe_audit) / "summary.json")
    count = sum(r["planned_fits"] for r in probe.plan())
    if (
        audited["successful_fits"] != count
        or audited["unattempted_fits"]
        or audited["screening_withheld"]
    ):
        raise ValueError("Storage preflight requires successful independently audited probes")
    rows = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    sizes = {r["cell"]: r["size_bytes"] for r in rows}
    if set(sizes) != {c.name for c in config.cells} or len(rows) != len(sizes):
        raise ValueError("Every study cell needs exactly one measured probe")
    projected = sum(sizes[r["cell"]] for r in config.plan())
    replay_allowance = 25 * max(sizes.values())
    existing = retained_bytes(base)
    new = math.ceil(1.25 * (projected + replay_allowance)) + 128 * 1024**2
    free = shutil.disk_usage(base).free
    passed = existing + new <= config.max_artifact_bytes and free >= config.min_free_bytes + new
    result = {
        "status": "passed" if passed else "failed",
        "artifact_root": str(base),
        "configuration_sha256": configuration_hash(config.to_dict()),
        "projected_new_bytes": new,
        "projected_complete_bytes": existing + new,
        "max_artifact_bytes": config.max_artifact_bytes,
        "current_free_bytes": free,
        "min_free_bytes": config.min_free_bytes,
        "probe_successful_fits": count,
        "growth_factor": 1.25,
        "reserve_bytes": 128 * 1024**2,
        "main_searches": sum(r["planned_fits"] for r in config.plan()) + 25,
        "planned_replays": 25,
    }
    write_json(result, output_file)
    if not passed:
        raise RuntimeError("Storage gate failed; obtain storage/design revision before launch")
    return result


def _replay(task):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.benchmarking.grouped_study import simulate_grouped
    from super_deli.workflows.predictive import raw_search

    source, row, raw_model, destination, scratch = task
    from super_deli.models.integrative_config import IntegrativeConfig

    model = IntegrativeConfig.from_dict(raw_model)
    root = Path(destination)
    base = root / ".working" / f"replay_{row['repetition']:04d}"
    base.mkdir(parents=True, exist_ok=False)
    values, layout, _ = simulate_grouped("independent_gaussian", 64, row["data_seed"])
    error = None
    with threadpool_limits(limits=1):
        record = raw_search(base, "full", values, layout, model)
        if record["status"] == "complete":
            with materialize(
                Path(source) / row["archive"], row, scratch=scratch, min_free_bytes=8 * 1024**3
            ) as original:
                try:
                    prior = original / "baseline/run"
                    current = base / "full/run"
                    if read_json(prior / "candidates.json") != read_json(
                        current / "candidates.json"
                    ):
                        raise ValueError("Replay candidates differ")
                    for name in ("gmm_candidates.npz", "mixture.npz", "fusion.npz"):
                        with (
                            np.load(prior / name, allow_pickle=False) as a,
                            np.load(current / name, allow_pickle=False) as b,
                        ):
                            if set(a.files) != set(b.files):
                                raise ValueError("Replay array schema differs")
                            for k in a.files:
                                np.testing.assert_array_equal(a[k], b[k])
                    for i in range(2):
                        with (
                            np.load(
                                prior / f"view_{i:03d}/parameters/pca.npz", allow_pickle=False
                            ) as a,
                            np.load(
                                current / f"view_{i:03d}/parameters/pca.npz", allow_pickle=False
                            ) as b,
                        ):
                            for k in a.files:
                                np.testing.assert_array_equal(a[k], b[k])
                except (ValueError, AssertionError, KeyError) as exc:
                    error = str(exc)
                    write_json({"error": error}, base / "verification_failure.json")
    packed = compact_directory(base, root / f"replay_{row['repetition']:04d}.tar.gz")
    shutil.rmtree(base)
    return {
        "repetition": row["repetition"],
        "status": record["status"],
        "verified": record["status"] == "complete" and error is None,
        "verification_error": error,
        **packed,
    }


def replay_controls(config, *, source, output_dir, scratch, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    prior = read_json(Path(source) / "configuration.json")
    if prior["model"] != json.loads(json.dumps(config.model.to_dict())):
        raise ValueError("Replay baseline differs from archived diagnostic model")
    rows = [json.loads(s) for s in (Path(source) / "observations.jsonl").read_text().splitlines()]
    rows = sorted(
        [r for r in rows if r["cell"] == "independent_gaussian_n64"], key=lambda r: r["repetition"]
    )[:25]
    if len(rows) != 25 or [r["repetition"] for r in rows] != list(range(25)):
        raise ValueError("All 25 declared archived replay identities required")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(rows, root / "private-source-plan.json")
    results = []
    with (
        ProcessPoolExecutor(max_workers=workers) as executor,
        (root / "observations.jsonl").open("x") as handle,
    ):
        tasks = (
            (str(Path(source).resolve()), r, config.model.to_dict(), str(root), str(scratch))
            for r in rows
        )
        for result in executor.map(_replay, tasks, chunksize=1):
            results.append(result)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
    summary = {
        "requested_replays": 25,
        "verified_replays": sum(r["verified"] for r in results),
        "successful_fits": sum(r["status"] == "complete" for r in results),
        "failed_fits": sum(r["status"] != "complete" for r in results),
    }
    write_json(summary, root / "summary.json")
    return summary
