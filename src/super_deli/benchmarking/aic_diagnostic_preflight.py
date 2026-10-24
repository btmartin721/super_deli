"""Measured local evidence gate for the frozen point-only diagnostic."""

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.local_archives import retained_bytes
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.aic_diagnostic_config import AICDiagnosticConfig


def probe_config(config, *, random_seed=20261125):
    return replace(config, n_datasets=1, n_scores=len(config.cases), random_seed=random_seed)


def storage_preflight(
    config, *, probe_study, probe_audit, probe_replay, artifact_root, output_file
):
    root = Path(probe_study).resolve()
    artifact_root = Path(artifact_root).resolve()
    probe = AICDiagnosticConfig.from_dict(read_json(root / "configuration.json"))
    if (
        probe_config(config, random_seed=probe.random_seed) != probe
        or probe.random_seed == config.random_seed
    ):
        raise ValueError("Use independent full-settings probes")
    audit = read_json(Path(probe_audit) / "summary.json")
    expected = sum(r["planned_fits"] for r in probe.plan())
    if audit["point_fits"] != expected or audit["successful_fits"] != expected:
        raise ValueError("Storage gate requires fully successful audited probes")
    replay = read_json(Path(probe_replay) / "summary.json")
    if replay["requested_replays"] != 1 or replay["verified_replays"] != 1:
        raise ValueError("One verified archived replay required for storage preflight")
    ledger = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    sizes = {}
    for row in ledger:
        sizes.setdefault((row["case"], row["n"]), []).append(row["size_bytes"])
    projection = sum(max(sizes[(r["case"], r["n"])]) for r in config.plan())
    # Replays fit only one point and have no held-out data: use a conservative
    # full raw-observation archive allowance for each of the 50 replays.
    replay_allowance = 50 * max(
        max(v) for (case, _), v in sizes.items() if case != "gaussian_scores"
    )
    existing = retained_bytes(artifact_root)
    new = math.ceil(1.25 * (projection + replay_allowance)) + 64 * 1024**2
    free = shutil.disk_usage(artifact_root).free
    passed = existing + new <= config.max_artifact_bytes and free >= config.min_free_bytes + new
    result = {
        "status": "passed" if passed else "failed",
        "configuration_sha256": configuration_hash(config.to_dict()),
        "artifact_root": str(artifact_root),
        "projected_new_bytes": new,
        "projected_complete_bytes": existing + new,
        "max_artifact_bytes": config.max_artifact_bytes,
        "current_free_bytes": free,
        "min_free_bytes": config.min_free_bytes,
        "probe_successful_fits": expected,
        "growth_factor": 1.25,
        "reserve_bytes": 64 * 1024**2,
        "replay_allowance_bytes": replay_allowance,
        "planned_study_fits": sum(r["planned_fits"] for r in config.plan()),
        "planned_replays": 50,
        "scope": "Measured projection, not permission to discard evidence",
    }
    write_json(result, output_file)
    if not passed:
        raise RuntimeError("Storage gate failed; obtain storage/design revision before launch")
    return result
