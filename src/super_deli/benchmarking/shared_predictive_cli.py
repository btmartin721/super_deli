"""CLI and measured evidence preflight for fixed-family paired screening."""

import json
import math
import shutil
from pathlib import Path

from super_deli.benchmarking.predictive_study import retained_bytes
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.shared_predictive_config import SharedPredictiveConfig


def register(subparsers):
    for command in ("benchmark", "analyze", "preflight", "replay"):
        p = subparsers.add_parser(command + "-shared-predictive")
        p.add_argument("config", type=Path)
        if command == "preflight":
            for name in ("probe-study", "probe-audit", "artifact-root", "output-file"):
                p.add_argument("--" + name, type=Path, required=True)
        else:
            p.add_argument("--output-dir", type=Path, required=True)
            p.add_argument("--workers", type=int, default=1)
            if command == "benchmark":
                p.add_argument("--preflight", type=Path)
            else:
                p.add_argument("--scratch", type=Path, required=True)
            if command == "replay":
                p.add_argument("--source-study", type=Path, required=True)
                p.add_argument("--per-cell", type=int, default=5)

    return {
        command + "-shared-predictive"
        for command in ("benchmark", "analyze", "preflight", "replay")
    }


def storage_preflight(c, *, probe_study, probe_audit, artifact_root, output_file):
    probe = SharedPredictiveConfig.from_dict(read_json(Path(probe_study) / "configuration.json"))
    if c.probe(probe.study.random_seed) != probe or c.study.random_seed == probe.study.random_seed:
        raise ValueError("Independent complete settings probes required")
    audited = read_json(Path(probe_audit) / "summary.json")
    count = sum(r["planned_fits"] for r in probe.plan())
    if (
        audited["successful_fits"] != count
        or audited["unattempted_fits"]
        or audited["failed_fits"]
        or audited["screening_withheld"]
        or audited.get("external_fold_failures", 0)
    ):
        raise ValueError("Successful independently audited probe required")
    rows = [
        json.loads(s) for s in (Path(probe_study) / "observations.jsonl").read_text().splitlines()
    ]
    sizes = {r["cell"]: r["size_bytes"] for r in rows}
    if set(sizes) != {c.name for c in c.study.cells} or len(rows) != len(sizes):
        raise ValueError("One measured pair per cell required")
    raw = sum(sizes[r["cell"]] for r in c.plan())
    replay = 25 * max(sizes.values())
    new = math.ceil(1.25 * (raw + replay)) + 128 * 1024**2
    base = Path(artifact_root).resolve()
    total = retained_bytes(base) + new
    free = shutil.disk_usage(base).free
    passed = total <= c.study.max_artifact_bytes and free >= c.study.min_free_bytes + new
    result = {
        "status": "passed" if passed else "failed",
        "artifact_root": str(base),
        "configuration_sha256": configuration_hash(c.to_dict()),
        "projected_new_bytes": new,
        "projected_complete_bytes": total,
        "max_artifact_bytes": c.study.max_artifact_bytes,
        "current_free_bytes": free,
        "min_free_bytes": c.study.min_free_bytes,
        "probe_searches": count,
        "main_searches_including_replays": sum(r["planned_fits"] for r in c.plan()) + 150,
        "growth_factor": 1.25,
        "reserve_bytes": 128 * 1024**2,
        "replay_archive_allowance": replay,
    }
    write_json(result, output_file)
    if not passed:
        raise RuntimeError(
            "Storage gate failed; preserve evidence and revise budget/design explicitly"
        )
    return result


def dispatch(args):
    if args.command == "analyze-shared-predictive":
        from super_deli.benchmarking.shared_predictive_analysis import audit_study

        return audit_study(
            args.config, output_dir=args.output_dir, scratch=args.scratch, workers=args.workers
        )
    c = SharedPredictiveConfig.from_dict(read_json(args.config))
    if args.command == "benchmark-shared-predictive":
        from super_deli.benchmarking.shared_predictive_study import run_study

        return run_study(
            c, output_dir=args.output_dir, workers=args.workers, preflight=args.preflight
        )
    if args.command == "preflight-shared-predictive":
        return storage_preflight(
            c,
            probe_study=args.probe_study,
            probe_audit=args.probe_audit,
            artifact_root=args.artifact_root,
            output_file=args.output_file,
        )
    from super_deli.benchmarking.predictive_recovery_preflight import replay_workflows

    return replay_workflows(
        c,
        source=args.source_study,
        output_dir=args.output_dir,
        scratch=args.scratch,
        workers=args.workers,
        per_cell=args.per_cell,
    )
