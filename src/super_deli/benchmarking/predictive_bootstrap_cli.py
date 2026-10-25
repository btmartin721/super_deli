"""Explicit stability, bounded validation and audited resource-gate commands."""

import json
import math
import shutil
from pathlib import Path

from super_deli.io.config import read_json, write_json


def register(commands):
    names = {
        "cluster-integrative-predictive-stability",
        "benchmark-predictive-stability",
        "analyze-predictive-stability",
        "preflight-predictive-stability",
    }
    for name in sorted(names):
        p = commands.add_parser(name)
        p.add_argument("config", type=Path)
        if name == "preflight-predictive-stability":
            for key in ("probe-study", "probe-audit", "artifact-root", "output-file"):
                p.add_argument("--" + key, type=Path, required=True)
        else:
            p.add_argument("--output-dir", type=Path, required=True)
        if name.startswith("cluster-"):
            p.add_argument("--fold-size-policy", type=Path)
            for key in ("model-config", "screen-config", "bootstrap-config"):
                p.add_argument("--" + key, type=Path, required=True)
        elif name.startswith("benchmark-"):
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--preflight", type=Path)
        elif name.startswith("analyze-"):
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--scratch", type=Path, required=True)
    return names


def storage_preflight(c, *, probe_study, probe_audit, artifact_root, output_file):
    from super_deli.benchmarking.predictive_study import retained_bytes
    from super_deli.core.provenance import configuration_hash
    from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy

    source = Path(probe_study)
    probe = PredictiveBootstrapStudy.from_dict(read_json(source / "configuration.json"))
    if c.probe(probe.random_seed) != probe or c.random_seed == probe.random_seed:
        raise ValueError("Independent full-settings probe per cell required")
    audited = read_json(Path(probe_audit) / "summary.json")
    count = sum(r["planned_fits"] for r in probe.plan())
    if audited["audited_searches"] != count or audited["unattempted_draws"]:
        raise ValueError("Complete independently audited probe required")
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    sizes = {r["cell"]: r["size_bytes"] + (source / r["manifest"]).stat().st_size for r in rows}
    if len(sizes) != len(rows) or set(sizes) != {c.name for c in c.cells}:
        raise ValueError("Exactly one measured observation per cell required")
    # Replays contain six point searches rather than a hundred six-search workflows.
    replay = math.ceil(15 * max(sizes.values()) / (1 + c.n_bootstrap))
    new = math.ceil(1.25 * (sum(sizes[r["cell"]] for r in c.plan()) + replay)) + 128 * 1024**2
    root = Path(artifact_root).resolve()
    total, free = retained_bytes(root) + new, shutil.disk_usage(root).free
    passed = total <= c.max_artifact_bytes and free >= c.min_free_bytes + new
    result = {
        "status": "passed" if passed else "failed",
        "artifact_root": str(root),
        "configuration_sha256": configuration_hash(c.to_dict()),
        "projected_new_bytes": new,
        "projected_complete_bytes": total,
        "max_artifact_bytes": c.max_artifact_bytes,
        "current_free_bytes": free,
        "min_free_bytes": c.min_free_bytes,
        "growth_factor": 1.25,
        "reserve_bytes": 128 * 1024**2,
        "probe_searches": count,
        "main_searches_including_replays": sum(r["planned_fits"] for r in c.plan()) + 90,
        "replay_archive_allowance": replay,
        "failed_probe_draws_retained": audited["failed_draws"],
    }
    write_json(result, output_file)
    if not passed:
        raise RuntimeError("Storage gate failed; preserve evidence and revise budget explicitly")
    return result


def dispatch(args):
    if args.command.startswith("cluster-"):
        from super_deli.io.config import load_config
        from super_deli.models.integrative_config import IntegrativeConfig
        from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
        from super_deli.models.predictive_config import PredictiveScreenConfig
        from super_deli.models.predictive_fold_size import PredictiveFoldSizePolicy
        from super_deli.workflows.predictive_bootstrap import run_predictive_bootstrap

        options = (
            {}
            if args.fold_size_policy is None
            else dict(
                fold_size_policy=PredictiveFoldSizePolicy.from_dict(
                    read_json(args.fold_size_policy)
                )
            )
        )
        return run_predictive_bootstrap(
            load_config(args.config),
            model_config=IntegrativeConfig.from_dict(read_json(args.model_config)),
            screen_config=PredictiveScreenConfig.from_dict(read_json(args.screen_config)),
            bootstrap_config=PredictiveBootstrapConfig.from_dict(read_json(args.bootstrap_config)),
            base_dir=args.config.absolute().parent,
            output_dir=args.output_dir,
            **options,
        )
    if args.command.startswith("analyze-"):
        from super_deli.benchmarking.predictive_bootstrap_audit import audit_study

        return audit_study(
            args.config, output_dir=args.output_dir, scratch=args.scratch, workers=args.workers
        )
    from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy

    c = PredictiveBootstrapStudy.from_dict(read_json(args.config))
    if args.command.startswith("preflight-"):
        return storage_preflight(
            c,
            probe_study=args.probe_study,
            probe_audit=args.probe_audit,
            artifact_root=args.artifact_root,
            output_file=args.output_file,
        )
    from super_deli.benchmarking.predictive_bootstrap_study import run_study

    return run_study(c, output_dir=args.output_dir, workers=args.workers, preflight=args.preflight)
