"""Explicit paired validation and storage-gate commands for scaled fold sizes."""

from pathlib import Path

from super_deli.io.config import read_json


def register(commands):
    names = {
        verb + "-scaled-fold-stability" for verb in ("benchmark", "analyze", "preflight", "replay")
    }
    for name in sorted(names):
        p = commands.add_parser(name)
        p.add_argument("config", type=Path)
        if name.startswith("preflight-"):
            for key in (
                "probe-study",
                "probe-audit",
                "probe-check",
                "artifact-root",
                "output-file",
            ):
                p.add_argument("--" + key, type=Path, required=True)
        else:
            p.add_argument("--output-dir", type=Path, required=True)
            p.add_argument("--workers", type=int, default=1)
            if name.startswith(("analyze-", "replay-")):
                p.add_argument("--scratch", type=Path, required=True)
            if name.startswith("benchmark-"):
                p.add_argument("--preflight", type=Path)
            if name.startswith("replay-"):
                p.add_argument("--source-study", type=Path, required=True)
    return names


def dispatch(args):
    if args.command.startswith("analyze-"):
        from super_deli.benchmarking.scaled_fold_audit import audit_study

        return audit_study(
            args.config, output_dir=args.output_dir, scratch=args.scratch, workers=args.workers
        )
    from super_deli.models.scaled_fold_study import ScaledFoldStudy

    c = ScaledFoldStudy.from_dict(read_json(args.config))
    if args.command.startswith("benchmark-"):
        from super_deli.benchmarking.scaled_fold_study import run_study

        return run_study(
            c, output_dir=args.output_dir, workers=args.workers, preflight=args.preflight
        )
    if args.command.startswith("preflight-"):
        from super_deli.benchmarking.scaled_fold_preflight import storage_preflight

        return storage_preflight(
            c,
            probe_study=args.probe_study,
            probe_audit=args.probe_audit,
            probe_check=args.probe_check,
            artifact_root=args.artifact_root,
            output_file=args.output_file,
        )
    from super_deli.benchmarking.bootstrap_diagnostic_replay import replay_study

    return replay_study(
        c,
        source=args.source_study,
        output_dir=args.output_dir,
        scratch=args.scratch,
        workers=args.workers,
    )
