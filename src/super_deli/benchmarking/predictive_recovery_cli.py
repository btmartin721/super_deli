"""Lazy CLI for an isolated, bounded recovery diagnosis."""


def register_commands(commands):
    from pathlib import Path

    names = (
        "benchmark-predictive-recovery",
        "analyze-predictive-recovery",
        "diagnose-predictive-archives",
        "preflight-predictive-recovery",
        "replay-predictive-recovery",
    )
    for name in names:
        p = commands.add_parser(name, help="Bounded diagnosis of predictive-screen recovery")
        p.add_argument("config", type=Path)
        if name == "preflight-predictive-recovery":
            for key in ("probe-study", "probe-audit", "artifact-root", "output-file"):
                p.add_argument("--" + key, type=Path, required=True)
        else:
            p.add_argument("--output-dir", type=Path, required=True)
            p.add_argument("--workers", type=int, default=1)
        if name in {
            "analyze-predictive-recovery",
            "diagnose-predictive-archives",
            "replay-predictive-recovery",
        }:
            p.add_argument("--scratch", type=Path, required=True)
        if name in {"diagnose-predictive-archives", "replay-predictive-recovery"}:
            p.add_argument("--source-study", type=Path, required=True)
        if name == "diagnose-predictive-archives":
            p.add_argument("--artifact-root", type=Path, required=True)
        if name == "benchmark-predictive-recovery":
            p.add_argument("--preflight", type=Path)
        if name == "replay-predictive-recovery":
            p.add_argument("--per-cell", type=int, default=5)
    return frozenset(names)


def dispatch(args):
    from super_deli.io.config import read_json
    from super_deli.models.predictive_recovery_config import PredictiveRecoveryConfig

    if args.command == "analyze-predictive-recovery":
        from super_deli.benchmarking.predictive_recovery_analysis import audit_study

        return audit_study(
            args.config, output_dir=args.output_dir, scratch=args.scratch, workers=args.workers
        )
    config = PredictiveRecoveryConfig.from_dict(read_json(args.config))
    if args.command == "benchmark-predictive-recovery":
        from super_deli.benchmarking.predictive_recovery_study import run_study

        return run_study(
            config, output_dir=args.output_dir, workers=args.workers, preflight=args.preflight
        )
    if args.command == "diagnose-predictive-archives":
        from super_deli.benchmarking.predictive_recovery_archive import diagnose_archives

        return diagnose_archives(
            config,
            source=args.source_study,
            output_dir=args.output_dir,
            scratch=args.scratch,
            artifact_root=args.artifact_root,
            workers=args.workers,
        )
    if args.command == "preflight-predictive-recovery":
        from super_deli.benchmarking.predictive_recovery_preflight import storage_preflight

        return storage_preflight(
            config,
            probe_study=args.probe_study,
            probe_audit=args.probe_audit,
            artifact_root=args.artifact_root,
            output_file=args.output_file,
        )
    from super_deli.benchmarking.predictive_recovery_preflight import replay_workflows

    return replay_workflows(
        config,
        source=args.source_study,
        output_dir=args.output_dir,
        scratch=args.scratch,
        workers=args.workers,
        per_cell=args.per_cell,
    )
