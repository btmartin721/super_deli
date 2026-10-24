"""Lazy CLI registration for the bounded Gaussian-control diagnostic."""


def register_commands(commands):
    from pathlib import Path

    names = (
        "benchmark-aic-gaussian",
        "analyze-aic-gaussian",
        "audit-aic-gaussian-sources",
        "replay-aic-gaussian",
        "preflight-aic-gaussian",
    )
    for name in names:
        parser = commands.add_parser(name, help="Bounded point-only Gaussian AIC diagnosis")
        parser.add_argument("config", type=Path)
        if name != "preflight-aic-gaussian":
            parser.add_argument("--output-dir", type=Path, required=True)
            parser.add_argument("--workers", type=int, default=1)
        if name in {"analyze-aic-gaussian", "audit-aic-gaussian-sources", "replay-aic-gaussian"}:
            parser.add_argument("--scratch", type=Path, required=True)
        if name in {"audit-aic-gaussian-sources", "replay-aic-gaussian"}:
            parser.add_argument("--power-study", type=Path, required=True)
        if name == "audit-aic-gaussian-sources":
            parser.add_argument("--null-study", type=Path, required=True)
        if name == "replay-aic-gaussian":
            parser.add_argument("--limit", type=int, choices=[1])
        if name == "benchmark-aic-gaussian":
            parser.add_argument("--preflight", type=Path)
        if name == "preflight-aic-gaussian":
            for key in (
                "probe-study",
                "probe-audit",
                "probe-replay",
                "artifact-root",
                "output-file",
            ):
                parser.add_argument("--" + key, type=Path, required=True)
    return frozenset(names)


def dispatch(args):
    from super_deli.io.config import read_json
    from super_deli.models.aic_diagnostic_config import AICDiagnosticConfig

    if args.command == "analyze-aic-gaussian":
        from super_deli.benchmarking.aic_diagnostic_analysis import audit_study

        return audit_study(
            args.config, output_dir=args.output_dir, scratch=args.scratch, workers=args.workers
        )
    config = AICDiagnosticConfig.from_dict(read_json(args.config))
    if args.command == "benchmark-aic-gaussian":
        from super_deli.benchmarking.aic_diagnostic_study import run_study

        return run_study(
            config, output_dir=args.output_dir, workers=args.workers, preflight=args.preflight
        )
    if args.command == "preflight-aic-gaussian":
        from super_deli.benchmarking.aic_diagnostic_preflight import storage_preflight

        return storage_preflight(
            config,
            probe_study=args.probe_study,
            probe_audit=args.probe_audit,
            probe_replay=args.probe_replay,
            artifact_root=args.artifact_root,
            output_file=args.output_file,
        )
    if args.command == "replay-aic-gaussian":
        from super_deli.benchmarking.aic_diagnostic_sources import replay_controls

        return replay_controls(
            config,
            power_study=args.power_study,
            output_dir=args.output_dir,
            scratch=args.scratch,
            workers=args.workers,
            limit=args.limit,
        )
    from super_deli.benchmarking.aic_diagnostic_sources import audit_sources

    return audit_sources(
        config,
        null_study=args.null_study,
        power_study=args.power_study,
        output_dir=args.output_dir,
        scratch=args.scratch,
        workers=args.workers,
    )
