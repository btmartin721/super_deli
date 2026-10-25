"""Lazy opt-in screening and bounded benchmark CLI commands."""


def register_commands(commands):
    from pathlib import Path

    names = (
        "cluster-integrative-predictive",
        "benchmark-predictive-aic",
        "analyze-predictive-aic",
        "preflight-predictive-aic",
        "replay-predictive-aic",
    )
    for name in names:
        p = commands.add_parser(
            name, help="Opt-in predictive AIC screening or its bounded validation"
        )
        p.add_argument("config", type=Path)
        if name == "preflight-predictive-aic":
            for key in ("probe-study", "probe-audit", "artifact-root", "output-file"):
                p.add_argument("--" + key, type=Path, required=True)
        else:
            p.add_argument("--output-dir", type=Path, required=True)
        if name == "cluster-integrative-predictive":
            p.add_argument("--model-config", type=Path, required=True)
            p.add_argument("--screen-config", type=Path, required=True)
            p.add_argument("--fold-size-policy", type=Path)
        elif name != "preflight-predictive-aic":
            p.add_argument("--workers", type=int, default=1)
        if name in {"analyze-predictive-aic", "replay-predictive-aic"}:
            p.add_argument("--scratch", type=Path, required=True)
        if name == "benchmark-predictive-aic":
            p.add_argument("--preflight", type=Path)
        if name == "replay-predictive-aic":
            p.add_argument("--source-study", type=Path, required=True)
    return frozenset(names)


def dispatch(args):
    from super_deli.io.config import read_json

    if args.command == "cluster-integrative-predictive":
        from super_deli.io.config import load_config
        from super_deli.models.integrative_config import IntegrativeConfig
        from super_deli.models.predictive_config import PredictiveScreenConfig
        from super_deli.models.predictive_fold_size import PredictiveFoldSizePolicy
        from super_deli.workflows.predictive import run_predictive

        options = (
            {}
            if args.fold_size_policy is None
            else dict(
                fold_size_policy=PredictiveFoldSizePolicy.from_dict(
                    read_json(args.fold_size_policy)
                )
            )
        )
        return run_predictive(
            load_config(args.config),
            model_config=IntegrativeConfig.from_dict(read_json(args.model_config)),
            screen_config=PredictiveScreenConfig.from_dict(read_json(args.screen_config)),
            base_dir=args.config.absolute().parent,
            output_dir=args.output_dir,
            **options,
        )
    if args.command == "analyze-predictive-aic":
        from super_deli.benchmarking.predictive_analysis import audit_study

        return audit_study(
            args.config, output_dir=args.output_dir, scratch=args.scratch, workers=args.workers
        )
    from super_deli.models.predictive_study_config import PredictiveStudyConfig

    config = PredictiveStudyConfig.from_dict(read_json(args.config))
    if args.command == "benchmark-predictive-aic":
        from super_deli.benchmarking.predictive_study import run_study

        return run_study(
            config, output_dir=args.output_dir, workers=args.workers, preflight=args.preflight
        )
    if args.command == "preflight-predictive-aic":
        from super_deli.benchmarking.predictive_preflight import storage_preflight

        return storage_preflight(
            config,
            probe_study=args.probe_study,
            probe_audit=args.probe_audit,
            artifact_root=args.artifact_root,
            output_file=args.output_file,
        )
    from super_deli.benchmarking.predictive_preflight import replay_controls

    return replay_controls(
        config,
        source=args.source_study,
        output_dir=args.output_dir,
        scratch=args.scratch,
        workers=args.workers,
    )
