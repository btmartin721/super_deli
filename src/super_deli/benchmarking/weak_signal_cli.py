"""Opt-in archival diagnostics; dependency-free command registration."""

from pathlib import Path


def register(commands):
    parser = commands.add_parser("diagnose-weak-signal", help="Bounded archival recovery diagnosis")
    parser.add_argument("action", choices=("bind", "freeze", "run", "audit", "gate", "report"))
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-config", type=Path)
    parser.add_argument("--phase", choices=("preflight", "main"), default="preflight")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--staging-bytes", type=int)


def dispatch(args):
    from super_deli.benchmarking.coordinate_fit import frozen_only
    from super_deli.benchmarking.weak_signal_config import WeakSignalStudy
    from super_deli.io.config import read_json

    root, source = args.output_dir.resolve(), args.archive_root.resolve()
    config = (
        WeakSignalStudy.from_dict(read_json(args.study_config))
        if args.study_config
        else WeakSignalStudy()
    )
    with frozen_only():
        if args.action == "bind":
            from super_deli.benchmarking.weak_signal_source import bind

            return bind(source, root, fixture=config.fixture)
        from super_deli.benchmarking.weak_signal_study import engine, gate

        study = engine(root, source)
        if args.action == "freeze":
            return study.freeze(config)
        _, frozen = study.check()
        if args.study_config and frozen != config:
            raise ValueError("Configuration differs from source freeze")
        if args.action == "run":
            return study.run(args.phase, workers=args.workers)
        if args.action == "audit":
            return study.audit(args.phase)
        if args.action == "gate":
            return gate(root, source, staging_bytes=args.staging_bytes)
        from super_deli.benchmarking.weak_signal_report import report

        return report(root, source)
