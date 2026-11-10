"""Opt-in bounded power study; imports remain lazy for dependency-free commands."""

from pathlib import Path


def register(commands):
    parser = commands.add_parser(
        "benchmark-neural-power", help="Bounded paired calibration power/recovery study"
    )
    parser.add_argument(
        "action", choices=("bind", "freeze", "run", "audit", "gate", "report", "pause", "resume")
    )
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--study-config", type=Path)
    parser.add_argument("--phase", choices=("preflight", "main"), default="preflight")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--staging-bytes", type=int)


def dispatch(args):
    import uuid

    from super_deli.benchmarking.neural_false_split_state import atomic_json
    from super_deli.benchmarking.neural_power_config import NeuralPowerStudy
    from super_deli.io.config import read_json

    root, source = args.output_dir.resolve(), args.archive_root.resolve()
    config = (
        NeuralPowerStudy.from_dict(read_json(args.study_config))
        if args.study_config
        else NeuralPowerStudy()
    )
    if args.action == "bind":
        from super_deli.benchmarking.neural_power_archive import bind

        return bind(source, root, fixture=config.fixture)
    from super_deli.benchmarking.neural_power_study import engine

    study = engine(root, source)
    if args.action == "freeze":
        return study.freeze(config)
    _, stored = study.check()
    if args.study_config and stored != config:
        raise ValueError("Configuration differs from frozen study")
    if args.action == "run":
        return study.run(args.phase, workers=args.workers)
    if args.action == "audit":
        return study.audit(args.phase)
    if args.action == "gate":
        from super_deli.benchmarking.neural_power_budget import gate

        return gate(root, source, staging_bytes=args.staging_bytes)
    if args.action == "report":
        from super_deli.benchmarking.neural_power_report import report

        return report(root, source)
    if args.action == "pause":
        atomic_json(root / "PAUSE.json", dict(reason="Requested between-unit pause"))
    else:
        target = root / "ledger/pauses" / (uuid.uuid4().hex + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        (root / "PAUSE.json").rename(target)
    return dict(passed=True, action=args.action)
