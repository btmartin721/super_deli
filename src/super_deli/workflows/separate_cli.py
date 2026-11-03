"""Lazy, additive CLI for separately encoded matched individual views."""

from pathlib import Path

from super_deli.io.config import load_config, read_json


def register(commands):
    parser = commands.add_parser(
        "cluster-matched-separate-neural",
        help="Separate morphology/ecology encoders with four-coordinate clustering",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser = commands.add_parser(
        "benchmark-separate-neural", help="Gated separate-encoder comparison"
    )
    parser.add_argument(
        "action",
        choices=(
            "bind",
            "freeze",
            "preflight",
            "audit-preflight",
            "gate",
            "main",
            "audit-main",
            "report",
        ),
    )
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--study-config", type=Path)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--staging-bytes", type=int)


def dispatch(args):
    if args.command == "benchmark-separate-neural":
        return dispatch_study(args)
    from super_deli.models.separate_config import SeparateNeuralConfig
    from super_deli.workflows.separate_neural import run_separate_neural

    return run_separate_neural(
        load_config(args.config),
        model_config=SeparateNeuralConfig.from_dict(read_json(args.model_config)),
        base_dir=args.config.resolve().parent,
        output_dir=args.output_dir,
    )


def dispatch_study(args):
    from super_deli.benchmarking.separate_config import SeparateStudy

    config = (
        SeparateStudy.from_dict(read_json(args.study_config))
        if args.study_config
        else SeparateStudy()
    )
    root = args.output_dir.resolve()
    if args.action == "bind":
        from super_deli.benchmarking.separate_archive import bind

        return bind(args.archive_root, root, fixture=config.fixture)
    from super_deli.benchmarking.separate_study import (
        audit_phase,
        check_frozen,
        freeze,
        gate,
        run_phase,
    )

    if args.action == "freeze":
        return freeze(root, config)
    _, frozen = check_frozen(root)
    if args.study_config and config != frozen:
        raise ValueError("Configuration differs from freeze")
    if args.action in ("preflight", "main"):
        return run_phase(root, args.archive_root, args.action, workers=args.workers)
    if args.action.startswith("audit-"):
        return audit_phase(root, args.archive_root, args.action.removeprefix("audit-"))
    if args.action == "gate":
        return gate(root, staging_bytes=args.staging_bytes)
    from super_deli.benchmarking.separate_report import report

    return report(root)
