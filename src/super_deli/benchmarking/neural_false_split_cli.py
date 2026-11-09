"""Additive CLI for the approved, source-frozen larger Gaussian validation."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "benchmark-neural-false-splits",
        help="Batched Gaussian validation of unchanged neural/PCA calibration",
    )
    p.add_argument(
        "action",
        choices=("bind", "freeze", "run", "audit", "finalize", "gate", "report", "pause", "resume"),
    )
    p.add_argument("--archive-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--study-config", type=Path)
    p.add_argument("--phase", choices=("preflight", "main"), default="preflight")
    p.add_argument("--group", default="preflight")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--staging-bytes", type=int)


def dispatch(args):
    import uuid

    from super_deli.benchmarking.neural_false_split_config import NeuralFalseSplitStudy
    from super_deli.benchmarking.neural_false_split_state import atomic_json
    from super_deli.io.config import read_json

    root, source = args.output_dir.resolve(), args.archive_root.resolve()
    config = (
        NeuralFalseSplitStudy.from_dict(read_json(args.study_config))
        if args.study_config
        else NeuralFalseSplitStudy()
    )
    if args.action == "bind":
        from super_deli.benchmarking.neural_false_split_archive import bind

        return bind(source, root, fixture=config.fixture)
    from super_deli.benchmarking.neural_false_split_study import (
        audit_group,
        check_frozen,
        finalize_phase,
        freeze,
        run_group,
    )

    if args.action == "freeze":
        return freeze(root, config)
    _, stored = check_frozen(root)
    if args.study_config and stored != config:
        raise ValueError("Study configuration differs from source freeze")
    if args.action == "run":
        return run_group(root, source, args.phase, args.group, workers=args.workers)
    if args.action == "audit":
        return audit_group(root, source, args.phase, args.group)
    if args.action == "finalize":
        return finalize_phase(root, args.phase)
    if args.action == "gate":
        from super_deli.benchmarking.neural_false_split_budget import gate

        return gate(root, source, staging_bytes=args.staging_bytes)
    if args.action == "report":
        from super_deli.benchmarking.neural_false_split_report import report

        return report(root, source)
    if args.action == "pause":
        atomic_json(root / "PAUSE.json", dict(reason="Requested pause between work units"))
    else:
        target = root / "ledger/pauses" / (uuid.uuid4().hex + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        (root / "PAUSE.json").rename(target)
    return dict(passed=True, action=args.action)
