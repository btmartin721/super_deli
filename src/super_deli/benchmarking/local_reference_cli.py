"""Explicit sealed class-local assessment lifecycle; old benchmarks stay unchanged."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "benchmark-local-reference-support", help="Bounded class-local support validation"
    )
    p.add_argument("action", choices=("bind", "freeze", "run", "audit", "gate", "report"))
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--study-config", type=Path)
    p.add_argument("--phase", choices=("preflight", "main"), default="preflight")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--staging-bytes", type=int, default=0)
    p.add_argument("--fixture", action="store_true")


def dispatch(args):
    from super_deli.benchmarking.local_reference_config import LocalReferenceStudy
    from super_deli.benchmarking.local_reference_report import report
    from super_deli.benchmarking.local_reference_source import bind
    from super_deli.benchmarking.local_reference_study import engine, gate, seed_registry
    from super_deli.io.config import read_json, write_json

    root, source = args.output_dir, args.archive_root
    if args.action == "bind":
        return bind(source, root, fixture=args.fixture)
    study = engine(root, source)
    if args.action == "freeze":
        config = (
            LocalReferenceStudy.from_dict(read_json(args.study_config))
            if args.study_config
            else LocalReferenceStudy(fixture=args.fixture)
        )
        write_json(seed_registry(config), root / "seed-registry.json")
        return study.freeze(config)
    if args.action == "run":
        return study.run(args.phase, workers=args.workers)
    if args.action == "audit":
        return study.audit(args.phase)
    if args.action == "gate":
        return gate(root, source, staging_bytes=args.staging_bytes)
    return report(root, source)
