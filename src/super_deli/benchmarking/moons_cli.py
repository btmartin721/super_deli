"""Lazy registration of fit-free audit and fixed diagnostic study commands."""

from pathlib import Path

from super_deli.io.config import read_json


def register(commands):
    names = {
        "audit-noisy-moons-archive",
        "diagnose-noisy-moons",
        "audit-noisy-moons",
        "preflight-noisy-moons",
    }
    for name in sorted(names):
        parser = commands.add_parser(name, help="Frozen noisy-moons recovery diagnostic")
        parser.add_argument("config", type=Path)
        parser.add_argument("--archive-root", type=Path, required=True)
        if name == "diagnose-noisy-moons":
            parser.add_argument("--phase", choices=("preflight", "main"), required=True)
            parser.add_argument("--archive-audit", type=Path, required=True)
            parser.add_argument("--workers", type=int, default=1)
            parser.add_argument("--gate", type=Path)
        if name in {"audit-noisy-moons", "preflight-noisy-moons"}:
            parser.add_argument("--study-dir", type=Path, required=True)
        if name == "preflight-noisy-moons":
            parser.add_argument("--audit-file", type=Path, required=True)
            parser.add_argument("--artifact-root", type=Path, required=True)
            parser.add_argument("--output-file", type=Path, required=True)
        else:
            parser.add_argument("--output-dir", type=Path, required=True)
    return names


def dispatch(args):
    from super_deli.benchmarking.moons_config import MoonsDiagnosticConfig

    c = MoonsDiagnosticConfig.from_dict(read_json(args.config))
    if args.command == "audit-noisy-moons-archive":
        from super_deli.benchmarking.moons_archive import audit_archive

        return audit_archive(
            args.archive_root, output_dir=args.output_dir, sample_sizes=c.sample_sizes
        )
    if args.command == "audit-noisy-moons":
        from super_deli.benchmarking.moons_audit import audit_moons_diagnostic

        return audit_moons_diagnostic(
            args.study_dir, archive_root=args.archive_root, output_dir=args.output_dir
        )
    from super_deli.benchmarking.moons_study import moons_storage_gate, run_moons_diagnostic

    if args.command == "preflight-noisy-moons":
        return moons_storage_gate(
            c,
            probe_dir=args.study_dir,
            audit_file=args.audit_file,
            artifact_root=args.artifact_root,
            output_file=args.output_file,
        )
    return run_moons_diagnostic(
        c,
        archive_root=args.archive_root,
        phase=args.phase,
        output_dir=args.output_dir,
        archive_audit=args.archive_audit,
        workers=args.workers,
        gate=args.gate,
    )
