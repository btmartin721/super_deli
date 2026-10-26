"""Lazy density clustering study commands, available without numerical extras."""

from pathlib import Path

from super_deli.io.config import read_json


def register(commands):
    for name in ("benchmark-tabular-density", "audit-tabular-density", "preflight-tabular-density"):
        parser = commands.add_parser(name, help="Frozen density/GMM comparison and audit")
        parser.add_argument("config", type=Path)
        if name == "benchmark-tabular-density":
            parser.add_argument("--phase", choices=("preflight", "main"), required=True)
            parser.add_argument("--workers", type=int, default=1)
            parser.add_argument("--gate", type=Path)
        else:
            parser.add_argument("--study-dir", type=Path, required=True)
        if name == "preflight-tabular-density":
            parser.add_argument("--audit-file", type=Path, required=True)
            parser.add_argument("--artifact-root", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
    return {"benchmark-tabular-density", "audit-tabular-density", "preflight-tabular-density"}


def dispatch(args):
    from super_deli.benchmarking.density_study import DensityStudyConfig

    config = DensityStudyConfig.from_dict(read_json(args.config))
    if args.command == "audit-tabular-density":
        from super_deli.benchmarking.density_audit import audit_density_study

        return audit_density_study(args.study_dir, output_dir=args.output_dir)
    if args.command == "preflight-tabular-density":
        from super_deli.benchmarking.density_study import storage_gate

        return storage_gate(
            config,
            probe_dir=args.study_dir,
            audit_file=args.audit_file,
            artifact_root=args.artifact_root,
            output_file=args.output_dir,
        )
    from super_deli.benchmarking.density_study import run_density_study

    return run_density_study(
        config, phase=args.phase, output_dir=args.output_dir, workers=args.workers, gate=args.gate
    )
