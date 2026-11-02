"""Lazy entry points for the separate point model and its gated fresh study."""

from pathlib import Path

from super_deli.io.config import read_json


def register(commands):
    p = commands.add_parser(
        "cluster-tabular-spectral", help="Experimental unknown-K raw-trait graphs"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--modality")
    p.add_argument("--output-dir", type=Path, required=True)
    p = commands.add_parser("benchmark-unknown-k-spectral", help="Gated fresh unknown-K study")
    p.add_argument(
        "action",
        choices=("freeze", "preflight", "audit-preflight", "gate", "main", "audit-main", "report"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--study-config", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--staging-bytes", type=int)


def dispatch(args):
    if args.command == "cluster-tabular-spectral":
        from super_deli.core.schema import config_from_dict
        from super_deli.models.spectral_config import SpectralConfig
        from super_deli.workflows.tabular_spectral import run_tabular_spectral

        return run_tabular_spectral(
            config_from_dict(read_json(args.config)),
            model_config=SpectralConfig.from_dict(read_json(args.model_config)),
            base_dir=args.config.resolve().parent,
            output_dir=args.output_dir,
            modality_name=args.modality,
        )
    from super_deli.benchmarking.unknown_k_config import StudyConfig
    from super_deli.benchmarking.unknown_k_study import check_frozen, freeze, gate, run_phase

    root = args.output_dir.resolve()
    config = (
        StudyConfig.from_dict(read_json(args.study_config)) if args.study_config else StudyConfig()
    )
    if args.action == "freeze":
        return freeze(root, config)
    _, frozen = check_frozen(root)
    if args.study_config and frozen != config:
        raise ValueError("Study configuration differs from freeze")
    if args.action in {"preflight", "main"}:
        return run_phase(root, args.action, args.workers)
    if args.action == "gate":
        return gate(root, args.staging_bytes)
    if args.action.startswith("audit-"):
        from super_deli.benchmarking.unknown_k_audit import audit_phase

        return audit_phase(root, args.action.removeprefix("audit-"))
    from super_deli.benchmarking.unknown_k_report import report

    return report(root)
