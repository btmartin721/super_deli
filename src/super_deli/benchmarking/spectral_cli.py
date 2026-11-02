"""Lazy archival diagnosis CLI; it is not a species-discovery model."""

from pathlib import Path

from super_deli.io.config import read_json


def register(commands):
    p = commands.add_parser(
        "diagnose-spectral-graphs", help="Supplied-K spectral graph diagnosis on frozen coordinates"
    )
    p.add_argument(
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
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--staging-bytes", type=int)


def dispatch(args):
    from super_deli.benchmarking.spectral_config import SpectralConfig

    config = SpectralConfig.from_dict(read_json(args.config)) if args.config else SpectralConfig()
    root = args.output_dir.resolve()
    if args.action == "bind":
        from super_deli.benchmarking.spectral_archive import bind_archive

        binding = bind_archive(args.archive_root, root, fixture=config.fixture)
        return {
            "passed": binding["passed"],
            "kind": binding["kind"],
            "observations": len(binding["observations"]),
            "source_files": len(binding["source_sha256"]),
            "neural_training_fits": 0,
            "binding": str(root / "binding.json"),
        }
    from super_deli.benchmarking.spectral_study import check_frozen, freeze, gate, run_phase

    if args.action == "freeze":
        return freeze(root, config)
    _, frozen_config = check_frozen(root)
    if args.config and config != frozen_config:
        raise ValueError("Configuration differs from its freeze")
    if args.action in ("preflight", "main"):
        return run_phase(root, args.action, args.workers)
    if args.action == "gate":
        if args.staging_bytes is None:
            raise ValueError("Measure and supply --staging-bytes")
        return gate(root, args.staging_bytes)
    if args.action.startswith("audit-"):
        from super_deli.benchmarking.spectral_audit import audit_phase

        return audit_phase(root, args.action.removeprefix("audit-"))
    from super_deli.benchmarking.spectral_report import report

    return report(root)
