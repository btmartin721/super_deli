"""Benchmark-only module entry point; production model CLI stays unchanged."""

import argparse
import json
from pathlib import Path

from super_deli.io.config import read_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=("freeze", "preflight", "audit-preflight", "gate", "main", "audit-main", "report"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--study-config", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--staging-bytes", type=int)
    args = p.parse_args(argv)
    # Lazy scientific imports keep help and dependency-light installation usable.
    from super_deli.benchmarking.gaussian_graph_config import StudyConfig
    from super_deli.benchmarking.gaussian_graph_study import check_frozen, freeze, gate, run_phase

    root = args.output_dir.resolve()
    config = (
        StudyConfig.from_dict(read_json(args.study_config)) if args.study_config else StudyConfig()
    )
    if args.action == "freeze":
        result = freeze(root, config)
    else:
        _, frozen = check_frozen(root)
        if args.study_config and config != frozen:
            raise ValueError("Configuration differs from freeze")
        if args.action in {"preflight", "main"}:
            result = run_phase(root, args.action, args.workers)
        elif args.action == "gate":
            result = gate(root, args.staging_bytes)
        elif args.action.startswith("audit-"):
            from super_deli.benchmarking.gaussian_graph_audit import audit_phase

            result = audit_phase(root, args.action.removeprefix("audit-"))
        else:
            from super_deli.benchmarking.gaussian_graph_report import report

            result = report(root)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
