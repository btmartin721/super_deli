"""Explicit, immutable phases; keep detailed output outside the repository."""

import argparse
import json
import os


def verify_freeze(directory):
    import hashlib
    from importlib.metadata import version
    from pathlib import Path

    from super_deli.benchmarking.observed_study import source_identity
    from super_deli.io.config import read_json

    root = Path(directory)
    frozen = read_json(
        root
        / (
            "execution-freeze.json"
            if (root / "execution-freeze.json").exists()
            else "frozen-before-generation.json"
        )
    )
    if frozen["source_sha256"] != source_identity():
        raise ValueError("Frozen source mismatch")
    if any(version(name) != value for name, value in frozen["software"].items()):
        raise ValueError("Frozen scientific software versions changed")
    if (
        "runner_sha256" in frozen
        and frozen["runner_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("Frozen runner changed; review an explicit amendment")


def main():
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    from super_deli.benchmarking.observed_audit import aggregate, audit_phase
    from super_deli.benchmarking.observed_study import run_validation, storage_gate

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=["freeze", "preflight", "main", "audit-preflight", "audit-main", "gate", "report"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "freeze":
        import hashlib
        from importlib.metadata import version
        from pathlib import Path

        from super_deli import __version__
        from super_deli.benchmarking.observed_study import NAMES, SETTINGS, source_identity
        from super_deli.io.config import write_json
        from super_deli.models.observed_config import ObservedGMMConfig

        software = {
            name: version(name)
            for name in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
        }
        if software["super-deli"] != __version__:
            raise ValueError("Installed package metadata and imported source differ")
        root = Path(args.output_dir)
        root.mkdir(parents=True, exist_ok=False)
        result = dict(
            status="frozen_before_fresh_generation",
            settings=SETTINGS,
            model=ObservedGMMConfig(NAMES).to_dict(),
            source_sha256=source_identity(),
            software=software,
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            empirical_fits=False,
            pvalues=False,
            bootstrap=False,
        )
        write_json(result, root / "frozen-before-generation.json")
        print("Frozen settings and sources; no scientific fits generated")
        return
    verify_freeze(args.output_dir)
    if args.phase in ("preflight", "main"):
        result = run_validation(args.output_dir, args.phase, args.workers, resume=args.resume)
    elif args.phase.startswith("audit-"):
        result = audit_phase(args.output_dir, args.phase[6:], args.workers)
    elif args.phase == "gate":
        result = storage_gate(args.output_dir)
    else:
        result = aggregate(args.output_dir)
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
