"""Explicit phases for the frozen, local-only missingness study."""

import argparse
import json
from pathlib import Path


def main():
    from super_deli.benchmarking.tabular_missing_audit import audit_study
    from super_deli.benchmarking.tabular_missing_study import run_validation, storage_gate

    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["preflight", "main", "audit", "gate"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--study", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--scratch", type=Path)
    p.add_argument("--gate", type=Path)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if a.phase == "audit":
        r = audit_study(a.study, output_file=a.output, workers=a.workers)
        print(json.dumps({k: v for k, v in r.items() if k != "records"}))
    elif a.phase == "gate":
        r = storage_gate(
            a.study,
            a.audit,
            checkpoint_dir=a.checkpoint,
            scratch_dir=a.scratch,
            output_file=a.output,
        )
        print(json.dumps({k: v for k, v in r.items() if k not in {"classes", "source_identity"}}))
    else:
        r = run_validation(a.phase, output_dir=a.output, workers=a.workers, gate_file=a.gate)
        print(json.dumps({k: v for k, v in r.items() if k != "records"}))


if __name__ == "__main__":
    main()
