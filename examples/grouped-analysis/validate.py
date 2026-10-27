"""Explicit guarded validation phases; detailed study artifacts stay outside Git."""

import argparse
import json
from pathlib import Path

from super_deli.benchmarking.grouped_analysis_audit import audit_study
from super_deli.benchmarking.grouped_analysis_study import run_study, storage_gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("preflight", "main", "audit", "gate"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--study-dir", type=Path)
    p.add_argument("--audit-file", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--gate-file", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--artifact-cap-bytes", type=int)
    a = p.parse_args()
    if a.phase == "audit":
        if a.study_dir is None:
            p.error("--study-dir required")
        result = audit_study(a.study_dir, a.output, workers=a.workers)
    elif a.phase == "gate":
        if any(x is None for x in (a.study_dir, a.audit_file, a.artifact_root)):
            p.error("--study-dir, --audit-file and --artifact-root required")
        result = storage_gate(
            a.study_dir,
            a.audit_file,
            a.artifact_root,
            a.output,
            artifact_cap_bytes=a.artifact_cap_bytes,
        )
    else:
        if a.phase == "main" and a.gate_file is None:
            p.error("--gate-file required")
        result = run_study(a.phase, output_dir=a.output, workers=a.workers, gate_file=a.gate_file)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
