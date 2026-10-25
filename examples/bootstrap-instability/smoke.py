"""Installed four-cell paired diagnostic smoke, with retained subprocess failures."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def command(base, name, arguments):
    result = subprocess.run([sys.executable, *map(str, arguments)], capture_output=True, text=True)
    (base / (name + ".stdout.log")).write_text(result.stdout, encoding="utf-8")
    (base / (name + ".stderr.log")).write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        print(result.stderr, file=sys.stderr)
    result.check_returncode()
    return json.loads(result.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    base = args.output_dir.absolute()
    base.mkdir(exist_ok=False)
    example = Path(__file__).resolve().parent
    study = command(
        base,
        "study",
        [
            "-m",
            "super_deli",
            "benchmark-bootstrap-instability",
            example / "smoke.json",
            "--output-dir",
            base / "study",
            "--workers",
            2,
        ],
    )
    audit = command(
        base,
        "audit",
        [
            "-m",
            "super_deli",
            "analyze-bootstrap-instability",
            base / "study",
            "--output-dir",
            base / "audit",
            "--scratch",
            base / ".scratch",
            "--workers",
            2,
        ],
    )
    check = command(
        base,
        "bytes",
        [example / "check_diagnostic.py", base / "study", "--output", base / "stdlib.json"],
    )
    assert study["successful_fits"] == audit["audited_searches"] == check["audited_searches"] == 168
    assert study["successful_oracles"] == audit["audited_oracles"] == check["audited_oracles"] == 36
    for key in (
        "failed_fits",
        "unattempted_fits",
        "failed_oracles",
        "unattempted_oracles",
        "unavailable_diagnostics",
        "withheld_workflows",
    ):
        assert study[key] == 0
    result = dict(
        status="passed",
        observations=4,
        searches=168,
        oracles=36,
        scope="Reduced smoke only; no performance inference",
    )
    (base / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
