"""Installed eight-cell reduced verification, with retained subprocess diagnostics."""

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
            "benchmark-scaled-fold-stability",
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
            "analyze-scaled-fold-stability",
            base / "study",
            "--output-dir",
            base / "audit",
            "--scratch",
            base / ".scratch",
            "--workers",
            2,
        ],
    )
    checked = command(
        base,
        "bytes",
        [example / "check_study.py", base / "study", "--output", base / "stdlib.json"],
    )
    assert (
        study["successful_fits"] == audit["audited_searches"] == checked["audited_searches"] == 288
    )
    assert study["successful_draws"] == 32
    assert (
        study["failed_draws"]
        == study["failed_fits"]
        == study["unattempted_fits"]
        == study["unattempted_draws"]
        == 0
    )
    result = dict(
        status="passed",
        observations=8,
        searches=288,
        draws=32,
        scope="Reduced engineering smoke only; full validation uses 99 draws and ten starts",
    )
    (base / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
