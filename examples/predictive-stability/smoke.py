"""Installed eight-cell smoke: complete draw refits and two independent auditors."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def command(base, name, *arguments):
    result = subprocess.run(
        [sys.executable, "-m", "super_deli", *map(str, arguments)],
        capture_output=True,
        text=True,
    )
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
        "benchmark-predictive-stability",
        example / "smoke.json",
        "--output-dir",
        base / "study",
        "--workers",
        2,
    )
    audit = command(
        base,
        "audit",
        "analyze-predictive-stability",
        base / "study",
        "--output-dir",
        base / "audit",
        "--scratch",
        base / ".scratch",
        "--workers",
        2,
    )
    subprocess.run(
        [
            sys.executable,
            example / "check_stability.py",
            base / "study",
            "--output-file",
            base / "stdlib.json",
        ],
        check=True,
    )
    assert study["successful_fits"] == audit["audited_searches"] == 144
    assert study["failed_draws"] == study["failed_fits"] == study["unattempted_fits"] == 0
    assert json.loads((base / "stdlib.json").read_text())["audited_searches"] == 144
    result = dict(
        status="passed",
        observations=8,
        searches=144,
        draws=16,
        n_init=2,
        n_bootstrap=2,
        scope="Reduced smoke only; full validation uses ten starts and 99 draws",
    )
    with (base / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
