"""Installed diagnostic CLI over an explicitly invented source archive."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    from super_deli.benchmarking.moons_config import MoonsDiagnosticConfig
    from super_deli.benchmarking.moons_fixture import build_fixture
    from super_deli.io.config import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(exist_ok=False)
    assert build_fixture(out / "archive")["synthetic_fixture"]
    c = MoonsDiagnosticConfig(datasets_per_cell=1, sample_sizes=(32,), min_free_bytes=1)
    write_json(c.to_dict(), out / "configuration.json")

    def run(command, *extra):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "super_deli",
                command,
                str(out / "configuration.json"),
                "--archive-root",
                str(out / "archive"),
                *map(str, extra),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    assert (
        run("audit-noisy-moons-archive", "--output-dir", out / "archive-audit")["retained_draws"]
        == 5
    )
    result = run(
        "diagnose-noisy-moons",
        "--phase",
        "preflight",
        "--archive-audit",
        out / "archive-audit/summary.json",
        "--output-dir",
        out / "preflight",
    )
    assert result["successful_fits"] == 25 and not result["unavailable_hierarchies"]
    assert (
        run("audit-noisy-moons", "--study-dir", out / "preflight", "--output-dir", out / "audit")[
            "audited_successful_fits"
        ]
        == 25
    )
    print("Noisy-moons installed CLI, source replay and hierarchy audit verified")


if __name__ == "__main__":
    main()
