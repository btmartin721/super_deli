"""Installed full lifecycle on invented N=32 fixtures; never launches the real study."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from super_deli.benchmarking.neural_false_split_fixture import create


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    out = parser.parse_args().output_dir
    out.mkdir()
    source, root = out / "source", out / "study"
    create(source)
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented installed-command fixture only\n")
    policy = out / "fixture.json"
    policy.write_text(json.dumps(dict(schema_version=1, fixture=True)))
    commands = [("bind", "preflight", "preflight"), ("freeze", "preflight", "preflight")]
    commands += [(a, "preflight", "preflight") for a in ("run", "audit", "finalize", "gate")]
    commands += [(a, "main", group) for group in ("batch_00", "extras") for a in ("run", "audit")]
    commands += [("finalize", "main", "extras"), ("report", "main", "extras")]
    for i, (action, phase, group) in enumerate(commands):
        command = [
            sys.executable,
            "-m",
            "super_deli",
            "benchmark-neural-false-splits",
            action,
            "--archive-root",
            str(source),
            "--output-dir",
            str(root),
            "--study-config",
            str(policy),
            "--phase",
            phase,
            "--group",
            group,
        ]
        if action == "run":
            command.extend(["--workers", "2"])
        if action == "gate":
            command.extend(["--staging-bytes", "0"])
        result = subprocess.run(command, capture_output=True, text=True)
        (out / f"command_{i:02d}.log").write_text(result.stderr + "\n" + result.stdout)
        assert result.returncode == 0, (action, result.stdout[-2000:], result.stderr[-4000:])
    summary = json.loads((root / "summary.json").read_text())
    assert summary["fixture"] and summary["observations"] == 2
    assert summary["counts"]["main"]["workflows"] == 26
    assert summary["audits"]["main"]["verified_optimizer_updates"] == 13000
    assert summary["audits"]["preflight"]["verified_optimizer_updates"] == 10000
    print("Installed Gaussian false-split study, two-worker refits and archival parity passed")


if __name__ == "__main__":
    main()
