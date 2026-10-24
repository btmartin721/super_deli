"""Installed 102-search study/audit plus a six-search public CLI replay."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    from super_deli.benchmarking.local_archives import materialize
    from super_deli.io.config import read_json, write_json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def cli(*values):
        subprocess.run([sys.executable, "-m", "super_deli", *map(str, values)], check=True)

    cli(
        "benchmark-predictive-aic",
        Path(__file__).with_name("smoke.json"),
        "--output-dir",
        root / "study",
        "--workers",
        2,
    )
    cli(
        "analyze-predictive-aic",
        root / "study",
        "--output-dir",
        root / "audit",
        "--scratch",
        root / ".scratch",
        "--workers",
        2,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("check_screening.py")),
            "--study-root",
            str(root / "study"),
            "--audit-root",
            str(root / "audit"),
            "--output",
            str(root / "independent.json"),
        ],
        check=True,
    )
    ledger = [json.loads(s) for s in (root / "study/observations.jsonl").read_text().splitlines()]
    row = next(r for r in ledger if r["cell"] == "independent_gaussian_n64")
    with materialize(root / "study" / row["archive"], row, scratch=root / ".scratch") as base:
        source = read_json(base / "workflow/configuration.json")
        write_json(source["predictive_model"], root / "model.json")
        write_json(source["screening"], root / "screen.json")
        cli(
            "cluster-integrative-predictive",
            base / "observed/dataset.json",
            "--model-config",
            root / "model.json",
            "--screen-config",
            root / "screen.json",
            "--output-dir",
            root / "direct",
        )
        actual = read_json(root / "direct/summary.json")
        assert actual["selection"] == row["selection"]
        assert actual["successful_fits"] == 6
    result = read_json(root / "audit/summary.json")
    assert result["successful_fits"] == 102 and result["failed_fits"] == 0
    assert result["screening_withheld"] == 0 and result["primary_family_size"] == 28
    print(
        json.dumps(
            {
                "predictive_screen_smoke_passed": True,
                "study_searches": 102,
                "public_cli_replay_searches": 6,
                "primary_rates": 28,
                "independent_checker": read_json(root / "independent.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
