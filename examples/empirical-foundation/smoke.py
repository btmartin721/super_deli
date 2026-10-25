"""Offline installed CLI smoke, using invented records and no estimator imports."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parent / "fixture"
    command = [
        sys.executable,
        "-m",
        "super_deli",
        "curate-empirical",
        str(fixture / "profile.json"),
        "--source",
        str(fixture / "source.csv"),
        "--protocol",
        str(fixture / "protocol.json"),
        "--output-dir",
        str(args.output_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    summary = json.loads(result.stdout)
    assert summary["source_rows"] == 12 and summary["included_rows"] == 10
    assert summary["readiness"]["statistical_fitting"] is False
    assert summary["ingestion_status"] == "complete"
    print(
        json.dumps(
            dict(
                status="passed",
                synthetic_source_rows=12,
                included_rows=10,
                clustering_fits=0,
                network_requests=0,
            )
        )
    )


if __name__ == "__main__":
    main()
