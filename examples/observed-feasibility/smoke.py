"""Invented installed feasibility lifecycle; no empirical inputs."""

import argparse
import json
from pathlib import Path


def main():
    from super_deli.benchmarking.observed_feasibility_fixture import smoke

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    print(json.dumps(smoke(args.output_dir, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
