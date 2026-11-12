"""Invented installed CLI/API and complete sealed study; no empirical specimens."""

import argparse
import json
from pathlib import Path


def main():
    from super_deli.benchmarking.local_reference_fixture import smoke

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    print(json.dumps(smoke(args.output_dir.resolve(), workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
