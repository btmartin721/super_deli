"""Invented installed check for known-only support and group exclusions."""

import argparse
import json
from pathlib import Path

from super_deli.benchmarking.known_reference_fixture import smoke

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(smoke(args.output_dir), indent=2, sort_keys=True, allow_nan=False))
