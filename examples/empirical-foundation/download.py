"""Download only the explicitly locked CC0 mirror; no network access occurs in CI."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from super_deli.core.empirical import EmpiricalProfile
from super_deli.io.config import read_json
from super_deli.workflows.empirical_curation import fingerprint


def download(profile_path, output_dir):
    profile = EmpiricalProfile.from_dict(read_json(profile_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    partial = out / "source_001.csv.partial"
    try:
        with urlopen(profile.source.url, timeout=45) as response, partial.open("xb") as handle:
            if response.url != profile.source.url:
                raise ValueError("Unexpected redirect from the locked source URL")
            remaining = profile.source.size_bytes
            while remaining:
                chunk = response.read(min(65536, remaining))
                if not chunk:
                    raise ValueError("Source ended before its declared byte count")
                handle.write(chunk)
                remaining -= len(chunk)
            if response.read(1):
                raise ValueError("Source exceeds declared size")
        fingerprint(partial, profile.source)
        partial.rename(out / "source_001.csv")
        (out / "acquisition.json").write_text(
            json.dumps(profile.to_dict()["source"], indent=2) + "\n"
        )
    except Exception as exc:
        (out / "failure.json").write_text(json.dumps(dict(status="failed", error=str(exc))) + "\n")
        raise
    return out / "source_001.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(download(args.profile, args.output_dir))


if __name__ == "__main__":
    main()
