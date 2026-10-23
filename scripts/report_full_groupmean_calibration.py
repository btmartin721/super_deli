"""Export audited full-covariance group-mean calibration evidence without refits."""

import argparse
from pathlib import Path

from grouped_calibration_reporting import collect
from report_groupmean_calibration import plot, recovery
from report_shared_calibration import digest, read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_dir, args.figure_dir):
        if any(
            output.resolve().is_relative_to((args.artifact_root / n).resolve())
            for n in ("study", "study-audit")
        ):
            raise ValueError("Reports must be outside immutable study/audit directories")
    model = read(args.artifact_root / "study-audit/summary.json")["study"]["calibration"]["model"]
    morphology = next(v for v in model["views"] if v["kind"] == "morphological")
    if morphology.get("pca_fit") != "target_means" or model["n_pcs"] != 4:
        raise ValueError("Expected the frozen group-mean/four-joint-PC declaration")
    summary = collect(args.artifact_root, args.output_dir, covariance_type="full")
    recovery(args.artifact_root, args.output_dir)
    plot(summary, args.figure_dir)
    for path in tuple(args.figure_dir.iterdir()):
        path.rename(path.with_name("full-" + path.name))
    save(
        args.output_dir / "manifest.json",
        {
            "local_artifact_id": "full-groupmean-null-20261023",
            "frozen_wheel_sha256": read(args.artifact_root / "runtime_package.json")["sha256"],
            "audit_summary_sha256": digest(args.artifact_root / "study-audit/summary.json"),
            "audited_inventory_sha256": digest(args.artifact_root / "study-audit/files.json"),
            "published_artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir())},
        },
    )


if __name__ == "__main__":
    main()
