"""Export paired changes and baseline retention/loadings from a completed audit."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_table(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(study, audit, destination):
    points = json.loads((study / "points.json").read_text())
    verified = json.loads((audit / "summary.json").read_text())
    if verified["successful_points"] != len(points) or any(
        p["status"] != "complete" for p in points
    ):
        raise ValueError("Supplemental tables require the audited complete point grid")
    lookup = {(p["cohort"], p["repetition"], p["variant"]): p for p in points}
    cohorts = tuple(dict.fromkeys(p["cohort"] for p in points))
    variants = tuple(dict.fromkeys(p["variant"] for p in points))
    paired, retention, loadings = [], [], []
    for cohort in cohorts:
        indices = sorted(
            p["repetition"] for p in points if p["cohort"] == cohort and p["variant"] == "baseline"
        )
        for variant in variants[1:]:
            differences = [
                lookup[cohort, i, variant]["metrics"]["ari"]
                - lookup[cohort, i, "baseline"]["metrics"]["ari"]
                for i in indices
            ]
            paired.append(
                {
                    "cohort": cohort,
                    "variant": variant,
                    "n_paired": len(indices),
                    "mean_ari_change": float(np.mean(differences)),
                    "median_ari_change": float(np.median(differences)),
                    "ari_increased": sum(d > 1e-9 for d in differences),
                    "ari_decreased": sum(d < -1e-9 for d in differences),
                    "ari_unchanged": sum(abs(d) <= 1e-9 for d in differences),
                    "selected_k_changed": sum(
                        lookup[cohort, i, variant]["selected_k"]
                        != lookup[cohort, i, "baseline"]["selected_k"]
                        for i in indices
                    ),
                }
            )
        for i in indices:
            metrics = lookup[cohort, i, "baseline"]["metrics"]
            spaces = {
                **metrics["view_axis_r2"],
                "fused": metrics["fused_axis_r2"],
                "joint": metrics["joint_axis_r2"],
            }
            for name, values in spaces.items():
                retention.append(
                    {
                        "cohort": cohort,
                        "repetition": i,
                        "space": name,
                        "a_r2": values["a"],
                        "b_r2": values["b"],
                    }
                )
            for j, variance in enumerate(metrics["joint_variance_ratio"]):
                loadings.append(
                    {
                        "cohort": cohort,
                        "repetition": i,
                        "joint_pc": j + 1,
                        "explained_variance_ratio": variance,
                        **{
                            kind: values[j]
                            for kind, values in metrics["joint_squared_loading_by_view"].items()
                        },
                    }
                )
    destination.mkdir(exist_ok=False)
    for name, rows in (
        ("paired.csv", paired),
        ("retention.csv", retention),
        ("loadings.csv", loadings),
    ):
        write_table(destination / name, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.study, args.audit, args.output_dir)
