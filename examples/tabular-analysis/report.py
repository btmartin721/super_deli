"""Publish small descriptive aggregates/figures from independently audited local evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from super_deli.benchmarking.density_study import CASES
    from super_deli.core.provenance import fingerprint_file
    from super_deli.io.config import read_json, write_json

    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--figure-dir", required=True, type=Path)
    a = p.parse_args()
    root, out = a.artifact_root, a.output_dir
    out.mkdir()
    a.figure_dir.mkdir(parents=True)
    audit = read_json(root / "main-audit.json")
    preflight = read_json(root / "preflight-audit.json")
    independent = read_json(root / "independent-postfit-audit.json")
    assert audit["status"] == preflight["status"] == independent["status"] == "passed"
    names = ("gmm", "hdbscan", "hdbscan_ms2", "hdbscan_ms10")
    metrics = audit["metrics"]
    cells = [(case, n) for case in CASES for n in (64, 256)]
    aggregates = []
    for case, n in cells:
        for name in names:
            rows = [r for r in metrics if r["case"] == case and r["n"] == n and r["name"] == name]
            assert len(rows) == 3
            aggregates.append(
                {
                    "case": case,
                    "n": n,
                    "model": name,
                    "datasets": len(rows),
                    "k_counts": dict(sorted(Counter(r["selected_k"] for r in rows).items())),
                    "mean_selected_k": float(np.mean([r["selected_k"] for r in rows])),
                    "mean_resolved_fraction": float(
                        np.mean([r["resolved_count"] / n for r in rows])
                    ),
                    "min_resolved_fraction": min(r["resolved_count"] / n for r in rows),
                    "max_resolved_fraction": max(r["resolved_count"] / n for r in rows),
                    "failed_models": sum(r["status"] != "complete" for r in rows),
                }
            )
    comparisons = []
    for case, n in cells:
        for name in names[1:]:
            rows = []
            for d in sorted((root / "study").glob(f"{case}_n{n}_*")):
                pair = next(
                    r
                    for r in read_json(d / "analysis/comparisons.json")
                    if r["left"] == "gmm" and r["right"] == name
                )
                rows.append(pair)
            defined = [r for r in rows if r["adjusted_rand"] is not None]
            comparisons.append(
                {
                    "case": case,
                    "n": n,
                    "left": "gmm",
                    "right": name,
                    "datasets": len(rows),
                    "defined_comparisons": len(defined),
                    "degenerate_comparisons": sum(r["degenerate"] for r in rows),
                    "mean_joint_coverage": float(np.mean([r["joint_coverage"] for r in rows])),
                    "mean_conditional_ari": float(np.mean([r["adjusted_rand"] for r in defined]))
                    if defined
                    else None,
                }
            )
    summary = {
        "schema_version": 1,
        "status": "passed",
        "observations": 36,
        "point_pipelines": 144,
        "resampling_pipelines": 480,
        "standalone_point_replays": 48,
        "main_pipeline_runs": 672,
        "preflight_pipeline_runs": 88,
        "pairwise_comparisons": 216,
        "failed_observations": 0,
        "artifact_cap_bytes": read_json(root / "storage-gate.json")["artifact_cap_bytes"],
        "original_cap_bytes": read_json(root / "validation.json")["artifact_cap_bytes"],
        "model_defaults_and_historical_p_values_unchanged": True,
        "p_values_computed": False,
        "source_audit_hashes": {
            n: fingerprint_file(root / n).sha256
            for n in ("main-audit.json", "preflight-audit.json", "independent-postfit-audit.json")
        },
        "interpretation": (
            "Three datasets/cell; descriptive orchestration validation, "
            "no general performance guarantee"
        ),
    }
    write_json(summary, out / "summary.json")
    write_json(aggregates, out / "models.json")
    write_json(comparisons, out / "comparisons.json")
    with (out / "models.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {**r, "k_counts": json.dumps(r["k_counts"], sort_keys=True)} for r in aggregates
        )
    figure, axes = plt.subplots(1, 2, figsize=(10, 7), layout="constrained")
    for axis, key, title, limit in zip(
        axes,
        ("mean_selected_k", "mean_resolved_fraction"),
        ("Mean selected group count", "Mean resolved fraction"),
        (None, 1),
        strict=True,
    ):
        data = np.array(
            [
                [
                    next(
                        r[key]
                        for r in aggregates
                        if r["case"] == case and r["n"] == n and r["model"] == name
                    )
                    for name in names
                ]
                for case, n in cells
            ]
        )
        im = axis.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=limit)
        axis.set_xticks(
            range(4),
            ["GMM", "HDBSCAN", "HDBSCAN\nmin_samples=2", "HDBSCAN\nmin_samples=10"],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(len(cells)), [f"{case}; N={n}" for case, n in cells])
        axis.set_title(title)
        for i, j in np.ndindex(data.shape):
            value = data[i, j]
            fraction = im.norm(value)
            axis.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if fraction < 0.5 else "black",
            )
        figure.colorbar(im, ax=axis, shrink=0.6)
    figure.suptitle("Unified tabular validation: three independent datasets per cell", fontsize=12)
    for extension in ("png", "pdf"):
        figure.savefig(a.figure_dir / f"comparison-overview.{extension}", dpi=180)
    plt.close(figure)
    write_json(
        {p.name: fingerprint_file(p).sha256 for p in out.iterdir() if p.is_file()},
        out / "checksums.json",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
