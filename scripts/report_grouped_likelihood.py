"""Publish paired AIC/BIC recovery and likelihood/BIC calibration without refitting."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def plot(root, figures):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paired = read(root / "study-audit/paired.json")["summaries"]
    cases = list(dict.fromkeys(r["case"] for r in paired))
    names = {
        "independent_gaussian": "Independent null",
        "correlated_gaussian": "Correlated null",
        "complementary": "Four groups · 64",
        "heavy_tailed": "Heavy tails",
        "zero_conditional_gaussian": "Zero-rank null",
        "rank_two_gaussian": "Rank-two null",
        "weak_conditional_gaussian": "Weak-rank null",
        "unequal_variance": "Unequal variance",
        "cline": "Continuous cline",
        "complementary_256": "Four groups · 256",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for offset, criterion, color, label in (
        (-0.12, "bic", "#0072B2", "BIC"),
        (0.12, "aic", "#D55E00", "AIC / likelihood gain"),
    ):
        records = [
            next(r for r in paired if r["case"] == case and r["criterion"] == criterion)
            for case in cases
        ]
        x = np.arange(len(cases)) + offset
        for ax, values, intervals in (
            (
                axes[0],
                [
                    r["rejected"] / r["available_tests"] if r["available_tests"] else np.nan
                    for r in records
                ],
                [r["rejection_wilson_95"] for r in records],
            ),
            (
                axes[1],
                [
                    r["selected_k_counts"]["4"] / r["available_points"]
                    if r["available_points"]
                    else np.nan
                    for r in records
                ],
                [r["selected_four_wilson_95"] for r in records],
            ),
        ):
            bounds = np.array([v or [np.nan, np.nan] for v in intervals]).T
            y = np.array(values)
            ax.errorbar(
                x,
                y,
                yerr=np.maximum(0, [y - bounds[0], bounds[1] - y]),
                fmt="o",
                color=color,
                capsize=2,
                label=label,
            )
        alt = [(j, r) for j, r in enumerate(records) if r["ari"]]
        axes[2].errorbar(
            [j + offset for j, r in alt],
            [r["ari"]["mean"] for j, r in alt],
            yerr=[r["ari"]["mean_mcse"] for j, r in alt],
            fmt="o",
            color=color,
            capsize=2,
        )
    for ax, title in zip(
        axes,
        (
            "Global rejection · Wilson 95%",
            "K=4 selection · Wilson 95%",
            "Generating-label ARI · mean ± MCSE",
        ),
        strict=True,
    ):
        ax.set_xticks(
            range(len(cases)), [names[c] for c in cases], rotation=60, ha="right", fontsize=8
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-0.08, 1.08)
        ax.grid(axis="y", alpha=0.2)
    axes[0].axhline(0.05, color="gray", linestyle=":", linewidth=1)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Grouped full-covariance study · paired candidate fits and raw null draws", fontsize=13
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"grouped-likelihood.{suffix}", dpi=180)
    svg = figures / "grouped-likelihood.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def report(root, output, figures):
    output.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "datasets.csv", "rates.csv", "paired.csv", "paired.json"):
        shutil.copyfile(root / "study-audit" / name, output / name)
    for name in (
        "study.json",
        "runtime_sources.json",
        "runtime_versions.json",
        "source-changes.json",
        "legacy-replay.json",
        "reference-pins.json",
        "study-execution.json",
        "format-equivalence.json",
        "independent-tails.json",
    ):
        shutil.copyfile(root / name, output / name)
    candidate_rows = []
    study = read(root / "study.json")
    d = study["calibration"]["model"]["n_pcs"]
    for path in sorted((root / "study").glob("*/*/calibration/observed.json")):
        record = read(path)
        if record["status"] != "complete":
            continue
        case, dataset = path.parts[-4], path.parts[-3]
        for c in record["candidates"]:
            k = c["k"]
            candidate_rows.append(
                {
                    "case": case,
                    "repetition": int(dataset.rsplit("_", 1)[1]),
                    "n": record["n_retained"],
                    "d": d,
                    "parameters": k - 1 + k * d + k * d * (d + 1) // 2,
                    **{
                        key: value
                        for key, value in c.items()
                        if key not in {"hard_cluster_sizes", "distinct_unit_sizes"}
                    },
                    "minimum_size": min(c["hard_cluster_sizes"])
                    if "hard_cluster_sizes" in c
                    else None,
                }
            )
    fields = list(dict.fromkeys(k for row in candidate_rows for k in row))
    with (output / "observed_candidates.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidate_rows)
    plot(root, figures)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    manifest = {
        "local_artifact_id": "aic-grouped-20261023",
        "study_inventory_sha256": digest(root / "study-audit/files.json"),
        "legacy_replay_inventory_sha256": digest(root / "legacy-replay-files.json"),
        "runtime_wheel": read(root / "runtime_package.json"),
        "published_artifacts": {p.name: digest(p) for p in sorted(output.iterdir())},
        "figure_files": {p.name: digest(p) for p in sorted(figures.iterdir())},
        "report_script_sha256": digest(Path(__file__)),
        "interpretation": (
            "Prespecified primary likelihood-gain test and secondary BIC "
            "comparison share fits/draws. "
            "No statistic/family selection or species inference is calibrated; "
            "AIC is a component "
            "selection heuristic. All p-values are approximate fitted-model departure tests."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.artifact_root, args.output_dir, args.figure_dir)


if __name__ == "__main__":
    main()
