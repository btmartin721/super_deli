"""Publish portable tables and figures from the audited shared-family joint study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

NAMES = {
    "zero_association": "Zero association · null",
    "linear_association": "Linear association · null",
    "complementary": "Complementary four groups · alternative",
    "heavy_tailed": "Heavy-tailed residuals · stress",
    "cline": "Continuous cline · stress",
}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, sorted(set().union(*(row.keys() for row in rows))), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def collect(root, output):
    """Use checked relative paths; publish no source paths, raw data or specimen IDs."""
    import numpy as np

    audit = read(root / "study-audit/summary.json")
    hashes = read(root / "study-audit/files.json")
    if audit["study"]["calibration"]["model"].get("covariance_type") != "tied":
        raise ValueError("Expected a prespecified shared-covariance study")

    def verified(relative):
        path = root / "study" / relative
        if digest(path) != hashes.get(relative):
            raise ValueError(f"Changed audited input: {relative}")
        return path

    output.mkdir(exist_ok=False)
    records, diagnostics, sizes = [], [], {}
    for case in NAMES:
        shard = read(verified(f"{case}/summary.json"))
        counts = Counter()
        for row in shard["datasets"]:
            prefix = f"{case}/{row['directory']}/calibration"
            # Outer/null-fit failures retain their original ledger without replacement.
            diagnostic = {"case": case, "repetition": row["repetition"]}
            if row["outcome_file"] == "summary.json":
                summary = read(verified(f"{prefix}/summary.json"))
                inference, fit = summary["inference"], summary["null_fit"]
                bounds = inference.get("conditional_null_tail_interval_95")
                diagnostic.update(
                    exceedances=inference.get("exceedances"),
                    tail_lower_95=bounds[0] if bounds else None,
                    tail_upper_95=bounds[1] if bounds else None,
                    successful_draws=inference["successful"],
                    failed_draws=inference["failed"],
                    design_rank=fit["design_rank"],
                    residual_df=fit["residual_df"],
                    observed_genotype_profiles=fit["genomic"]["n_observed_profiles"],
                )
                with np.load(verified(f"{prefix}/null_model.npz"), allow_pickle=False) as arrays:
                    covariance = arrays["residual_covariance"]
                    scales = np.sqrt(np.diag(covariance))
                    diagnostic["residual_correlation_min_eigenvalue"] = float(
                        np.linalg.eigvalsh(covariance / scales[:, None] / scales[None, :])[0]
                    )
            diagnostic.update(
                statistic=row["statistic"],
                p_value=row["p_value"],
                reject=row["reject"],
                status=row["status"],
            )
            diagnostics.append(diagnostic)
            records.append(row)
            if row["selected_k"] is not None:
                counts[str(row["selected_k"])] += 1
        sizes[case] = {
            str(k): counts[str(k)]
            for k in range(1, audit["study"]["calibration"]["model"]["max_k"] + 1)
        }
    table(output / "datasets.csv", records)
    table(output / "diagnostics.csv", diagnostics)
    save(output / "summary.json", {**audit, "selected_k_counts": sizes})
    for name in ("runtime_sources.json", "runtime_versions.json", "study.json"):
        save(output / name, read(root / name))
    save(output / "execution.json", read(root / "study-execution.json"))
    return audit


def plot(summary, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output.mkdir(exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "super-deli-shared-calibration-v1",
        }
    )
    figure, axis = plt.subplots(figsize=(9, 4.8))
    rates = {row["case"]: row for row in summary["rates"]}
    for i, case in enumerate(NAMES):
        rate = rates[case]
        if rate["mean"] is None:
            continue
        mean, (low, high) = rate["mean"], rate["wilson_95"]
        axis.errorbar(
            mean,
            i,
            xerr=[[max(0, mean - low)], [max(0, high - mean)]],
            fmt="o",
            color="#0072B2" if i < 2 else "#009E73" if i == 2 else "#D55E00",
            capsize=4,
            markersize=7,
        )
        axis.annotate(
            f"{rate['events']}/{rate['n_available']}",
            (mean, i),
            xytext=(-6 if mean > 0.9 else 6, 9),
            textcoords="offset points",
            ha="right" if mean > 0.9 else "left",
        )
    axis.axvline(0.05, color="#777777", linestyle="--", linewidth=1, label="Declared alpha = 5%")
    axis.set(
        title="Shared-covariance full-pipeline joint-null calibration",
        xlabel="Rejected / available independent datasets",
        xlim=(-0.02, 1.02),
        ylim=(4.55, -0.65),
        yticks=list(range(5)),
        yticklabels=list(NAMES.values()),
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False)
    figure.text(
        0.02,
        0.025,
        f"Pointwise Wilson 95% intervals; {summary['study']['n_datasets']} datasets/scenario. "
        "Stress rejection measures misspecification sensitivity.\n"
        "Approximate fitted-null tests; no general size guarantee "
        "or species-boundary significance.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    for extension in ("png", "pdf", "svg"):
        path = output / f"shared-calibration.{extension}"
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if extension == "pdf"
            else {"Date": None}
            if extension == "svg"
            else {}
        )
        figure.savefig(path, dpi=180, metadata=metadata)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_dir, args.figure_dir):
        if any(
            output.resolve().is_relative_to((args.artifact_root / name).resolve())
            for name in ("study", "study-audit")
        ):
            raise ValueError("Reports must be outside immutable study/audit directories")
    summary = collect(args.artifact_root, args.output_dir)
    plot(summary, args.figure_dir)
    save(
        args.output_dir / "manifest.json",
        {
            "local_artifact_id": "shared-null-20261022",
            "frozen_wheel_sha256": read(args.artifact_root / "runtime_package.json")["sha256"],
            "audit_summary_sha256": digest(args.artifact_root / "study-audit/summary.json"),
            "audited_inventory_sha256": digest(args.artifact_root / "study-audit/files.json"),
            "published_artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir())},
        },
    )


if __name__ == "__main__":
    main()
