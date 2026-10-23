"""Export audited grouped-mean PCA calibration results without pipeline refits."""

import argparse
import io
from pathlib import Path
from zipfile import ZipFile

from report_shared_calibration import digest, read, save, table
from report_shared_grouped_calibration import NAMES, collect


def plot(summary, output):
    """Show independent-observation rates and pointwise intervals, without refits."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output.mkdir(exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "super-deli-groupmean-calibration-v1",
        }
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    rates = {row["case"]: row for row in summary["rates"]}
    for i, case in enumerate(NAMES):
        rate = rates[case]
        if rate["mean"] is None:
            axis.text(0.03, i, "No available tests")
            continue
        mean, (low, high) = rate["mean"], rate["wilson_95"]
        color = (
            "#D55E00"
            if case == "heavy_tailed"
            else "#009E73"
            if case == "complementary"
            else "#0072B2"
        )
        axis.errorbar(
            mean,
            i,
            xerr=[[max(0, mean - low)], [max(0, high - mean)]],
            fmt="o",
            capsize=4,
            color=color,
            markersize=6,
        )
        axis.annotate(
            f"{rate['events']}/{rate['n_available']} rejected; "
            f"{rate['n_available']}/{summary['study']['n_datasets']} available",
            (mean, i),
            xytext=(-6 if mean > 0.75 else 6, 9),
            textcoords="offset points",
            ha="right" if mean > 0.75 else "left",
            fontsize=8,
        )
    axis.axvline(
        summary["study"]["calibration"]["alpha"],
        color=".5",
        linestyle="--",
        linewidth=1,
        label="Declared alpha = 5%",
    )
    axis.set(
        title="Group-mean PCA: full-pipeline fitted-null calibration",
        xlabel="Rejected / available independent observations",
        xlim=(-0.02, 1.02),
        ylim=(6.65, -0.65),
        yticks=range(7),
        yticklabels=list(NAMES.values()),
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    figure.text(
        0.02,
        0.025,
        "Pointwise Wilson 95% intervals conditional on availability; "
        f"{summary['study']['n_datasets']} observations/scenario. "
        f"{summary['study']['calibration']['n_simulations']} draws/test.\n"
        "Boundary p-values are nonregular and approximate; "
        "no general size or species-boundary guarantee.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.085, 1, 1))
    for extension in ("png", "pdf", "svg"):
        path = output / f"groupmean-calibration.{extension}"
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


def recovery(root, output):
    """Evaluate observed alternative partitions only after all unsupervised fits."""
    import numpy as np

    from super_deli.benchmarking.recovery_analysis import _ari

    hashes = read(root / "study-audit/files.json")
    records = read(root / "study/complementary/summary.json")["datasets"]
    rows = []
    for record in records:
        relative = f"complementary/{record['directory']}"
        truth = root / "study" / relative / "truth.json"
        archive = root / "study" / relative / "calibration/pipeline_artifacts.zip"
        if digest(truth) != hashes[relative + "/truth.json"]:
            raise ValueError("Changed audited truth")
        result = {
            "repetition": record["repetition"],
            "selected_k": record["selected_k"],
            "ari": None,
        }
        if archive.exists():
            if digest(archive) != hashes[relative + "/calibration/pipeline_artifacts.zip"]:
                raise ValueError("Changed audited point archive")
            with ZipFile(archive) as source:
                name = "observed/run/mixture.npz"
                if name in source.namelist():
                    with np.load(io.BytesIO(source.read(name)), allow_pickle=False) as arrays:
                        labels = arrays["labels"].copy()
                    result["ari"] = _ari(read(truth)["labels"], labels)
        rows.append(result)
    table(output / "recovery.csv", rows)
    values = [r["ari"] for r in rows if r["ari"] is not None]
    save(
        output / "recovery_summary.json",
        {
            "scope": (
                "Post-fit generating-partition agreement; "
                "not species truth or selected-K significance"
            ),
            "requested": len(rows),
            "available": len(values),
            "mean_ari": float(np.mean(values)) if values else None,
            "median_ari": float(np.median(values)) if values else None,
            "mean_mcse": float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else None,
        },
    )


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
    declaration = read(args.artifact_root / "study-audit/summary.json")["study"]
    model = declaration["calibration"]["model"]
    morphology = next(v for v in model["views"] if v["kind"] == "morphological")
    if morphology.get("pca_fit") != "target_means" or model["n_pcs"] != 4:
        raise ValueError("Expected the prespecified group-mean/four-joint-PC study")
    summary = collect(args.artifact_root, args.output_dir)
    recovery(args.artifact_root, args.output_dir)
    plot(
        summary,
        args.figure_dir,
    )
    save(
        args.output_dir / "manifest.json",
        {
            "local_artifact_id": "groupmean-null-20261023",
            "frozen_wheel_sha256": read(args.artifact_root / "runtime_package.json")["sha256"],
            "audit_summary_sha256": digest(args.artifact_root / "study-audit/summary.json"),
            "audited_inventory_sha256": digest(args.artifact_root / "study-audit/files.json"),
            "published_artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir())},
        },
    )


if __name__ == "__main__":
    main()
