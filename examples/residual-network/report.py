"""Export aggregate-only results and figures from an audited local study."""

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def export(study, destination):
    audit = read(study / "report-qa.json")
    if not audit["passed"]:
        raise ValueError("Independent report verification required")
    destination.mkdir(exist_ok=False)
    groups = read(study / "groups.json")
    for group in groups:
        group.pop("paired_accuracy_gain_values", None)
    write(destination / "groups.json", groups)
    for name in ("summary.json", "primary-rates.json"):
        write(destination / name, read(study / name))
    frozen = read(study / "frozen.json")
    write(
        destination / "study-design.json",
        {
            "version": frozen["package_version"],
            "settings": frozen["settings"],
            "counts": frozen["counts"],
            "training": frozen["training"],
            "independent_datasets_per_cell": 5,
            "known_class_evaluation_units": 2048,
            "shifted_evaluation_units": 2048,
            "unfamiliar_class_units": 512,
            "network_parameters": {"residual": 4772, "separate_autoencoders": 1072},
            "parameter_note": (
                "Neural weights/biases only; labels, objectives and training differ. "
                "PCA/GMM estimated parameters are retained in native local evidence."
            ),
            "software": frozen["software"],
            "source_scope": "Independent matched invented continuous measurements",
        },
    )
    write(
        destination / "independent-audit.json",
        {k: v for k, v in audit.items() if k not in {"seconds"}},
    )
    export_seed_sensitivity(study, destination)


def export_seed_sensitivity(study, destination):
    cells = {}
    for observation in read(study / "assessments.json"):
        task = observation["task"]
        for row in observation["fits"]:
            if row["task"] != "unsupervised_discovery":
                continue
            cells.setdefault((task["case"], task["n"], row["arm"]), []).append(row)
    result = []
    for (case, n, arm), rows in sorted(cells.items()):
        available = [r for r in rows if r["status"] == "available"]
        result.append(
            {
                "case": case,
                "n": n,
                "arm": arm,
                "requested": len(rows),
                "available": len(available),
                "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in available)),
                "four_group_recovery_count": sum(
                    bool(r["metrics"]["iid"].get("four_group_recovery", False)) for r in available
                )
                if case not in ("gaussian", "gradient")
                else None,
                "interpretation": "Seeds share datasets and are not independent replicates",
            }
        )
    write(destination / "seed-sensitivity.json", result)


def save_figure(fig, directory, stem):
    for suffix in ("png", "svg", "pdf"):
        path = directory / (stem + "." + suffix)
        fig.savefig(path, dpi=190, bbox_inches="tight")
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )


def figures(results, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, rates = read(results / "groups.json"), read(results / "primary-rates.json")
    cases = [
        "linear_weak",
        "linear_standard",
        "allometric",
        "saturating",
        "combined",
        "confounded_size",
    ]
    names = [
        "Weak linear",
        "Standard linear",
        "Allometry",
        "Saturation",
        "Allometry + saturation",
        "Confounded size",
    ]
    directory.mkdir(exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    data = {"supervised": [], "discovery": []}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    arms = ("logistic", "residual_0", "residual_101", "residual_202")
    colors = ("#333333", "#0072B2", "#009E73", "#CC79A7")
    labels = ("Logistic", "Residual seed 0", "Residual seed 101", "Residual seed 202")
    for ax, n in zip(axes, (64, 256), strict=True):
        for j, case in enumerate(cases):
            group = next(g for g in groups if g["case"] == case and g["n"] == n)
            by_arm = {a["arm"]: a for a in group["arms"]}
            for i, arm in enumerate(arms):
                value = by_arm[arm]["cohorts"]["iid"]["balanced_accuracy"]
                y = j + (i - 1.5) * 0.14
                ax.errorbar(
                    value["mean"],
                    y,
                    xerr=[[value["mean"] - value["minimum"]], [value["maximum"] - value["mean"]]],
                    fmt="o",
                    markersize=4,
                    capsize=2,
                    color=colors[i],
                    label=labels[i] if j == 0 and n == 64 else None,
                )
                data["supervised"].append(dict(case=case, n=n, arm=arm, **value))
        ax.set(
            xlim=(0.5, 1.01),
            ylim=(5.5, -0.5),
            xlabel="Known-class balanced accuracy",
            title=f"N = {n} labelled training specimens",
        )
        ax.set_yticks(range(6), names)
        ax.grid(axis="x", color=".9", linewidth=0.6)
    fig.suptitle("Supervised reference assignment on independent known-class specimens", y=0.99)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.text(
        0.5,
        0.005,
        "Five datasets/law. Points are means; whiskers are observed dataset ranges, "
        "not confidence intervals.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    save_figure(fig, directory, "reference-accuracy")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, n in zip(axes, (64, 256), strict=True):
        for j, case in enumerate(cases):
            for i, arm in enumerate(("pca4", "neural_0")):
                value = next(
                    r
                    for r in rates
                    if r["case"] == case
                    and r["n"] == n
                    and r["arm"] == arm
                    and r["metric"] == "four_group_recovery"
                )
                fraction = value["count"] / value["requested"]
                low, high = value["descriptive_95_interval"]
                y = j + (i - 0.5) * 0.22
                ax.errorbar(
                    fraction,
                    y,
                    xerr=[[fraction - low], [high - fraction]],
                    fmt="o" if i == 0 else "D",
                    color=colors[i],
                    markersize=4,
                    capsize=2,
                    label=("PCA/shared GMM" if i == 0 else "Separate encoders/shared GMM, seed 0")
                    if j == 0 and n == 64
                    else None,
                )
                data["discovery"].append(
                    dict(
                        case=case,
                        n=n,
                        arm=arm,
                        count=value["count"],
                        requested=value["requested"],
                        low=low,
                        high=high,
                    )
                )
        ax.set(
            xlim=(-0.025, 1.025),
            ylim=(5.5, -0.5),
            xlabel="Generating-partition recovery fraction",
            title=f"N = {n} unlabelled training specimens",
        )
        ax.set_yticks(range(6), names)
        ax.grid(axis="x", color=".9", linewidth=0.6)
    fig.suptitle("Unsupervised recovery: selected K = 4 and held-out ARI ≥ 0.8", y=0.99)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.text(
        0.5,
        0.005,
        "Five datasets/law; marginal exact 95% intervals. "
        "Separate 40-rate simultaneous intervals are exported.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    save_figure(fig, directory, "discovery-recovery")
    plt.close(fig)
    write(directory / "figure-data.json", data)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    export(args.study_root, args.output_dir)
    if args.figure_dir:
        figures(args.output_dir, args.figure_dir)


if __name__ == "__main__":
    main()
