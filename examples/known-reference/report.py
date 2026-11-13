"""Export small parent-weighted aggregates and figures without fitting models."""

import argparse
import json
from pathlib import Path

CASES = ("linear_weak", "linear_standard", "combined", "confounded_size")
LABELS = ("Weak linear", "Standard linear", "Allometry + saturation", "Confounded size")
POLICIES = ("unscreened", "probability", "distance", "local", "density")
COLORS = ("#888888", "#D55E00", "#0072B2", "#009E73", "#CC79A7")


def stat(values):
    import numpy as np

    a = [float(v) for v in values if v is not None]
    return dict(
        n=len(a),
        mean=float(np.mean(a)) if a else None,
        minimum=min(a) if a else None,
        maximum=max(a) if a else None,
    )


def aggregate_points(observations, groups):
    points = []
    overlap = []
    for case in CASES:
        for n in (64, 256):
            parents = [o for o in observations if (o["task"]["case"], o["task"]["n"]) == (case, n)]
            for arm in ("logistic", "residual_0"):
                for policy in POLICIES:
                    coverage = []
                    accuracy = []
                    common_cov = []
                    worst = []
                    matching = []
                    unfamiliar = []
                    folds = []
                    common_folds = []
                    for parent in parents:
                        rows = [
                            r["metrics"][policy]
                            for r in parent["fits"]
                            if r["arm"] == arm
                            and r["status"] == "available"
                            and r["policy_status"][policy] == "available"
                        ]
                        common = [m for m in rows if m["iid"]["accepted_accuracy"] is not None]

                        def mean(a):
                            return sum(a) / len(a) if a else None

                        folds.append(len(rows))
                        common_folds.append(len(common))
                        coverage.append(mean([m["iid"]["accepted_fraction"] for m in rows]))
                        common_cov.append(mean([m["iid"]["accepted_fraction"] for m in common]))
                        accuracy.append(mean([m["iid"]["accepted_accuracy"] for m in common]))
                        worst.append(
                            mean(
                                [
                                    max(
                                        m[c]["accepted_fraction"]
                                        for c in ("heldout", "near", "far")
                                    )
                                    for m in rows
                                ]
                            )
                        )
                        matching.append(
                            mean(
                                [
                                    m["iid"]["per_class"][0]["accepted"]
                                    / m["iid"]["per_class"][0]["n"]
                                    for m in rows
                                ]
                            )
                        )
                        unfamiliar.append(mean([m["overlap"]["accepted_fraction"] for m in rows]))
                    identity = dict(
                        case=case, n=n, known_training_n=3 * n // 4, arm=arm, policy=policy
                    )
                    group = next(
                        g
                        for g in groups
                        if (g["case"], g["n"], g["arm"], g["policy"]) == (case, n, arm, policy)
                    )
                    points.append(
                        dict(
                            **identity,
                            requested_parents=len(parents),
                            available_folds=stat(folds),
                            accuracy_common_folds=stat(common_folds),
                            coverage=stat(coverage),
                            accuracy_common_coverage=stat(common_cov),
                            accuracy=stat(accuracy),
                            worst_unfamiliar=stat(worst),
                            joint_parent_success=group["three_of_four_joint_target"],
                        )
                    )
                    overlap.append(
                        dict(
                            **identity,
                            matching_known_coverage=stat(matching),
                            unfamiliar_overlap_acceptance=stat(unfamiliar),
                        )
                    )
    return points, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.study_root
    repo = args.repository

    def read(name):
        return json.loads((root / name).read_text())

    out = repo / "examples/known-reference/results"
    figures = repo / "docs/figures/known-reference"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    def write(path, value):
        path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        assert path.stat().st_size < 1024**2, path

    for name in (
        "summary",
        "primary-rates",
        "paired-contrasts",
        "stopping-assessment",
        "study-design",
    ):
        write(out / (name + ".json"), read(name + ".json"))
    write(out / "independent-audit.json", read("report-qa.json"))
    for category in ("groups", "fold-groups", "classwise"):
        rows = read(category + ".json")
        for case in CASES:
            write(out / (category + "-" + case + ".json"), [r for r in rows if r["case"] == case])
    points, overlap = aggregate_points(read("observations.json"), read("groups.json"))
    write(out / "overlap-summary.json", overlap)
    write(
        figures / "figure-data.json",
        dict(
            points=points,
            range_definition="Observed range of independent parent means, not confidence intervals",
            unit_definition=(
                "Each parent averages its available heldout folds; folds are repeated measurements"
            ),
            accuracy_definition=(
                "Accuracy and its coverage use the same nonempty "
                "accepted-known folds within each parent"
            ),
            unfamiliar_definition="Parent mean of per-fold maximum heldout/near/far acceptance",
            target_note=(
                "Lines show individual target components; parent success "
                "requires three of four folds to meet all criteria"
            ),
            training_note=(
                "Original four-group panel64/256 becomes known "
                "training48/192; no unfamiliar training examples"
            ),
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 9,
            "svg.hashsalt": "super-deli-known-reference",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for name, second in (("coverage-accuracy", False), ("coverage-unfamiliar", True)):
        fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
        for row, case in enumerate(CASES):
            for col, n in enumerate((64, 256)):
                ax = axes[row, col]
                ax.set_xlim(-3, 103)
                ax.set_ylim(-3, 103)
                ax.grid(alpha=0.15)
                ax.axvline(80, ls=":", lw=1, c=".4")
                ax.axhline(10 if second else 90, ls=":", lw=1, c=".4")
                for i, policy in enumerate(POLICIES):
                    for arm, marker in (("logistic", "o"), ("residual_0", "s")):
                        p = next(
                            v
                            for v in points
                            if (v["case"], v["n"], v["arm"], v["policy"]) == (case, n, arm, policy)
                        )
                        x = p["coverage"] if second else p["accuracy_common_coverage"]
                        y = p["worst_unfamiliar"] if second else p["accuracy"]
                        if x["mean"] is None or y["mean"] is None:
                            continue
                        ax.errorbar(
                            100 * x["mean"],
                            100 * y["mean"],
                            xerr=[
                                [100 * (x["mean"] - x["minimum"])],
                                [100 * (x["maximum"] - x["mean"])],
                            ],
                            yerr=[
                                [100 * (y["mean"] - y["minimum"])],
                                [100 * (y["maximum"] - y["mean"])],
                            ],
                            fmt=marker,
                            ms=5,
                            color=COLORS[i],
                            mfc="white" if arm == "logistic" else COLORS[i],
                            capsize=1.5,
                            elinewidth=0.6,
                            alpha=0.9,
                            zorder=3,
                        )
                target = [
                    next(
                        v
                        for v in points
                        if (v["case"], v["n"], v["arm"], v["policy"]) == (case, n, a, "density")
                    )["joint_parent_success"]["count"]
                    for a in ("logistic", "residual_0")
                ]
                ax.set_title(f"{LABELS[row]} | known train N={3 * n // 4}", fontsize=10)
                ax.text(
                    0.02,
                    0.03,
                    f"New-policy joint parents: LR {target[0]}/10; residual {target[1]}/10",
                    transform=ax.transAxes,
                    fontsize=7.5,
                )
                if row == 3:
                    ax.set_xlabel("Known specimens accepted (%)")
                if col == 0:
                    ax.set_ylabel(
                        "Worst unfamiliar\nacceptance per fold (%)"
                        if second
                        else "Accuracy among accepted known (%)"
                    )
        handles = [
            Line2D([0], [0], marker="o", color=c, lw=0, label=label)
            for c, label in zip(
                COLORS,
                (
                    "Unscreened",
                    "Probability",
                    "Pooled distance",
                    "Fifth neighbor",
                    "Class-specific",
                ),
                strict=True,
            )
        ]
        handles.extend(
            [
                Line2D([0], [0], marker="o", mfc="white", color="black", lw=0, label="Logistic"),
                Line2D([0], [0], marker="s", color="black", lw=0, label="Residual seed 0"),
            ]
        )
        fig.legend(
            handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.025)
        )
        fig.suptitle(
            "Known-only reference support with entire generating groups held out",
            fontsize=12,
            y=0.995,
        )
        fig.text(
            0.5,
            0.012,
            (
                "Points: parent means; bars: observed parent ranges, not "
                "confidence intervals. Targets evaluated within each fold."
            ),
            ha="center",
            fontsize=7.3,
        )
        fig.tight_layout(rect=(0, 0.095, 1, 0.975))
        for ext in ("png", "svg", "pdf"):
            metadata = (
                {"CreationDate": None, "ModDate": None}
                if ext == "pdf"
                else {"Date": None}
                if ext == "svg"
                else None
            )
            fig.savefig(
                figures / (name + "." + ext), dpi=180, bbox_inches="tight", metadata=metadata
            )
        svg = figures / (name + ".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)
    public = [*out.iterdir(), *figures.iterdir()]
    assert all(p.stat().st_size < 1024**2 for p in public)
    print(
        json.dumps(
            dict(
                public_files=len(public),
                public_bytes=sum(p.stat().st_size for p in public),
                point_summaries=len(points),
                overlap_summaries=len(overlap),
                individual_records=0,
            )
        )
    )


if __name__ == "__main__":
    main()
