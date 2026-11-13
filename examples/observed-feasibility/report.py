"""Small aggregate exports and audited scientific figures; no model fitting."""

import argparse
import json
from pathlib import Path

CASES = ("linear_weak", "linear_standard", "combined", "confounded_size")
LABELS = ("Weak linear", "Standard linear", "Allometry + saturation", "Confounded size")
POLICIES = ("unscreened", "probability", "distance", "local", "diagnostic")
COLORS = ("#888888", "#D55E00", "#0072B2", "#009E73", "#CC79A7")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.study_root
    repo = args.repository

    def read(name):
        return json.loads((root / name).read_text())

    out = repo / "examples/observed-feasibility/results"
    figures = repo / "docs/figures/observed-feasibility"
    out.mkdir(exist_ok=True, parents=True)
    figures.mkdir(exist_ok=True, parents=True)

    def write(name, value):
        p = out / name
        p.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        assert p.stat().st_size < 1024**2, name

    for name in (
        "summary.json",
        "primary-rates.json",
        "paired-contrasts.json",
        "qualification.json",
        "stopping-assessment.json",
        "study-design.json",
    ):
        write(name, read(name))
    write("independent-audit.json", read("report-qa.json"))
    for category in ("groups", "classwise"):
        rows = read(category + ".json")
        for case in CASES:
            write(category + "-" + case + ".json", [v for v in rows if v["case"] == case])
    observations = read("observations.json")
    points = []
    curve_data = []

    def stat(values):
        a = [float(v) for v in values if v is not None]
        return dict(
            n=len(a),
            mean=sum(a) / len(a) if a else None,
            minimum=min(a) if a else None,
            maximum=max(a) if a else None,
        )

    for case in CASES:
        for n in (64, 256):
            obs = [o for o in observations if (o["task"]["case"], o["task"]["n"]) == (case, n)]
            for arm in ("logistic", "residual_0"):
                rows = [next(v for v in o["assessments"] if v["arm"] == arm) for o in obs]
                for policy in POLICIES:
                    good = [
                        r["metrics"][policy]
                        for r in rows
                        if r["status"] == "available" and r["metrics"][policy] is not None
                    ]
                    common = [m for m in good if m["iid"]["accepted_accuracy"] is not None]
                    points.append(
                        dict(
                            case=case,
                            n=n,
                            arm=arm,
                            policy=policy,
                            requested=len(rows),
                            available=len(good),
                            coverage=stat([m["iid"]["accepted_fraction"] for m in good]),
                            accuracy_common_coverage=stat(
                                [m["iid"]["accepted_fraction"] for m in common]
                            ),
                            accuracy=stat([m["iid"]["accepted_accuracy"] for m in common]),
                            worst_unfamiliar=stat(
                                [
                                    max(
                                        m["near"]["accepted_fraction"],
                                        m["far"]["accepted_fraction"],
                                    )
                                    for m in good
                                ]
                            ),
                        )
                    )
            # Detector scores/coverage do not depend on the frozen reference head.
            rows = [next(v for v in o["assessments"] if v["arm"] == "residual_0") for o in obs]
            for i in range(101):
                known = [r["evaluation_curves"]["iid"] for r in rows]
                near = [r["evaluation_curves"]["near"] for r in rows]
                far = [r["evaluation_curves"]["far"] for r in rows]
                curve_data.append(
                    dict(
                        case=case,
                        n=n,
                        threshold=i / 100,
                        coverage=stat([m["accepted"][i] / m["n"] for m in known]),
                        worst_unfamiliar=stat(
                            [
                                max(a["accepted"][i] / a["n"], b["accepted"][i] / b["n"])
                                for a, b in zip(near, far, strict=True)
                            ]
                        ),
                    )
                )
    write("curve-points.json", curve_data)
    payload = dict(
        points=points,
        curves=curve_data,
        range_definition="Observed dataset minimum/maximum, not confidence intervals",
        original_n_definition=(
            "N identifies original reference-head training; "
            "detector training is 2048 specimens in every scientific cell"
        ),
        information=(
            "Specified unfamiliar examples supplied in independent detector training/calibration; "
            "no unknown-generalization guarantee"
        ),
        target_note=(
            "Plotted target lines show components only; joint success is evaluated per dataset, "
            "not from plotted means"
        ),
    )
    (figures / "figure-data.json").write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 9,
            "svg.hashsalt": "super-deli-observed-feasibility",
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
                if second:
                    c = [v for v in curve_data if (v["case"], v["n"]) == (case, n)]
                    ax.plot(
                        [100 * v["coverage"]["mean"] for v in c],
                        [100 * v["worst_unfamiliar"]["mean"] for v in c],
                        color=".75",
                        lw=1,
                        zorder=1,
                    )
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
                qual = [
                    next(
                        v
                        for v in points
                        if (v["case"], v["n"], v["arm"], v["policy"]) == (case, n, a, "diagnostic")
                    )["available"]
                    for a in ("logistic", "residual_0")
                ]
                ax.set_title(f"{LABELS[row]} | reference N={n}", fontsize=10)
                ax.text(
                    0.02,
                    0.03,
                    f"Diagnostic point: LR {qual[0]}/10; residual {qual[1]}/10",
                    transform=ax.transAxes,
                    fontsize=7.5,
                )
                if row == 3:
                    ax.set_xlabel("Known specimens accepted (%)")
                if col == 0:
                    ax.set_ylabel(
                        "Worse of near/far acceptance (%)"
                        if second
                        else "Accuracy among accepted known (%)"
                    )
        handles = [
            Line2D([0], [0], marker="o", color=color, lw=0, label=label)
            for color, label in zip(
                COLORS,
                ("Unscreened", "Probability", "Distance", "Class-local", "Exposed diagnostic"),
                strict=True,
            )
        ]
        handles += [
            Line2D([0], [0], marker="o", mfc="white", color="black", lw=0, label="Logistic"),
            Line2D([0], [0], marker="s", color="black", lw=0, label="Residual seed 0"),
        ]
        if second:
            handles.append(
                Line2D([0], [0], color=".75", lw=1, label="All 101 fixed score thresholds")
            )
        fig.legend(
            handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.025)
        )
        fig.suptitle(
            "Observed-trait feasibility with specified unfamiliar exposure", fontsize=13, y=0.995
        )
        fig.text(
            0.5,
            0.012,
            "Points: dataset means; ranges: observed minima/maxima, not confidence intervals. "
            "Targets evaluated per dataset.",
            ha="center",
            fontsize=7.5,
        )
        fig.tight_layout(rect=(0, 0.095, 1, 0.975))
        for extension in ("png", "svg", "pdf"):
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            fig.savefig(
                figures / (name + "." + extension), dpi=180, bbox_inches="tight", metadata=metadata
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
                individual_records=0,
                point_summaries=len(points),
                curve_summaries=len(curve_data),
            )
        )
    )


if __name__ == "__main__":
    main()
