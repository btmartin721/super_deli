"""Export audited aggregates and figures; individual scientific records stay local."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

CASES = ("linear_weak", "linear_standard", "combined", "confounded_size")
TITLES = {
    "linear_weak": "Weak linear",
    "linear_standard": "Standard linear",
    "combined": "Allometry + saturation",
    "confounded_size": "Confounded size",
}
COLORS = {"distance": "#7A7A7A", "probability": "#0072B2", "local": "#009E73"}
LABELS = {"distance": "Pooled distance", "probability": "Probability", "local": "Class-local"}
MODELS = (("logistic", "Logistic", "o"), ("residual_0", "Residual seed 0", "^"))


def read(root, name):
    return json.loads((root / name).read_text())


def write(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) >= 1024**2:
        text = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    path.write_text(text)


def panels(title):
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=15, y=0.985)
    for row, case in enumerate(CASES):
        for col, n in enumerate((64, 256)):
            ax = axes[row, col]
            ax.set_title(f"{TITLES[case]}, N={n}", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.22)
    return fig, axes


def save(fig, directory, name, legend=None):
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    if legend:
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.005),
        )
    for suffix in ("png", "svg", "pdf"):
        path = directory / (name + "." + suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def coverage_error(root, directory):
    observations = read(root, "observations.json")
    points = []
    for case in CASES:
        for n in (64, 256):
            selected = [
                r for r in observations if r["task"]["case"] == case and r["task"]["n"] == n
            ]
            for arm, _label, _marker in MODELS:
                rows = [next(v for v in r["fits"] if v["arm"] == arm) for r in selected]
                for policy in COLORS:
                    values = [
                        r["metrics"][policy]["iid"]
                        for r in rows
                        if r["status"] == "available"
                        and (policy != "local" or r.get("local_status") == "available")
                    ]
                    pairs = [v for v in values if v["accepted_accuracy"] is not None]
                    item = dict(
                        case=case,
                        n=n,
                        arm=arm,
                        policy=policy,
                        requested=len(rows),
                        available=len(values),
                        defined_pairs=len(pairs),
                        zero_acceptance=sum(v["accepted"] == 0 for v in values),
                        overall_mean_coverage=float(
                            np.mean([v["accepted_fraction"] for v in values])
                        )
                        if values
                        else None,
                    )
                    if pairs:
                        x = np.array([v["accepted_fraction"] for v in pairs])
                        y = 1 - np.array([v["accepted_accuracy"] for v in pairs])
                        item.update(
                            x_mean=float(x.mean()),
                            y_mean=float(y.mean()),
                            x_range=[float(x.min()), float(x.max())],
                            y_range=[float(y.min()), float(y.max())],
                        )
                    points.append(item)
    ymax = max(0.1, max((p["y_range"][1] for p in points if p["defined_pairs"]), default=0.1))
    ymax = min(1, np.ceil((ymax + 0.005) * 20) / 20)
    fig, axes = panels("Reference support: known coverage and accepted-assignment error")
    for item in points:
        if not item["defined_pairs"]:
            continue
        ax = axes[CASES.index(item["case"]), int(item["n"] == 256)]
        x, y = item["x_mean"], item["y_mean"]
        ax.errorbar(
            x,
            y,
            xerr=[[max(0, x - item["x_range"][0])], [max(0, item["x_range"][1] - x)]],
            yerr=[[max(0, y - item["y_range"][0])], [max(0, item["y_range"][1] - y)]],
            color=COLORS[item["policy"]],
            marker="o" if item["arm"] == "logistic" else "^",
            capsize=2,
            markersize=6,
            linewidth=1,
            alpha=0.85,
        )
    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.axvline(0.8, color="black", ls=":", lw=0.8)
        ax.axhline(0.1, color="black", ls=":", lw=0.8)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes[-1]:
        ax.set_xlabel("Known specimens accepted")
    for ax in axes[:, 0]:
        ax.set_ylabel("Error among accepted known specimens")
    legend = [Line2D([], [], color=COLORS[p], lw=2, label=LABELS[p]) for p in COLORS]
    legend += [
        Line2D([], [], color="black", marker=m, ls="", label=label) for _, label, m in MODELS
    ]
    legend += [Line2D([], [], color="black", lw=1, label="Dataset min–max ranges")]
    save(fig, directory, "coverage-error", legend)
    return points


def unfamiliar(root, directory):
    groups = read(root, "groups.json")
    points = []
    fig, axes = panels("Unfamiliar acceptance remains a separate operating requirement")
    width = 0.12
    for row, case in enumerate(CASES):
        for col, n in enumerate((64, 256)):
            ax = axes[row, col]
            j = 0
            for arm, _, _ in MODELS:
                for policy in COLORS:
                    group = next(
                        g
                        for g in groups
                        if (g["case"], g["n"], g["arm"], g["policy"]) == (case, n, arm, policy)
                    )
                    for x, cohort in enumerate(("near", "far")):
                        value = group["cohorts"].get(cohort, {}).get("accepted_fraction")
                        item = dict(
                            case=case,
                            n=n,
                            arm=arm,
                            policy=policy,
                            cohort=cohort,
                            requested=group["requested"],
                            available=0 if value is None else value["available"],
                        )
                        if value is not None and value["mean"] is not None:
                            mean = value["mean"]
                            item.update(mean=mean, range=[value["minimum"], value["maximum"]])
                            ax.bar(
                                x + (j - 2.5) * width,
                                mean,
                                width,
                                color=COLORS[policy],
                                hatch="//" if arm != "logistic" else "",
                                edgecolor="black",
                                linewidth=0.35,
                            )
                            ax.errorbar(
                                x + (j - 2.5) * width,
                                mean,
                                yerr=[
                                    [max(0, mean - value["minimum"])],
                                    [max(0, value["maximum"] - mean)],
                                ],
                                color="black",
                                lw=0.7,
                                capsize=1.5,
                            )
                        points.append(item)
                    j += 1
            ax.axhline(0.1, color="black", ls=":", lw=0.8)
            ax.set_ylim(0, 1)
            ax.set_xlim(-0.5, 1.5)
            ax.set_xticks([0, 1], ["Nearby unfamiliar", "Distant unfamiliar"])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes[:, 0]:
        ax.set_ylabel("Unfamiliar specimens accepted")
    legend = [Patch(facecolor=COLORS[p], label=LABELS[p]) for p in COLORS]
    legend += [
        Patch(facecolor="white", edgecolor="black", label="Logistic"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Residual seed 0"),
        Line2D([], [], color="black", label="Dataset min–max ranges"),
    ]
    save(fig, directory, "unfamiliar-acceptance", legend)
    return points


def known_law(root, directory):
    groups = read(root, "oracle-groups.json")
    points = []
    fig, axes = panels(
        "Known-generating-law diagnostic: extra latent and unfamiliar-law information"
    )
    for row, case in enumerate(CASES):
        for col, n in enumerate((64, 256)):
            selected = sorted(
                [g for g in groups if g["case"] == case and g["n"] == n],
                key=lambda v: v["nominal_withholding"],
            )
            ax = axes[row, col]
            for cohort, color, style in (("near", "#D55E00", "-"), ("far", "#0072B2", "--")):
                x = []
                y = []
                for g in selected:
                    coverage = g["cohorts"]["iid"]["accepted_fraction"]
                    acceptance = g["cohorts"][cohort]["accepted_fraction"]
                    x.append(coverage["mean"])
                    y.append(acceptance["mean"])
                    points.append(
                        dict(
                            case=case,
                            n=n,
                            cohort=cohort,
                            nominal_withholding=g["nominal_withholding"],
                            x_mean=coverage["mean"],
                            y_mean=acceptance["mean"],
                            available=min(coverage["available"], acceptance["available"]),
                        )
                    )
                ax.plot(x, y, style, color=color, marker=".", markersize=4, lw=1.2)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axvline(0.8, color="black", ls=":", lw=0.8)
            ax.axhline(0.1, color="black", ls=":", lw=0.8)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes[-1]:
        ax.set_xlabel("Known specimens accepted by diagnostic rule")
    for ax in axes[:, 0]:
        ax.set_ylabel("Unfamiliar specimens accepted")
    save(
        fig,
        directory,
        "known-law-diagnostic",
        [
            Line2D([], [], color="#D55E00", label="Nearby unfamiliar"),
            Line2D([], [], color="#0072B2", ls="--", label="Distant unfamiliar"),
        ],
    )
    return points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    root, repo = args.study_root, args.repo_root
    summary = read(root, "summary.json")
    assert summary["completed"] and not summary["default_changes"] and not summary["pvalue_changes"]
    assert (
        read(root, "main-audit.json")["passed"] and not read(root, "study-design.json")["fixture"]
    )
    out = repo / "examples/local-reference-support/results"
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "summary.json",
        "groups.json",
        "classwise.json",
        "primary-rates.json",
        "paired-contrasts.json",
        "oracle-groups.json",
        "study-design.json",
    ):
        write(out / name, read(root, name))
    # Archive comparisons are descriptive; all seeds remain retained locally.
    for name in ("archival-groups.json", "archival-classwise.json"):
        write(out / name, [g for g in read(root, name) if g["arm"] in ("logistic", "residual_0")])
    write(out / "archival-oracle-groups.json", read(root, "archival-oracle-groups.json"))
    audits = {}
    for phase in ("preflight", "main"):
        value = read(root, phase + "-audit.json")
        audits[phase] = dict(
            passed=value["passed"],
            neural_fits=value["neural_fits"],
            verified_optimizer_updates=value["verified_optimizer_updates"],
            policy_evaluations=sum(r["policy_evaluations"] for r in value["records"]),
            oracle_evaluations=sum(r["oracle_evaluations"] for r in value["records"]),
        )
    write(out / "independent-audit.json", audits)
    directory = repo / "docs/figures/local-reference-support"
    directory.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.hashsalt": "super-deli-local-reference-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    data = dict(
        scope="Fresh dataset aggregates only; no individual specimens",
        ranges="Observed dataset minima–maxima, not confidence intervals",
        coverage_error=coverage_error(root, directory),
        unfamiliar_acceptance=unfamiliar(root, directory),
        known_law=known_law(root, directory),
    )
    write(directory / "figure-data.json", data)
    paths = [p for p in out.iterdir() if p.is_file()] + [
        p for p in directory.iterdir() if p.is_file()
    ]
    assert all(p.stat().st_size < 1024**2 for p in paths), [
        (p.name, p.stat().st_size) for p in paths
    ]
    print(
        json.dumps(
            dict(
                passed=True,
                aggregate_files=len(list(out.glob("*.json"))),
                figure_points={k: len(v) for k, v in data.items() if isinstance(v, list)},
                public_bytes=sum(p.stat().st_size for p in paths),
            )
        )
    )


if __name__ == "__main__":
    main()
