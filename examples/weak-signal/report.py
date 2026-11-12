"""Export audited cell summaries and figures; individual evidence stays local."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    root = parser.parse_args().study.resolve()
    assert read(root / "report-qa.json")["passed"] and not (root / "STOP.json").exists()
    summary, groups = read(root / "summary.json"), read(root / "groups.json")
    assert not summary["fixture"] and summary["observations"] == 80
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=False)
    for name in ("summary", "groups"):
        (output / (name + ".json")).write_text(
            json.dumps(read(root / (name + ".json")), indent=2) + "\n"
        )
    frozen = read(root / "frozen.json")
    design = {
        k: frozen[k]
        for k in (
            "configuration",
            "settings",
            "arms",
            "counts",
            "software",
            "package_version",
            "plan_sha256",
        )
    }
    design["source_map_sha256"] = hashlib.sha256(
        json.dumps(frozen["source_sha256"], sort_keys=True).encode()
    ).hexdigest()
    design["source_head"] = "cf6082d383f15a5b058c49585f5b4bbbfa4e0aa6"
    (output / "study-design.json").write_text(json.dumps(design, indent=2) + "\n")
    qa = read(root / "report-qa.json")
    qa["audits"] = {
        p: {k: v for k, v in read(root / (p + "-audit.json")).items() if k != "records"}
        for p in ("preflight", "main")
    }
    (output / "independent-audit.json").write_text(json.dumps(qa, indent=2) + "\n")
    weak = {
        g["n"]: g
        for g in groups
        if g["case"] == "four_complementary" and g["signal_multiplier"] == 0.5
    }
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.6, 1]})
    plotted = []
    names = [
        "Known raw\nparameters",
        "Known PCA\nparameters",
        "PCA\nselected K",
        "Neural\nselected K",
        "PCA\nstored K=4",
        "Neural\nstored K=4",
        "PCA\ntruth start",
        "Neural\ntruth start",
    ]
    for offset, n in enumerate((64, 256)):
        group = weak[n]
        values = [group["references"][s]["ari"] for s in ("raw", "pca4")]
        values += [
            group["heads"][a][field]
            for field in ("native_ari", "stored_k4_ari", "truth_ari")
            for a in ("pca4", "neural")
        ]
        x = np.arange(len(names)) + (offset - 0.5) * 0.36
        y = np.array([v["median"] for v in values])
        err = np.array(
            [
                [v["median"] - v["minimum"] for v in values],
                [v["maximum"] - v["median"] for v in values],
            ]
        )
        axes[0].bar(x, y, width=0.34, color=("#286f9b", "#ca7e20")[offset], label=f"N={n}")
        axes[0].errorbar(x, y, yerr=err, fmt="none", ecolor="black", capsize=2, linewidth=0.8)
        plotted.extend(
            dict(panel="ari", n=n, method=name, **value)
            for name, value in zip(names, values, strict=True)
        )
    axes[0].axhline(0.8, color="#777777", linestyle="--", linewidth=1)
    axes[0].set(
        xticks=np.arange(len(names)),
        xticklabels=names,
        ylim=(-0.08, 1.02),
        ylabel="Adjusted Rand agreement",
        title="Weak signal: median agreement and observed range",
    )
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].legend(frameon=False, loc="upper left")
    labels, gains, penalties = [], [], []
    for n in (64, 256):
        for arm in ("pca4", "neural"):
            h = weak[n]["heads"][arm]
            labels.append(f"N={n}\n{'PCA' if arm == 'pca4' else 'Neural'}")
            gains.append(h["k4_twice_likelihood_gain"]["median"])
            penalties.append(h["k4_bic_penalty"]["median"])
            plotted.extend(
                [
                    dict(panel="cost", n=n, arm=arm, measure="twice_ll_gain", value=gains[-1]),
                    dict(panel="cost", n=n, arm=arm, measure="bic_penalty", value=penalties[-1]),
                ]
            )
    x = np.arange(4)
    axes[1].bar(x - 0.18, gains, width=0.34, color="#3b8c6e", label="Twice likelihood gain")
    axes[1].bar(x + 0.18, penalties, width=0.34, color="#9370a3", label="Additional BIC penalty")
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Within-space contrast against selected K",
        title="K=4 fit gain versus complexity cost: cell medians",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("Archived weak-signal diagnosis: ten datasets per cell")
    figure.text(
        0.5,
        0.025,
        (
            "Whiskers show observed ranges, not confidence intervals. "
            "Dashed line is the benchmark ARI criterion.\n"
            "Known-parameter and truth-start results are diagnostic; "
            "they are not selectable species partitions or ARI ceilings."
        ),
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.85, bottom=0.2, left=0.055, right=0.99, wspace=0.28)
    target = Path(__file__).resolve().parents[2] / "docs/figures/weak-signal"
    target.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(target / ("diagnosis." + suffix), dpi=180)
    svg = target / "diagnosis.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    (target / "figure-data.json").write_text(json.dumps(plotted, indent=2) + "\n")
    print(
        json.dumps(
            dict(public_groups=8, figure_points=len(plotted), individual_records_published=False)
        )
    )


if __name__ == "__main__":
    main()
