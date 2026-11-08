"""Publish audited cell aggregates and a fixed pilot figure; keep all fits local."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    root = parser.parse_args().study.resolve()
    repo = Path(__file__).resolve().parents[2]
    assert read(root / "main-audit.json")["passed"] and not (root / "STOP.json").exists()
    target = Path(__file__).parent / "results"
    target.mkdir(exist_ok=False)
    for name in ("summary", "groups", "primary-rates", "paired-comparisons"):
        (target / (name + ".json")).write_text(
            json.dumps(read(root / (name + ".json")), indent=2) + "\n"
        )
    audits = read(root / "summary.json")["audits"]
    assert all("records" not in value for value in audits.values())
    (target / "independent-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    frozen = read(root / "frozen.json")
    design = {
        k: frozen[k]
        for k in (
            "configuration",
            "settings",
            "arms",
            "training",
            "counts",
            "python",
            "platform",
            "package_version",
            "software",
        )
    }
    (target / "study-design.json").write_text(json.dumps(design, indent=2) + "\n")
    groups = read(root / "groups.json")
    cases = ("gaussian_independent", "gaussian_correlated", "four_complementary", "heavy_tail")
    names = (
        "Gaussian: independent views",
        "Gaussian: correlated views",
        "Complementary four groups",
        "Heavy-tailed stress",
    )
    colors = ("#2b6f9b", "#ce7e24")
    endpoints = ("ordinary_split", "global_rejection")
    labels = ("BIC selected K > 1", "Global Gaussian test rejected")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    plotted = []
    for case, title, axis in zip(cases, names, axes.flat, strict=True):
        for j, arm in enumerate(("neural", "pca4")):
            group = next(r for r in groups if r["case"] == case and r["arm"] == arm)
            for index, endpoint in enumerate(endpoints):
                value = group[endpoint]
                center = j + (-0.18 if index == 0 else 0.18)
                fraction = value["count"] / value["requested"]
                low, high = value["simultaneous_failure_interval"]
                axis.bar(center, fraction, width=0.3, color=colors[index], label=labels[index])
                axis.errorbar(
                    center,
                    fraction,
                    yerr=[[fraction - low], [high - fraction]],
                    color="black",
                    capsize=4,
                    lw=1,
                )
                text = f"{value['count']}/{value['requested']}"
                if value["unavailable"]:
                    text += f" ?{value['unavailable']}"
                axis.text(center, high + 0.035, text, ha="center", fontsize=10)
                plotted.append(
                    dict(
                        case=case,
                        arm=arm,
                        endpoint=endpoint,
                        count=value["count"],
                        requested=value["requested"],
                        unavailable=value["unavailable"],
                        interval=[low, high],
                    )
                )
        if case.startswith("gaussian_"):
            axis.axhline(0.05, color="#777777", ls=":", lw=1)
        axis.set_title(title)
        axis.set_xticks([0, 1], ["Separate encoders", "Per-view PCA"])
        axis.set_ylim(0, 1.16)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_ylabel("Fraction of requested datasets")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    handles, legend = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles[:2],
        legend[:2],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    figure.suptitle(
        "Full-pipeline calibration pilot: N=64, ten independent datasets per law", y=0.99
    )
    figure.text(
        0.5,
        0.025,
        "Counts / requested datasets; ? marks unavailable outcomes. Whiskers: simultaneous 95% "
        "intervals across 16 rates, including failure bounds.\nDotted line: nominal alpha=0.05 "
        "in Gaussian controls. Heavy-tail rejection is model departure; no species guarantee.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.85, bottom=0.14, hspace=0.4, wspace=0.22)
    figures = repo / "docs/figures/neural-calibration"
    figures.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(figures / ("calibration-pilot." + suffix), dpi=180)
    svg = figures / "calibration-pilot.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    (figures / "figure-data.json").write_text(json.dumps(plotted, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                groups=len(groups),
                primary_rates=16,
                figure_cells=len(plotted),
                public_json_bytes=sum(p.stat().st_size for p in target.iterdir()),
            )
        )
    )


if __name__ == "__main__":
    main()
