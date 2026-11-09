"""Export only audited Gaussian rate aggregates and the fixed archived diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    root = parser.parse_args().study.resolve()
    repo = Path(__file__).resolve().parents[2]
    summary = read(root / "summary.json")
    assert not summary["fixture"] and summary["observations"] == 400
    assert read(root / "main-audit.json")["passed"] and not (root / "STOP.json").exists()
    target = Path(__file__).parent / "results"
    target.mkdir(exist_ok=False)
    for name in (
        "summary",
        "groups",
        "primary-rates",
        "paired-comparisons",
        "precision-diagnostic",
    ):
        (target / (name + ".json")).write_text(
            json.dumps(read(root / (name + ".json")), indent=2) + "\n"
        )
    (target / "independent-audit.json").write_text(json.dumps(summary["audits"], indent=2) + "\n")
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
            "plan_sha256",
        )
    }
    design["archived_pilot"] = {
        "study": "neural-calibration-20261107",
        "published_head": "54dd00c5710bb26a5d73115696a8b55b92820695",
        "selected_case": "main/gaussian_independent_0009",
        "role": "Archived precision diagnostic; excluded from fresh rates",
    }
    design["source_map_sha256"] = hashlib.sha256(
        json.dumps(frozen["source_sha256"], sort_keys=True).encode()
    ).hexdigest()
    (target / "study-design.json").write_text(json.dumps(design, indent=2) + "\n")
    primary = read(root / "primary-rates.json")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    plotted = []
    cases = ("gaussian_independent", "gaussian_correlated")
    titles = ("Independent views", "Correlated views")
    endpoints = ("ordinary_split", "global_rejection")
    headings = ("Ordinary BIC selected K > 1", "Global Gaussian test rejected")
    colors = ("#2b6f9b", "#ce7e24")
    for row, (case, title) in enumerate(zip(cases, titles, strict=True)):
        for column, (endpoint, heading) in enumerate(zip(endpoints, headings, strict=True)):
            axis = axes[row, column]
            cell = [
                next(
                    p
                    for p in primary
                    if (p["case"], p["arm"], p["endpoint"]) == (case, arm, endpoint)
                )
                for arm in ("neural", "pca4")
            ]
            # A plotting-only rule chosen before outcomes: show every interval;
            # ordinary selection has no nominal 5% guarantee.
            top = max(
                0.25 if column == 0 else 0.15,
                max(p["simultaneous_failure_interval"][1] for p in cell) * 1.18,
            )
            for i, value in enumerate(cell):
                fraction = value["count"] / value["requested"]
                low, high = value["simultaneous_failure_interval"]
                axis.bar(i, fraction, width=0.55, color=colors[i])
                axis.errorbar(
                    i,
                    fraction,
                    yerr=[[fraction - low], [high - fraction]],
                    color="black",
                    capsize=5,
                    lw=1,
                )
                label = f"{value['count']}/{value['requested']}"
                if value["unavailable"]:
                    label += f" ?{value['unavailable']}"
                axis.text(i, high + 0.025 * top, label, ha="center", fontsize=10)
                plotted.append(
                    dict(
                        case=case,
                        arm=value["arm"],
                        endpoint=endpoint,
                        count=value["count"],
                        requested=value["requested"],
                        unavailable=value["unavailable"],
                        interval=[low, high],
                    )
                )
            if column == 1:
                axis.axhline(0.05, color="#555555", lw=1, ls=":", label="Nominal alpha=0.05")
                axis.legend(frameon=False, fontsize=9, loc="upper right")
            axis.set_title(title + "\n" + heading)
            axis.set_xticks([0, 1], ["Separate encoders", "Per-view PCA"])
            axis.set_ylim(0, top)
            axis.set_ylabel("Fraction of requested datasets")
            axis.grid(axis="y", alpha=0.15)
            axis.set_axisbelow(True)
    figure.suptitle("Frozen Gaussian validation: N=64, 200 independent datasets per law", y=0.98)
    figure.text(
        0.5,
        0.025,
        "Whiskers: simultaneous 95% exact intervals across eight rates, including failure "
        "bounds. ? marks unavailable outcomes.\nArchived precision/replay cases are excluded. "
        "Gaussian-model departure is not species significance; panel scales differ.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.87, bottom=0.14, hspace=0.5, wspace=0.32)
    figures = repo / "docs/figures/neural-false-splits"
    figures.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(figures / ("gaussian-validation." + suffix), dpi=180)
    svg = figures / "gaussian-validation.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    (figures / "figure-data.json").write_text(json.dumps(plotted, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                primary_rates=len(primary),
                figure_cells=len(plotted),
                public_json_bytes=sum(p.stat().st_size for p in target.iterdir()),
            )
        )
    )


if __name__ == "__main__":
    main()
