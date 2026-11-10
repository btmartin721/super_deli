"""Export audited cell summaries only; detailed scientific evidence remains local."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    root = parser.parse_args().study.resolve()
    summary = read(root / "summary.json")
    assert not summary["fixture"] and summary["observations"] == 100
    assert read(root / "main-audit.json")["passed"] and not (root / "STOP.json").exists()
    repo = Path(__file__).resolve().parents[2]
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=False)
    for name in ("summary", "groups", "primary-rates", "paired-comparisons"):
        (out / (name + ".json")).write_text(
            json.dumps(read(root / (name + ".json")), indent=2) + "\n"
        )
    (out / "independent-audit.json").write_text(json.dumps(summary["audits"], indent=2) + "\n")
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
    design["source_map_sha256"] = hashlib.sha256(
        json.dumps(frozen["source_sha256"], sort_keys=True).encode()
    ).hexdigest()
    (out / "study-design.json").write_text(json.dumps(design, indent=2) + "\n")
    rows = read(root / "primary-rates.json")
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    plotted = []
    colors = ("#286f9b", "#ca7e20")
    endpoints = ("global_rejection", "recovered", "joint_rejection_recovery")
    titles = ("Global Gaussian rejection", "K=4 and ARI >= 0.8", "Rejection AND recovery")
    for row, n in enumerate((64, 256)):
        for column, (endpoint, title) in enumerate(zip(endpoints, titles, strict=True)):
            axis = axes[row, column]
            for offset, arm in enumerate(("neural", "pca4")):
                cell = sorted(
                    [
                        r
                        for r in rows
                        if r["n"] == n and r["arm"] == arm and r["endpoint"] == endpoint
                    ],
                    key=lambda r: r["signal_multiplier"],
                )
                assert len(cell) == 3
                x = [r["signal_multiplier"] + (offset - 0.5) * 0.035 for r in cell]
                y = [r["count"] / r["requested"] for r in cell]
                low = [
                    v - r["simultaneous_failure_interval"][0] for r, v in zip(cell, y, strict=True)
                ]
                high = [
                    r["simultaneous_failure_interval"][1] - v for r, v in zip(cell, y, strict=True)
                ]
                axis.errorbar(
                    x,
                    y,
                    yerr=[low, high],
                    linestyle="none",
                    marker="o" if offset == 0 else "s",
                    color=colors[offset],
                    capsize=4,
                    label="Separate encoders" if offset == 0 else "Per-view PCA",
                )
                for r in cell:
                    plotted.append(
                        {
                            k: r[k]
                            for k in (
                                "cell",
                                "n",
                                "arm",
                                "endpoint",
                                "signal_multiplier",
                                "count",
                                "requested",
                                "unavailable",
                                "simultaneous_failure_interval",
                            )
                        }
                    )
            axis.set_title(f"N={n}: {title}")
            axis.set_xticks([0.5, 1, 1.5])
            axis.set_xlabel("Generating center multiplier")
            axis.set_ylim(-0.05, 1.05)
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel("Fraction of requested datasets")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Frozen paired power/recovery study: 10 independent datasets per point")
    figure.text(
        0.5,
        0.02,
        "Exact simultaneous 95% intervals across 36 rates; failure bounds included.\n"
        "Offsets separate arms. Gaussian departure is not a species or selected-K test.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.89, bottom=0.15, hspace=0.38, wspace=0.13)
    target = repo / "docs/figures/neural-power"
    target.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(target / ("power-recovery." + suffix), dpi=180)
    svg = target / "power-recovery.svg"
    svg.write_text("\n".join(x.rstrip() for x in svg.read_text().splitlines()) + "\n")
    (target / "figure-data.json").write_text(json.dumps(plotted, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                public_cells=len(rows), figure_points=len(plotted), private_records_published=False
            )
        )
    )


if __name__ == "__main__":
    main()
