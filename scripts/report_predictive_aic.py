"""Publish audited screening aggregates without raw observations or fitted models."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def describe(values):
    from super_deli.benchmarking.grouped_power_analysis import describe as summarize

    return summarize(values)


def mechanisms(rows):
    result = []
    for cell in sorted({r["cell"] for r in rows}):
        group = [r for r in rows if r["cell"] == cell]
        complete = [r for r in group if r["selection"]["status"] == "complete"]
        blocked = Counter()
        for r in complete:
            s = r["selection"]
            winner = next(c for c in s["candidates"] if c["k"] == s["ordinary_aic_k"])
            if s["ordinary_aic_k"] > 1 and not winner["passes"]:
                blocked[winner["reason"]] += 1
        pairs = [(r["performance"]["ordinary"], r["performance"]["screened"]) for r in complete]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        result.append(
            {
                "cell": cell,
                "requested": len(group),
                "available_pairs": len(pairs),
                "selection_changes": sum(a["k"] != b["k"] for a, b in pairs),
                "ordinary_winner_blocked": dict(blocked),
                "external_gain_screened_minus_ordinary": describe(
                    [
                        b["external_mean_logdensity_gain"] - a["external_mean_logdensity_gain"]
                        for a, b in pairs
                    ]
                ),
                "ari_screened_minus_ordinary": describe(
                    [
                        b["ari"] - a["ari"]
                        for a, b in pairs
                        if a["ari"] is not None and b["ari"] is not None
                    ]
                ),
            }
        )
    return result


def figure(summary, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw={"width_ratios": [1.2, 1]})
    names = {
        "independent_gaussian": "Independent",
        "correlated_gaussian": "Correlated",
        "zero_conditional_gaussian": "Zero conditional",
        "gaussian_scores": "Direct scores",
    }
    for ax, role in zip(axes, ("control", "alternative"), strict=True):
        cells = [r for r in summary["cells"] if r["cell"]["role"] == role]
        labels = []
        for y, row in enumerate(cells):
            cell = row["cell"]
            labels.append(
                f"{names[cell['law']]} / N={cell['n_units']}"
                if role == "control"
                else f"N={cell['n_units']} / signal {cell['signal']:g}"
            )
            for method, dy, color, marker in (
                ("ordinary", -0.12, "#D55E00", "o"),
                ("screened", 0.12, "#0072B2", "s"),
            ):
                rate = row["methods"][method]["rate"]
                if not rate["available"]:
                    continue
                x = rate["rate_among_available"]
                lo, hi = rate["bonferroni_familywise_95"]
                ax.errorbar(
                    x,
                    y + dy,
                    xerr=[[x - lo], [hi - x]],
                    fmt=marker,
                    color=color,
                    capsize=3,
                    label=method.title() + " AIC" if y == 0 else None,
                )
                ax.text(
                    1.03,
                    y + dy,
                    f"{rate['events']}/{rate['available']}",
                    va="center",
                    fontsize=8,
                    color=color,
                )
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.0)
        ax.grid(axis="x", alpha=0.2)
        ax.set_title("Gaussian controls" if role == "control" else "Four-group alternatives")
        ax.set_xlabel(
            "Fraction selecting K > 1" if role == "control" else "Fraction selecting K = 4"
        )
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Predictive screening reduces partitioning, with a recovery tradeoff\n"
        "95% simultaneous exact intervals across 28 prespecified rates",
        fontsize=12,
    )
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory / f"predictive-screen-rates.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    summary = read(args.audit / "summary.json")
    with gzip.open(args.audit / "points.jsonl.gz", "rt") as handle:
        rows = [json.loads(s) for s in handle]
    assert len(rows) == summary["observations"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary.json", summary), ("mechanisms.json", mechanisms(rows))):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "primary-rates.csv").open("w", newline="") as handle:
        fields = [
            "cell",
            "method",
            "event",
            "events",
            "available",
            "requested",
            "withheld",
            "rate_among_available",
            "mcse_among_available",
            "exact_interval_95",
            "bonferroni_familywise_95",
            "withheld_outcome_bounds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["primary_rates"]:
            writer.writerow({k: row[k] for k in fields})
    figure(summary, args.figures)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output_dir.iterdir())
        if p.is_file()
    }
    (args.output_dir / "checksums.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "observations": len(rows),
                "primary_rates": len(summary["primary_rates"]),
                "small_public_files": len(hashes),
            }
        )
    )


if __name__ == "__main__":
    main()
