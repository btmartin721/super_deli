"""Compact audited fixed-family comparison reports; detailed evidence stays local."""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path


def describe(values):
    from super_deli.benchmarking.grouped_power_analysis import describe as summarize

    result = summarize(values)
    result.pop("mcse_mean", None)
    result["scope"] = "Descriptive; folds are not independent datasets"
    return result


def mechanisms(rows):
    result = []
    for cell in dict.fromkeys(r["cell"] for r in rows):
        group = [r for r in rows if r["cell"] == cell]
        for family in ("full", "tied"):
            reasons = Counter()
            gains = []
            for row in group:
                selection = row["arms"][family]["selection"]
                if selection["status"] != "complete":
                    reasons["withheld"] += 1
                    continue
                c = next(c for c in selection["candidates"] if c["k"] == 4)
                if c["full_status"] != "eligible":
                    reason = "full_ineligible"
                elif any(s != "eligible" for s in c["fold_statuses"]):
                    reason = "fold_ineligible"
                elif not c["passes"]:
                    reason = "nonpositive_gain"
                elif selection["screened_k"] != 4:
                    reason = "aic_prefers_other"
                else:
                    reason = "selected_four"
                reasons[reason] += 1
                gains.append(c["pooled_validation_gain"])
            result.append(
                {
                    "cell": cell,
                    "family": family,
                    "observations": len(group),
                    "sequential_k4_reasons": dict(reasons),
                    "eligible_k4_validation_gain": describe(gains),
                }
            )
    return result


def diagnostics(rows, study):
    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "shared_bytes",
        Path(__file__).resolve().parents[1] / "examples/shared-predictive/check_shared.py",
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    groups = {}
    for row in rows:
        outer = checker.unpack(
            (study / row["archive"]).read_bytes(), (study / row["manifest"]).read_bytes(), row
        )
        for family in ("full", "tied"):
            r = row["arms"][family]
            files = checker.unpack(
                outer[family + "/" + r["archive"]], outer[family + "/" + r["manifest"]], r
            )
            ext = {r["prefix"]: r for r in json.loads(files["external_folds.json"])}
            for prefix in ["full"] + [f"fold_{i:03d}" for i in range(5)]:
                key = row["cell"], family, ("full" if prefix == "full" else "folds")
                entry = groups.setdefault(
                    key,
                    {
                        "cell": row["cell"],
                        "family": family,
                        "space": key[2],
                        "available_spaces": 0,
                        "eligible_four": 0,
                        "minimum_eigenvalues": [],
                        "condition_numbers": [],
                        "minimum_hard_sizes": [],
                        "external_gain_four": [],
                    },
                )
                path = f"workflow/{prefix}/run/"
                if path + "gmm_candidates.npz" not in files:
                    continue
                cs = json.loads(files[path + "candidates.json"])
                four = next(c for c in cs if c["k"] == 4)
                if four["status"] == "insufficient_samples":
                    continue
                entry["available_spaces"] += 1
                entry["eligible_four"] += four["status"] == "eligible"
                entry["minimum_hard_sizes"].append(min(four["hard_cluster_sizes"]))
                with np.load(
                    io.BytesIO(files[path + "gmm_candidates.npz"]), allow_pickle=False
                ) as p:
                    eigen = np.linalg.eigvalsh(p["k4_covariances"])
                    entry["minimum_eigenvalues"].append(float(eigen.min()))
                    entry["condition_numbers"].append(float((eigen[:, -1] / eigen[:, 0]).max()))
                if prefix == "full":
                    with np.load(
                        io.BytesIO(files["external_predictions.npz"]), allow_pickle=False
                    ) as p:
                        entry["external_gain_four"].append(float((p["k4"] - p["k1"]).mean()))
                elif ext[prefix]["status"] == "complete":
                    entry["external_gain_four"].append(ext[prefix]["gains"].get("4"))
    result = []
    for value in groups.values():
        result.append({k: describe(v) if isinstance(v, list) else v for k, v in value.items()})
    return result


def figures(summary, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "full_ordinary": "#D55E00",
        "full_screened": "#0072B2",
        "tied_ordinary": "#CC79A7",
        "tied_screened": "#009E73",
    }
    labels = {
        "full_ordinary": "Full / ordinary",
        "full_screened": "Full / screen",
        "tied_ordinary": "Shared / ordinary",
        "tied_screened": "Shared / screen",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 1.3]})
    for ax, role in zip(axes, ("alternative", "control"), strict=True):
        cells = [c for c in summary["cells"] if c["cell"]["role"] == role]
        for y, c in enumerate(cells):
            for j, (method, color) in enumerate(colors.items()):
                r = c["methods"][method]["rate"]
                if not r["available"]:
                    continue
                x = r["rate_among_available"]
                lo, hi = r["bonferroni_familywise_95"]
                dy = (j - 1.5) * 0.15
                ax.errorbar(
                    x,
                    y + dy,
                    xerr=[[x - lo], [hi - x]],
                    fmt="o" if j % 2 == 0 else "s",
                    color=color,
                    capsize=2,
                    label=labels[method] if y == 0 and role == "alternative" else None,
                )
        ax.set_yticks(
            range(len(cells)),
            [
                f"N={c['cell']['n_units']} / signal {c['cell']['signal']:g}"
                if role == "alternative"
                else c["cell"]["law"].replace("_gaussian", "").replace("_", " ").capitalize()
                + f" / N={c['cell']['n_units']}"
                for c in cells
            ],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.03)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel(
            "Fraction selecting K=4" if role == "alternative" else "Fraction selecting K>1"
        )
        ax.set_title(
            "Four-group alternatives" if role == "alternative" else "Gaussian control laws"
        )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(
        "Prespecified covariance families: ordinary and screened AIC\n"
        "95% simultaneous exact intervals across 40 primary rates",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    target.mkdir(parents=True, exist_ok=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(target / f"paired-selection-rates.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "study", "output-dir", "figures"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.audit / "summary.json").read_text())
    with gzip.open(args.audit / "points.jsonl.gz", "rt") as h:
        rows = [json.loads(s) for s in h]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "summary.json": summary,
        "gate-mechanisms.json": mechanisms(rows),
        "geometry-prediction.json": diagnostics(rows, args.study),
    }
    for name, value in outputs.items():
        (args.output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
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
    with (args.output_dir / "primary-rates.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in summary["primary_rates"])
    figures(summary, args.figures)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output_dir.iterdir()
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
                "paired_comparisons": len(summary["paired"]),
            }
        )
    )


if __name__ == "__main__":
    main()
