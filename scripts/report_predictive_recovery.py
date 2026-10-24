"""Publish compact audited diagnostic tables and figures; keep fitted evidence local."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def points(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(s) for s in handle]


def describe(values):
    from super_deli.benchmarking.grouped_power_analysis import describe as summarize

    result = summarize(values)
    result.pop("mcse_mean", None)
    result["scope"] = "Descriptive; no independent-fold standard error"
    return result


def archived_summary(rows):
    result = []
    for name in dict.fromkeys(r["cell"] for r in rows):
        group = [r for r in rows if r["cell"] == name]
        config = group[0]["cell_config"]
        k4 = []
        outcomes = {
            rule: Counter() for rule in ("ordinary", "eligibility_only", "gain_only", "actual")
        }
        sequential = Counter()
        for r in group:
            c = next(v for v in r["gates"]["candidates"] if v["k"] == 4)
            k4.append(c)
            for rule in outcomes:
                outcomes[rule][str(r["gates"][rule + "_k"])] += 1
            if c["full_status"] != "eligible":
                reason = "full_ineligible"
            elif c["fold_ineligible_count"]:
                reason = "fold_ineligible"
            elif not c["passes"]["actual"]:
                reason = "nonpositive_gain"
            elif r["gates"]["actual_k"] != 4:
                reason = "aic_prefers_other"
            else:
                reason = "selected_four"
            sequential[reason] += 1
        spaces = [
            s
            for r in group
            for s in r["spaces"]
            if s["prefix"] != "full" and s["status"] == "complete"
        ]
        four = [next(c for c in s["candidates"] if c["k"] == 4) for s in spaces]
        result.append(
            {
                "cell": config,
                "observations": len(group),
                "sequential_k4_reasons": dict(sequential),
                "k4_pooled_validation_gain_all_fitted": describe(
                    [c["pooled_gain_all_fitted_folds"] for c in k4]
                ),
                "k4_fold_external_gain": describe(
                    [c.get("heldout_mean_logdensity_gain_over_one") for c in four]
                ),
                "k4_fold_minimum_eigenvalue_over_regularizer": describe(
                    [c.get("minimum_eigenvalue_over_regularizer") for c in four]
                ),
                "k4_fold_minimum_hard_size": describe(
                    [min(c["hard_cluster_sizes"]) for c in four if "hard_cluster_sizes" in c]
                ),
                "signal_and_maps": signal_summary(spaces),
                "counterfactual_k_counts": {
                    rule: dict(values) for rule, values in outcomes.items()
                },
                "scope": "Descriptive archived gate removals; never new production selectors",
            }
        )
    return result


def signal_summary(spaces):
    """Describe fixed training maps without treating folds as independent replicates."""
    complete = [s for s in spaces if s["status"] == "complete"]
    result = {}
    for field in ("fused_axis_r2", "joint_axis_r2"):
        result[field] = {
            axis: describe([s.get(field, {}).get(axis) for s in complete]) for axis in ("a", "b")
        }
    result["upstream_axes"] = {
        view: {
            field: {
                axis: describe(
                    [
                        s.get("upstream_axes", {}).get(view, {}).get(field, {}).get(axis)
                        for s in complete
                    ]
                )
                for axis in ("a", "b")
            }
            for field in ("raw_group_axis_r2", "source_axis_r2", "profile_axis_r2", "count_axis_r2")
        }
        for view in ("morphological", "ecological")
    }
    result["maximum_map_angle_radians"] = describe(
        [max(m["angles_from_full_radians"]) for s in complete for m in s.get("maps", [])]
    )
    result["minimum_generating_count"] = describe(
        [min(s["generating_counts"]) for s in complete if "generating_counts" in s]
    )
    return result


def fresh_details(rows, study):
    import numpy as np

    from super_deli.benchmarking.local_archives import materialize

    details = []
    for row in rows:
        with materialize(study / row["archive"], row, scratch=study.parent / ".scratch") as base:
            changes = []
            for prefix in ["full"] + [f"fold_{i:03d}" for i in range(5)]:
                a = base / "part_000/workflow" / prefix / "run"
                b = base / "part_extended/workflow" / prefix / "run"
                if (
                    not (a / "gmm_candidates.npz").exists()
                    or not (b / "gmm_candidates.npz").exists()
                ):
                    continue
                with (
                    np.load(a / "gmm_candidates.npz", allow_pickle=False) as x,
                    np.load(b / "gmm_candidates.npz", allow_pickle=False) as y,
                ):
                    np.testing.assert_array_equal(x["scores"], y["scores"])
                for relative in (
                    "fusion.npz",
                    "view_000/parameters/pca.npz",
                    "view_001/parameters/pca.npz",
                    "view_000/parameters/profiles.npz",
                    "view_001/parameters/profiles.npz",
                ):
                    with (
                        np.load(a / relative, allow_pickle=False) as x,
                        np.load(b / relative, allow_pickle=False) as y,
                    ):
                        assert set(x.files) == set(y.files)
                        for key in x.files:
                            np.testing.assert_array_equal(x[key], y[key])
                ca, cb = read(a / "candidates.json"), read(b / "candidates.json")
                ka, kb = (next(c for c in cs if c["k"] == 4) for cs in (ca, cb))
                ma = read(base / "part_000/diagnostics/metrics.json")
                mb = read(base / "part_extended/diagnostics/metrics.json")
                sa = next(s for s in ma["spaces"] if s["prefix"] == prefix)
                sb = next(s for s in mb["spaces"] if s["prefix"] == prefix)
                xa, xb = (next(c for c in s["candidates"] if c["k"] == 4) for s in (sa, sb))
                changes.append(
                    {
                        "prefix": prefix,
                        "scores_and_maps_identical": True,
                        "k4_loglikelihood_change": kb.get("log_likelihood", 0)
                        - ka.get("log_likelihood", 0),
                        "k4_eligibility_before": ka["status"],
                        "k4_eligibility_after": kb["status"],
                        "k4_external_gain_change": xb.get(
                            "heldout_mean_logdensity_gain_over_one", 0
                        )
                        - xa.get("heldout_mean_logdensity_gain_over_one", 0),
                        "ari_before": xa.get("ari"),
                        "ari_after": xb.get("ari"),
                        "baseline_k4_parameters_per_unit": xa.get("parameters_per_unit"),
                        "baseline_joint_axis_r2": sa.get("joint_axis_r2"),
                        "baseline_fused_axis_r2": sa.get("fused_axis_r2"),
                        "baseline_map_angles": sa.get("maps"),
                        "baseline_space": sa,
                    }
                )
            details.append(
                {
                    "cell": row["cell"],
                    "repetition": row["repetition"],
                    "optimization": changes,
                    "oracles": row["oracles"],
                }
            )
    result = []
    for cell in dict.fromkeys(r["cell"] for r in rows):
        group = [r for r in details if r["cell"] == cell]
        out = {"cell": cell, "observations": len(group), "spaces": []}
        for prefix in ["full"] + [f"fold_{i:03d}" for i in range(5)]:
            pairs = [v for r in group for v in r["optimization"] if v["prefix"] == prefix]
            oracles = [v for r in group for v in r["oracles"] if v["prefix"] == prefix]
            available = [v for v in oracles if v["status"] == "complete"]
            out["spaces"].append(
                {
                    "prefix": prefix,
                    "available_optimization_pairs": len(pairs),
                    "scores_and_maps_identical": all(v["scores_and_maps_identical"] for v in pairs),
                    "k4_loglikelihood_change": describe(
                        [v["k4_loglikelihood_change"] for v in pairs]
                    ),
                    "eligibility_lost": sum(
                        v["k4_eligibility_before"] == "eligible"
                        and v["k4_eligibility_after"] != "eligible"
                        for v in pairs
                    ),
                    "eligibility_gained": sum(
                        v["k4_eligibility_before"] != "eligible"
                        and v["k4_eligibility_after"] == "eligible"
                        for v in pairs
                    ),
                    "k4_external_gain_change": describe(
                        [v["k4_external_gain_change"] for v in pairs]
                    ),
                    "signal_and_maps": signal_summary([v["baseline_space"] for v in pairs]),
                    "baseline_ari": describe([v["ari_before"] for v in pairs]),
                    "extended_ari": describe([v["ari_after"] for v in pairs]),
                    "oracle_attempted": len(oracles),
                    "oracle_completed": len(available),
                    "oracle_eligible": sum(v["eligible"] for v in available),
                    "oracle_ari": describe([v["ari"] for v in available]),
                    "oracle_loglikelihood_gain_over_unsupervised_k4": describe(
                        [v["loglikelihood_gain_over_unsupervised_k4"] for v in available]
                    ),
                    "oracle_external_gain": describe(
                        [v["external_mean_logdensity_gain"] for v in available]
                    ),
                }
            )
        result.append(out)
    return result


def fresh_gates(rows):
    """Retain dataset-level gate counts for each prespecified partition and arm."""
    result = []
    for cell in dict.fromkeys(r["cell"] for r in rows):
        group = [r for r in rows if r["cell"] == cell]
        for part in [p["name"] for p in group[0]["parts"]]:
            members = [next(p for p in r["parts"] if p["name"] == part) for r in group]
            reasons = Counter()
            counts = {
                rule: Counter() for rule in ("ordinary", "eligibility_only", "gain_only", "actual")
            }
            gains = []
            for p in members:
                gates = p["gates"]
                if gates is None:
                    if p["selection"]["status"] == "complete":
                        raise ValueError("Complete selection is missing gate diagnostics")
                    reasons["withheld"] += 1
                    for rule in counts:
                        counts[rule]["None"] += 1
                    continue
                c = next(v for v in gates["candidates"] if v["k"] == 4)
                if p["selection"]["status"] != "complete":
                    reason = "withheld"
                elif c["full_status"] != "eligible":
                    reason = "full_ineligible"
                elif c["fold_ineligible_count"]:
                    reason = "fold_ineligible"
                elif not c["passes"]["actual"]:
                    reason = "nonpositive_gain"
                elif gates["actual_k"] != 4:
                    reason = "aic_prefers_other"
                else:
                    reason = "selected_four"
                reasons[reason] += 1
                gains.append(c["pooled_gain_all_fitted_folds"])
                for rule in counts:
                    counts[rule][str(gates[rule + "_k"])] += 1
            result.append(
                {
                    "cell": cell,
                    "partition": part,
                    "observations": len(members),
                    "sequential_k4_reasons": dict(reasons),
                    "k4_pooled_validation_gain_all_fitted": describe(gains),
                    "counterfactual_k_counts": {rule: dict(v) for rule, v in counts.items()},
                    "scope": "Repeated partitions describe the same datasets",
                }
            )
    return result


def figures(summary, archived, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    colors = {
        "baseline_ordinary": "#D55E00",
        "baseline_screened": "#0072B2",
        "extended_ordinary": "#CC79A7",
        "extended_screened": "#009E73",
    }
    labels = {
        "baseline_ordinary": "10 starts / ordinary",
        "baseline_screened": "10 starts / screen",
        "extended_ordinary": "100 starts / ordinary",
        "extended_screened": "100 starts / screen",
    }
    for ax, role in zip(axes, ("alternative", "control"), strict=True):
        cells = [c for c in summary["cells"] if c["cell"]["role"] == role]
        ticks = []
        for y, c in enumerate(cells):
            ticks.append(
                f"N={c['cell']['n_units']} / signal {c['cell']['signal']:g}"
                if role == "alternative"
                else "Gaussian / N=64"
            )
            for i, (method, color) in enumerate(colors.items()):
                r = c["methods"][method]["rate"]
                if not r["available"]:
                    continue
                x = r["rate_among_available"]
                lo, hi = r["bonferroni_familywise_95"]
                dy = (i - 1.5) * 0.13
                ax.errorbar(
                    x,
                    y + dy,
                    xerr=[[x - lo], [hi - x]],
                    fmt="o" if i % 2 == 0 else "s",
                    color=color,
                    capsize=3,
                    label=labels[method] if y == 0 and role == "control" else None,
                )
                ax.text(
                    1.05,
                    y + dy,
                    f"{r['events']}/{r['available']}",
                    va="center",
                    fontsize=8,
                    color=color,
                )
        ax.set_yticks(range(len(ticks)), ticks, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(len(ticks) - 0.4, -0.4)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel(
            "Fraction selecting K = 4" if role == "alternative" else "Fraction selecting K > 1"
        )
        ax.set_title("Four-group alternatives" if role == "alternative" else "Gaussian control")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Fresh first-partition outcomes: ten versus 100 starts\n"
        "95% simultaneous exact intervals across 16 prespecified rates",
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory / f"first-partition-rates.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    rows = [r for r in archived if r["cell"]["role"] == "alternative"]
    categories = {
        "full_ineligible": ("Full fit ineligible", "#999999"),
        "fold_ineligible": ("A fold ineligible", "#E69F00"),
        "nonpositive_gain": ("Nonpositive gain", "#CC79A7"),
        "aic_prefers_other": ("AIC prefers another K", "#56B4E9"),
        "selected_four": ("K=4 selected", "#009E73"),
    }
    left = [0.0] * len(rows)
    for key, (label, color) in categories.items():
        x = [r["sequential_k4_reasons"].get(key, 0) / r["observations"] for r in rows]
        ax.barh(range(len(rows)), x, left=left, label=label, color=color)
        for y, (lo, v, r) in enumerate(zip(left, x, rows, strict=True)):
            if v > 0.06:
                ax.text(
                    lo + v / 2,
                    y,
                    str(r["sequential_k4_reasons"][key]),
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        left = [a + b for a, b in zip(left, x, strict=True)]
    ax.set_yticks(
        range(len(rows)),
        [f"N={r['cell']['n_units']} / signal {r['cell']['signal']:g}" for r in rows],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of archived datasets; 25 per cell")
    ax.set_title("Archived K=4 decisions: sequential gate diagnosis")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory / f"archived-gates.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "archived", "study", "output-dir", "figures"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    summary = read(args.audit / "summary.json")
    rows = points(args.audit / "points.jsonl.gz")
    old = points(args.archived / "points.jsonl.gz")
    archived = archived_summary(old)
    optimization = fresh_details(rows, args.study)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("summary.json", summary),
        ("archived-mechanisms.json", archived),
        ("optimization.json", optimization),
        ("fresh-gates.json", fresh_gates(rows)),
    ):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "primary-rates.csv").open("w", newline="") as out:
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
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for r in summary["primary_rates"]:
            writer.writerow({k: r[k] for k in fields})
    figures(summary, archived, args.figures)
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
                "fresh_observations": len(rows),
                "archived_observations": len(old),
                "primary_rates": len(summary["primary_rates"]),
                "all_compared_score_maps_verified_identical": all(
                    s["scores_and_maps_identical"] for r in optimization for s in r["spaces"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
