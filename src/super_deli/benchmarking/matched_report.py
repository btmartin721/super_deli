"""Truth-separated pilot recovery, coverage and conditional native stability."""

from collections import Counter
from pathlib import Path

import numpy as np

from super_deli.benchmarking.gaussian_graph_report import difference_interval
from super_deli.benchmarking.matched_config import ARMS, CASES, tasks
from super_deli.benchmarking.matched_study import check_frozen, inventory
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_fit import recovery
from super_deli.benchmarking.unknown_k_report import interval
from super_deli.io.config import read_json, write_json


def endpoints(case):
    if case.startswith("gaussian_"):
        return ("nontrivial_return",)
    if case in ("four_redundant", "four_complementary", "morphology_only"):
        return ("returned_correct_k", "returned_correct_k_ari08")
    return ()


def good(row, name):
    return (
        row["status"] == "available" and row["selected_k"] > 1
        if name == "nontrivial_return"
        else row[name]
    )


def assess(root, config):
    rows = []
    stability = []
    for task in tasks(config, "main"):
        out = root / "main" / task["key"]
        source = arrays(out / "generating-reference.npz")
        truth = source.get("truth")
        for record in read_json(out / "fits.json"):
            if record["kind"] != "point":
                continue
            arm = record["arm"]
            folder = out / arm
            row = {
                **record,
                "key": task["key"],
                "case": task["case"],
                "n": task["n"],
                "generating_k": len(set(truth)) if truth is not None else None,
                "returned_correct_k": False,
                "returned_correct_k_ari08": False,
                "coverage": 0.0,
                "ari": None,
                "assigned_ari": None,
            }
            if record["status"] == "available":
                a = arrays(folder / ("mixture.npz" if arm.endswith("gmm") else "point.npz"))
                labels = a["labels"]
                row["coverage"] = float(np.mean(labels >= 0))
                row["noise_count"] = int(sum(labels < 0))
                if truth is not None:
                    row.update(recovery(labels, truth))
                    row["returned_correct_k"] = row["selected_k"] == row["generating_k"]
                    row["returned_correct_k_ari08"] = (
                        row["returned_correct_k"] and row["ari"] >= 0.8
                    )
                if task["case"] == "discordant":
                    row["morphology_reference_ari"] = recovery(labels, source["morphology_truth"])[
                        "ari"
                    ]
                    row["ecology_reference_ari"] = recovery(labels, source["ecology_truth"])["ari"]
                if (folder / "training.json").exists():
                    diag = read_json(folder / "training.json")
                    row["training_loss"] = diag.get("final_loss")
                    row["per_view_standardized_mse"] = diag.get("per_view_standardized_mse")
                if record["draws_requested"]:
                    subs = read_json(folder / "subsamples.json")
                    s = arrays(folder / "stability.npz")
                    done = [r for r in subs if r["status"] == "complete"]
                    rates = (
                        [
                            recovery(np.array(r["labels"]), truth[np.array(r["training_indices"])])[
                                "ari"
                            ]
                            for r in done
                        ]
                        if truth is not None
                        else []
                    )
                    stability.append(
                        {
                            "case": task["case"],
                            "n": task["n"],
                            "arm": arm,
                            "requested": len(subs),
                            "successful": len(done),
                            "failed": len(subs) - len(done),
                            "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in done)),
                            "mean_included_truth_ari": float(np.mean(rates)) if rates else None,
                            "unit_coverage": float(np.mean(s["unit_successful"] > 0)),
                            "pair_coverage": float(
                                np.mean(s["pair_successful"][np.triu_indices(task["n"], 1)] > 0)
                            ),
                            "mean_noise_frequency": float(np.nanmean(s["noise_frequency"]))
                            if done
                            else None,
                            "mean_coassignment": float(
                                np.nanmean(s["coassignment"][np.triu_indices(task["n"], 1)])
                            )
                            if done
                            else None,
                            "denominator": (
                                "successful unit/pair inclusions; five repeated draws, "
                                "one biological dataset"
                            ),
                        }
                    )
            rows.append(row)
    return rows, stability


def report(root):
    root = Path(root)
    f, config = check_frozen(root)
    require(
        read_json(root / "main-audit.json")["passed"]
        and inventory(root / "main") == read_json(root / "main-inventory.json"),
        "Passed unchanged main audit required",
    )
    rows, stability = assess(root, config)
    groups = []
    primary = []
    paired = []
    for case in CASES:
        for n in sorted({r["n"] for r in rows}):
            cell = [r for r in rows if (r["case"], r["n"]) == (case, n)]
            if not cell:
                continue
            for arm in ARMS:
                selected = [r for r in cell if r["arm"] == arm]
                done = [r for r in selected if r["status"] == "available"]
                size = len(selected)
                unknown = size - len(done)
                vals = [r["ari"] for r in done if r["ari"] is not None]
                group = {
                    "case": case,
                    "n": n,
                    "arm": arm,
                    "requested": size,
                    "available": len(done),
                    "statuses": dict(Counter(r["status"] for r in selected)),
                    "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in done)),
                    "mean_coverage_all_requests": float(np.mean([r["coverage"] for r in selected])),
                    "available_mean_ari": float(np.mean(vals)) if vals else None,
                    "all_request_ari_bounds": [
                        (sum(vals) - unknown) / size,
                        (sum(vals) + unknown) / size,
                    ]
                    if selected[0]["generating_k"] is not None
                    else None,
                    "overpartitioned_returns": sum(
                        r["selected_k"] > r["generating_k"] for r in done
                    )
                    if selected[0]["generating_k"] is not None
                    else None,
                }
                if case == "discordant":
                    for view in ("morphology", "ecology"):
                        group[view + "_reference_mean_ari"] = (
                            float(np.mean([r[view + "_reference_ari"] for r in done]))
                            if done
                            else None
                        )
                groups.append(group)
                for endpoint in endpoints(case):
                    count = sum(good(r, endpoint) for r in selected)
                    upper = count + unknown if endpoint == "nontrivial_return" else count
                    primary.append(
                        {
                            **{k: group[k] for k in ("case", "n", "arm", "requested")},
                            "endpoint": endpoint,
                            "count": count,
                            "rate": count / size,
                            "unavailable": unknown,
                            "identification_bounds": [count / size, upper / size],
                            "marginal_95_interval": interval(count, size, 0.05),
                            "simultaneous_95_interval": interval(count, size, 0.05 / 224),
                            "simultaneous_failure_interval": [
                                interval(count, size, 0.05 / 224)[0],
                                interval(upper, size, 0.05 / 224)[1],
                            ],
                        }
                    )
            for seed in (0, 101, 202):
                a = {r["key"]: r for r in cell if r["arm"] == f"paired_{seed}"}
                for baseline in (f"morphology_{seed}", f"ecology_{seed}", "integrative_gmm"):
                    b = {r["key"]: r for r in cell if r["arm"] == baseline}
                    for endpoint in endpoints(case):
                        positive = sum(good(a[k], endpoint) and not good(b[k], endpoint) for k in a)
                        negative = sum(good(b[k], endpoint) and not good(a[k], endpoint) for k in a)
                        size = len(a)
                        paired.append(
                            {
                                "case": case,
                                "n": n,
                                "neural_seed": seed,
                                "baseline": baseline,
                                "endpoint": endpoint,
                                "requested": size,
                                "paired_only": positive,
                                "baseline_only": negative,
                                "rate_difference": (positive - negative) / size,
                                "marginal_95_interval": difference_interval(
                                    positive, negative, size, 0.05
                                ),
                                "simultaneous_95_interval": difference_interval(
                                    positive, negative, size, 0.05 / 144
                                ),
                                "common_available": sum(
                                    a[k]["status"] == b[k]["status"] == "available" for k in a
                                ),
                            }
                        )
    if not config.fixture:
        require(
            len(rows) == 2240
            and len(groups) == 224
            and len(primary) == 224
            and len(paired) == 144
            and len(stability) == 88,
            "Frozen report families",
        )
    result = {
        "schema_version": 1,
        "fixture": config.fixture,
        "observations": len(tasks(config, "main")),
        "main_workflow_fits": f["counts"]["main"]["workflows"],
        "point_outputs": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "groups": groups,
        "primary_rates": primary,
        "paired_comparisons": paired,
        "stability": stability,
        "primary_interval_family": 224,
        "paired_interval_family": 144,
        "interpretation": [
            "Shared representation is a transductive research baseline, not validated species",
            "All three seeds retained as repeated measurements, no winning seed or default",
            "Equal view loss does not imply equal biological influence",
            "No new p-values or transfer of GMM significance",
            "Ten independent observations/cell give pilot evidence; "
            "five subsamples only describe conditional sensitivity",
            "Discordant/heavy-tail stress has no joint species truth; moons is secondary",
            "GMM K=1..4 and density unknown K have different search/noise policies",
        ],
    }
    write_json(rows, root / "point-assessments.json")
    write_json(result, root / "aggregate.json")
    return {
        "fixture": config.fixture,
        "point_outputs": len(rows),
        "groups": len(groups),
        "primary_rates": len(primary),
        "paired_comparisons": len(paired),
    }
