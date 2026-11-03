"""Truth-separated, all-seed sensitivity summaries; no winning weight selection."""

from collections import Counter
from pathlib import Path

import numpy as np

from super_deli.benchmarking.gaussian_graph_report import difference_interval
from super_deli.benchmarking.matched_study import inventory
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_fit import recovery
from super_deli.benchmarking.unknown_k_report import interval
from super_deli.benchmarking.view_signal_metrics import linear_recovery
from super_deli.benchmarking.view_signal_report import spread
from super_deli.benchmarking.view_weight_config import ARMS, JOINT, SEEDS, tasks
from super_deli.benchmarking.view_weight_study import check_frozen
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def assess(root, config):
    rows, stability = [], []
    for task in tasks(config, "main"):
        base = root / "main" / task["key"]
        source = arrays(base / "generating-reference.npz")
        for record in read_json(base / "fits.json"):
            arm = record["arm"]
            folder = base / arm
            row = {
                "key": task["key"],
                "case": task["case"],
                "n": task["n"],
                "arm": arm,
                "status": record["status"],
                "error": record.get("error"),
                "selected_k": record.get("selected_k"),
                "endpoint": False,
                "coverage": 0.0,
                "ari": None,
                "assigned_ari": None,
            }
            if row["status"] == "available":
                point = arrays(folder / ("mixture.npz" if arm.endswith("gmm") else "point.npz"))
                row.update(recovery(point["labels"], source["truth"]))
                row["endpoint"] = (
                    row["selected_k"] > 1
                    if task["case"].startswith("gaussian_")
                    else row["selected_k"] == 4 and row["ari"] >= 0.8
                )
                row["linear_morphology"] = linear_recovery(
                    point["scores"], source["latent_morphology"]
                )
                row["linear_ecology"] = linear_recovery(point["scores"], source["latent_ecology"])
                if arm in JOINT:
                    summary = read_json(folder / "summary.json")
                    row["standardized_mse_by_view"] = summary["per_view_standardized_mse"]
                if record["draws_requested"]:
                    records = read_json(folder / "subsamples.json")
                    support = arrays(folder / "stability.npz")
                    done = [r for r in records if r["status"] == "complete"]
                    values = [
                        recovery(np.array(r["labels"]), source["truth"][r["training_indices"]])[
                            "ari"
                        ]
                        for r in done
                    ]
                    point_reference = point["labels"].copy()
                    noise = point_reference < 0
                    point_reference[noise] = (
                        np.arange(sum(noise)) + max(int(point_reference.max()), 0) + 1
                    )
                    point_agreement = [
                        recovery(np.array(r["labels"]), point_reference[r["training_indices"]])[
                            "ari"
                        ]
                        for r in done
                    ]
                    upper = np.triu_indices(task["n"], 1)
                    stability.append(
                        {
                            "key": task["key"],
                            "case": task["case"],
                            "n": task["n"],
                            "arm": arm,
                            "requested": len(records),
                            "successful": len(done),
                            "failed": len(records) - len(done),
                            "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in done)),
                            "included_truth_ari": values,
                            "own_point_ari": point_agreement,
                            "unit_coverage": float(np.mean(support["unit_successful"] > 0)),
                            "pair_coverage": float(np.mean(support["pair_successful"][upper] > 0)),
                            "mean_coassignment": float(np.nanmean(support["coassignment"][upper]))
                            if done
                            else None,
                            "denominator": (
                                "Successful unit/pair inclusions; three repeated native draws, "
                                "one dataset"
                            ),
                        }
                    )
            rows.append(row)
    return rows, stability


def summarize(rows):
    groups, primary, paired = [], [], []
    for case, n in sorted({(r["case"], r["n"]) for r in rows}):
        cell = [r for r in rows if (r["case"], r["n"]) == (case, n)]
        for arm in ARMS:
            selected = [r for r in cell if r["arm"] == arm]
            done = [r for r in selected if r["status"] == "available"]
            size, unknown = len(selected), len(selected) - len(done)
            count = sum(r["endpoint"] for r in selected)
            group = {
                "case": case,
                "n": n,
                "arm": arm,
                "requested": size,
                "available": len(done),
                "failures": unknown,
                "endpoint_count": count,
                "endpoint_rate_all_requests": count / size,
                "endpoint": "nontrivial_return"
                if case.startswith("gaussian_")
                else "k4_full_ari_ge_0_8",
                "available_only_rate": count / len(done) if done else None,
                "failure_bounds": [count / size, (count + unknown) / size],
                "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in done)),
            }
            for metric in ("coverage", "ari", "assigned_ari"):
                group[metric] = spread([r[metric] for r in done])
            for view in ("morphology", "ecology"):
                group["linear_" + view] = [
                    spread([r["linear_" + view]["r2"][i] for r in done]) for i in (0, 1)
                ]
            if arm in JOINT:
                group["standardized_mse_by_view"] = [
                    spread([r["standardized_mse_by_view"][i] for r in done]) for i in (0, 1)
                ]
                primary.append(
                    {
                        **{k: group[k] for k in ("case", "n", "arm", "requested", "endpoint")},
                        "count": count,
                        "rate": count / size,
                        "unavailable": unknown,
                        "identification_bounds": group["failure_bounds"],
                        "marginal_95_interval": interval(count, size, 0.05),
                        "simultaneous_95_interval": interval(count, size, 0.05 / 108),
                        "simultaneous_failure_interval": [
                            interval(count, size, 0.05 / 108)[0],
                            interval(count + unknown, size, 0.05 / 108)[1],
                        ],
                    }
                )
            groups.append(group)
        for seed in SEEDS:
            equal = {r["key"]: r for r in cell if r["arm"] == f"joint_equal_{seed}"}
            for weight in ("morphology", "ecology"):
                alternative = {r["key"]: r for r in cell if r["arm"] == f"joint_{weight}_{seed}"}
                common = [
                    k
                    for k in equal
                    if equal[k]["status"] == alternative[k]["status"] == "available"
                ]
                positive = sum(
                    alternative[k]["endpoint"] and not equal[k]["endpoint"] for k in common
                )
                negative = sum(
                    equal[k]["endpoint"] and not alternative[k]["endpoint"] for k in common
                )
                size, unknown = len(equal), len(equal) - len(common)
                paired.append(
                    {
                        "case": case,
                        "n": n,
                        "neural_seed": seed,
                        "weight": weight,
                        "requested": size,
                        "common_available": len(common),
                        "unknown_pairs": unknown,
                        "weighted_only": positive,
                        "equal_only": negative,
                        "rate_difference_known_contribution": (positive - negative) / size,
                        "identification_bounds": [
                            (positive - negative - unknown) / size,
                            (positive - negative + unknown) / size,
                        ],
                        "marginal_95_interval": difference_interval(positive, negative, size, 0.05),
                        "simultaneous_95_interval": difference_interval(
                            positive, negative, size, 0.05 / 72
                        ),
                        "simultaneous_failure_interval": [
                            difference_interval(positive, negative + unknown, size, 0.05 / 72)[0],
                            difference_interval(positive + unknown, negative, size, 0.05 / 72)[1],
                        ],
                    }
                )
    return groups, primary, paired


def report(root):
    root = Path(root)
    frozen, config = check_frozen(root)
    audit = read_json(root / "main-audit.json")
    require(
        audit["passed"]
        and audit["freeze_sha256"] == fingerprint_file(root / "frozen.json").sha256
        and inventory(root / "main") == read_json(root / "main-inventory.json"),
        "Bound unchanged main audit required",
    )
    rows, stability = assess(root, config)
    groups, primary, paired = summarize(rows)
    require(config.fixture or (len(primary), len(paired)) == (108, 72), "Frozen primary endpoints")
    summary = {
        "schema_version": 1,
        "observations": len(tasks(config, "main")),
        "counts": frozen["counts"],
        "primary_rates": len(primary),
        "paired_comparisons": len(paired),
        "failed_points": sum(r["status"] != "available" for r in rows),
        "main_audit": audit,
        "defaults_changed": False,
        "historical_pvalues_changed": False,
        "interpretation": (
            "Prespecified sensitivity, five independent datasets/cell. "
            "Weights/seeds/subsamples are repeated measurements. No winning weight, "
            "species support or calibrated control error guarantee. "
            "Different weighted training losses are not ranked."
        ),
    }
    for name, value in (
        ("assessments", rows),
        ("all-groups", groups),
        ("primary-rates", primary),
        ("paired-comparisons", paired),
        ("native-stability", stability),
        ("summary", summary),
    ):
        write_json(value, root / (name + ".json"))
    return summary
