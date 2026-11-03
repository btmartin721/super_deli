"""Truth-separated neural architecture comparisons; dimension and compute stay explicit."""

from collections import Counter
from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_study import inventory
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_fit import recovery
from super_deli.benchmarking.neural_summary import summarize_rates
from super_deli.benchmarking.separate_config import ARMS, NATIVE, PRIMARY, SEEDS, tasks
from super_deli.benchmarking.separate_study import check_frozen
from super_deli.benchmarking.view_signal_metrics import linear_recovery
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def assess(root, config):
    rows, stability = [], []
    for task in tasks(config, "main"):
        base = root / "main" / task["key"]
        source = arrays(base / "generating-reference.npz")
        truth = source.get("truth")
        for record in read_json(base / "fits.json"):
            arm = record["arm"]
            folder = base / arm
            endpoint = (
                None
                if truth is None
                else "nontrivial_return"
                if task["case"].startswith("gaussian_")
                else "k2_full_ari_ge_0_8"
                if task["case"] == "moons"
                else "k4_full_ari_ge_0_8"
            )
            row = {
                "key": task["key"],
                "case": task["case"],
                "n": task["n"],
                "arm": arm,
                "status": record["status"],
                "error": record.get("error"),
                "selected_k": record.get("selected_k"),
                "endpoint_name": endpoint,
                "endpoint": False,
                "coverage": 0.0,
                "ari": None,
                "assigned_ari": None,
                "workflow_seconds": record.get("seconds"),
            }
            if row["status"] == "available":
                point = arrays(folder / ("mixture.npz" if arm.endswith("gmm") else "point.npz"))
                labels = point["labels"]
                row["coverage"] = float(np.mean(labels >= 0))
                row["noise_count"] = int(sum(labels < 0))
                if truth is not None:
                    row.update(recovery(labels, truth))
                    row["endpoint"] = (
                        row["selected_k"] > 1
                        if task["case"].startswith("gaussian_")
                        else row["selected_k"] == len(set(truth)) and row["ari"] >= 0.8
                    )
                else:
                    for view in ("morphology", "ecology"):
                        row[view + "_reference_ari"] = recovery(labels, source[view + "_truth"])[
                            "ari"
                        ]
                for view in ("morphology", "ecology"):
                    row["linear_" + view] = linear_recovery(
                        point["scores"], source["latent_" + view]
                    )
                singular = np.linalg.svd(point["scores"], compute_uv=False)
                row["effective_rank"] = int(np.sum(singular > 1e-12 * singular[0]))
                row["neural_parameters"] = sum(
                    v.size
                    for k, v in point.items()
                    if k.startswith(("encoder_", "decoder_"))
                    or "__encoder_" in k
                    or "__decoder_" in k
                )
                row["point_neural_trainings"] = (
                    2
                    if arm.startswith("separate_")
                    else 0
                    if arm in ("pca4", "integrative_gmm")
                    else 1
                )
                if arm in NATIVE or arm == "pca4":
                    row["standardized_mse_by_view"] = read_json(folder / "summary.json")[
                        "per_view_standardized_mse"
                    ]
                if record["draws_requested"]:
                    records = read_json(folder / "subsamples.json")
                    support = arrays(folder / "stability.npz")
                    done = [r for r in records if r["status"] == "complete"]
                    reference_labels = labels.copy()
                    noise = reference_labels < 0
                    reference_labels[noise] = (
                        np.arange(sum(noise)) + max(int(reference_labels.max()), 0) + 1
                    )
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
                            "own_point_ari": [
                                recovery(
                                    np.array(r["labels"]), reference_labels[r["training_indices"]]
                                )["ari"]
                                for r in done
                            ],
                            "included_truth_ari": [
                                recovery(np.array(r["labels"]), truth[r["training_indices"]])["ari"]
                                for r in done
                            ]
                            if truth is not None
                            else None,
                            "unit_coverage": float(np.mean(support["unit_successful"] > 0)),
                            "pair_coverage": float(np.mean(support["pair_successful"][upper] > 0)),
                            "mean_coassignment": float(np.nanmean(support["coassignment"][upper]))
                            if done
                            else None,
                            "denominator": (
                                "successful unit/pair inclusions; three repeated subsamples of one "
                                "dataset"
                            ),
                        }
                    )
            rows.append(row)
    return rows, stability


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
    comparisons = [
        {
            "reference": f"joint2_{s}",
            "alternative": f"{family}_{s}",
            "labels": {
                "neural_seed": s,
                "family": family,
                "reference_arm": f"joint2_{s}",
                "comparison_arm": f"{family}_{s}",
            },
            "positive_key": "comparison_only",
            "negative_key": "reference_only",
        }
        for s in SEEDS
        for family in ("separate", "joint4")
    ]
    groups, primary, paired = summarize_rates(
        rows, arms=ARMS, joint=NATIVE, comparisons=comparisons, primary_cases=PRIMARY
    )
    require(
        config.fixture or (len(primary), len(paired)) == (108, 72), "Frozen primary endpoint grid"
    )
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
            "Prespecified workflow comparison with five independent datasets/cell. "
            "Two encoders change capacity and compute; dimension controls do not "
            "isolate every cause. Seeds and subsamples are repeated measurements. No "
            "winning model, calibrated control rate or species guarantee."
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
