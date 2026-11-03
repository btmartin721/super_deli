"""Dataset-level descriptive summaries from reused evidence, never model choice."""

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import beta

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.view_signal_config import REPRESENTATIONS
from super_deli.benchmarking.view_signal_study import check_frozen, selected
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def interval(success, n):
    return [
        float(beta.ppf(0.025, success, n - success + 1)) if success else 0.0,
        float(beta.ppf(0.975, success + 1, n - success)) if success < n else 1.0,
    ]


def spread(values):
    values = [v for v in values if v is not None]
    return {
        "available": len(values),
        "median": float(np.median(values)) if values else None,
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
    }


def report(root):
    root = Path(root)
    frozen, _ = check_frozen(root)
    audit = read_json(root / "main-audit.json")
    require(
        audit["passed"] and audit["freeze_sha256"] == fingerprint_file(root / "frozen.json").sha256,
        "Bound final audit required",
    )
    binding = read_json(root / "archive-binding.json")
    entries = selected(binding, "main")
    groups, decoder, gmm = [], [], []
    cells = sorted({(e["task"]["case"], e["task"]["n"]) for e in entries})
    for case, n in cells:
        cell = [e for e in entries if (e["task"]["case"], e["task"]["n"]) == (case, n)]
        for name in REPRESENTATIONS:
            records = [
                read_json(root / "main" / Path(e["path"]).name / name / "result.json") for e in cell
            ]
            available = [r for r in records if r["status"] == "complete"]
            good = sum(
                r["recovery"]["selected_k"] == 4 and r["recovery"]["ari"] >= 0.8 for r in available
            )
            hierarchy = [r for r in available if r["hierarchy_status"] == "available"]
            row = {
                "case": case,
                "n": n,
                "representation": name,
                "requested": len(cell),
                "available": len(available),
                "failures": len(cell) - len(available),
                "selected_k": dict(
                    sorted(Counter(str(r["recovery"]["selected_k"]) for r in available).items())
                ),
                "k4_ari_ge_0_8": good,
                "all_request_rate": good / len(cell),
                "available_only_rate": good / len(available) if available else None,
                "unadjusted_exact_95_interval": interval(good, len(cell)),
                "failure_bounds": [
                    good / len(cell),
                    (good + len(cell) - len(available)) / len(cell),
                ],
                "hierarchies_available": len(hierarchy),
                "four_branches_at_0_8": sum(
                    r["branch_reference"]["four_disjoint_at_0_8"] for r in hierarchy
                ),
                "root_selected": sum(r["selection"]["root_selected"] for r in hierarchy),
            }
            for metric in ("ari", "assigned_ari", "coverage"):
                row[metric] = spread([r["recovery"][metric] for r in available])
            for metric in ("scatter_joint", "scatter_axis_m", "scatter_axis_e"):
                row[metric] = spread([r["geometry"][metric] for r in available])
            for view in ("morphology", "ecology"):
                for axis in (0, 1):
                    row[f"linear_{view}_axis{axis}"] = spread(
                        [r["geometry"]["linear_" + view]["r2"][axis] for r in available]
                    )
            for k in (5, 15):
                for metric in ("purity", "morphology_overlap", "available_rows", "tied_rows"):
                    row[f"neighbors_{k}_{metric}"] = spread(
                        [r["geometry"][f"neighbors_{k}"][metric] for r in available]
                    )
            groups.append(row)
        for arm in (
            "paired_0",
            "paired_101",
            "paired_202",
            "morphology_0",
            "morphology_101",
            "morphology_202",
            "ecology_0",
            "ecology_101",
            "ecology_202",
        ):
            values = [
                read_json(root / "main" / Path(e["path"]).name / "decoder.json").get(arm)
                for e in cell
            ]
            values = [v for v in values if v is not None]
            if values:
                decoder.append(
                    {
                        "case": case,
                        "n": n,
                        "arm": arm,
                        "available": len(values),
                        "mse_by_view": [
                            spread([v["standardized_mse_by_view"][i] for v in values])
                            for i in range(len(values[0]["standardized_mse_by_view"]))
                        ],
                        "centroid_mse_by_view": [
                            spread(
                                [
                                    float(np.mean(np.array(v["centroid_mse_by_group_view"])[:, i]))
                                    for v in values
                                ]
                            )
                            for i in range(len(values[0]["standardized_mse_by_view"]))
                        ],
                    }
                )
        records = [
            r for e in cell for r in read_json(root / f"main-gmm-{Path(e['path']).name}.json")
        ]
        for arm in sorted({r["arm"] for r in records}):
            rows = [r for r in records if r["arm"] == arm]
            for k in range(1, 5):
                candidates = [r for r in rows if r["k"] == k]
                gmm.append(
                    {
                        "case": case,
                        "n": n,
                        "arm": arm,
                        "k": k,
                        "available": len(candidates),
                        "selected": sum(r["selected"] for r in candidates),
                        "eligible": sum(r["status"] == "eligible" for r in candidates),
                        "bic_improvement_over_one": spread(
                            [r["bic_improvement_over_one"] for r in candidates]
                        ),
                        "log_likelihood": spread([r["log_likelihood"] for r in candidates]),
                        "penalty": spread([r["penalty"] for r in candidates]),
                    }
                )
    summary = {
        "schema_version": 1,
        "observations": len(entries),
        "cells": len(cells),
        "representations": len(REPRESENTATIONS),
        "main_fits": frozen["counts"]["main"],
        "preflight_fits": frozen["counts"]["preflight"],
        "main_audit": audit,
        "new_neural_training": 0,
        "gmm_refits": 0,
        "new_scientific_datasets": 0,
        "new_resampling_or_significance_draws": 0,
        "defaults_changed": False,
        "historical_pvalues_changed": False,
        "interpretation": (
            "Exploratory archival diagnosis. Seeds/stages are repeated measures; "
            "intervals are descriptive and unadjusted, "
            "not confirmatory comparisons or species support."
        ),
    }
    write_json(summary, root / "summary.json")
    write_json(groups, root / "all-spaces.json")
    write_json(decoder, root / "decoder-summary.json")
    write_json(gmm, root / "archived-gmm-summary.json")
    return summary
