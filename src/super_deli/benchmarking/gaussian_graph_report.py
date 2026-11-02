"""All-request feasibility rates; estimated Gaussian departure is not count support."""

from collections import Counter
from pathlib import Path

import numpy as np

from super_deli.benchmarking.gaussian_graph_config import ARMS, CASES, CONDITIONS, tasks
from super_deli.benchmarking.gaussian_graph_study import check_frozen, inventory
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_fit import recovery
from super_deli.benchmarking.unknown_k_report import interval
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def assessments(root, config):
    rows = []
    for task in tasks(config, "main"):
        folder = root / "main" / task["key"]
        source = arrays(folder / "generating-reference.npz")
        truth = source.get("truth")
        for condition in CONDITIONS:
            point = folder / condition / "point"
            fitted = arrays(point / "gaussian.npz") if (point / "gaussian.npz").exists() else {}
            for record in read_json(point / "fits.json"):
                arm = record["arm"]
                row = {
                    **record,
                    "key": task["key"],
                    "case": task["case"],
                    "n": task["n"],
                    "condition": condition,
                    "generating_k": len(set(truth)) if truth is not None else None,
                    "effective_rank": int(fitted["effective_rank"]) if fitted else None,
                    "returned_correct_k": False,
                    "returned_correct_k_ari08": False,
                    "coverage": 0.0,
                    "ari": None,
                    "assigned_ari": None,
                }
                if row["status"] == "available":
                    a = arrays(point / arm / ("mixture.npz" if arm == "gmm" else "point.npz"))
                    labels = a["labels"]
                    row["coverage"] = float(np.mean(labels >= 0))
                    row["noise_count"] = int(sum(labels < 0))
                    if truth is not None:
                        row.update(recovery(labels, truth))
                        require(row["selected_k"] == record["selected_k"], "Recovered count")
                        row["returned_correct_k"] = record["selected_k"] == row["generating_k"]
                        row["returned_correct_k_ari08"] = (
                            row["returned_correct_k"] and row["ari"] >= 0.8
                        )
                if row.get("p_reference") is not None:
                    lo, hi = interval(row["exceedances"], 99, 0.05)
                    row["conditional_tail_probability_interval"] = [lo, hi]
                    row["corrected_tail_rank_mc_interval"] = [
                        (1 + 99 * lo) / 100,
                        (1 + 99 * hi) / 100,
                    ]
                rows.append(row)
    return rows


def endpoint_names(case):
    if case in {"gaussian", "gaussian_8d"}:
        return ("nontrivial_return",)
    if case in {"four_spherical", "four_unequal", "moons"}:
        return ("returned_correct_k", "returned_correct_k_ari08")
    return ()


def success(row, endpoint):
    if endpoint == "nontrivial_return":
        return row["status"] == "available" and row["selected_k"] > 1
    return bool(row[endpoint])


def difference_interval(positive, negative, size, alpha):
    p, q = interval(positive, size, alpha / 2), interval(negative, size, alpha / 2)
    return [p[0] - q[1], p[1] - q[0]]


def report(root):
    root = Path(root)
    frozen, config = check_frozen(root)
    audit = read_json(root / "main-audit.json")
    require(
        inventory(root / "main") == read_json(root / "main-inventory.json"),
        "Audited inventory unchanged",
    )
    require(
        audit["passed"]
        and audit["execution_sha256"] == fingerprint_file(root / "main-execution.json").sha256,
        "Passed bound main audit",
    )
    rows = assessments(root, config)
    primary, groups, pairs = [], [], []
    sizes = sorted({t["n"] for t in tasks(config, "main")})
    for case in CASES:
        for n in sizes:
            for condition in CONDITIONS:
                cell = [
                    r for r in rows if (r["case"], r["n"], r["condition"]) == (case, n, condition)
                ]
                if not cell:
                    continue
                for arm in ARMS:
                    selected = [r for r in cell if r["arm"] == arm]
                    available = [r for r in selected if r["status"] == "available"]
                    size = len(selected)
                    unknown = size - len(available)
                    vals = [r["ari"] for r in available if r["ari"] is not None]
                    group = {
                        "case": case,
                        "n": n,
                        "condition": condition,
                        "arm": arm,
                        "requested": size,
                        "available": len(available),
                        "statuses": dict(Counter(r["status"] for r in selected)),
                        "reasons": dict(Counter(r.get("reason") or "available" for r in selected)),
                        "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in available)),
                        "screen_evaluable": sum(r.get("p_reference") is not None for r in selected),
                        "screen_passes": sum(r.get("screen_pass", False) for r in selected),
                        "effective_ranks": dict(
                            Counter(str(r["effective_rank"]) for r in selected)
                        ),
                        "mean_coverage_all_requests": float(
                            np.mean([r["coverage"] for r in selected])
                        ),
                        "available_mean_ari": float(np.mean(vals)) if vals else None,
                        "all_request_ari_bounds": [
                            (sum(vals) - unknown) / size,
                            (sum(vals) + unknown) / size,
                        ]
                        if selected[0]["generating_k"] is not None
                        else None,
                    }
                    groups.append(group)
                    for endpoint in endpoint_names(case):
                        total = sum(success(r, endpoint) for r in selected)
                        upper = total + unknown if endpoint == "nontrivial_return" else total
                        primary.append(
                            {
                                **{
                                    k: group[k]
                                    for k in ("case", "n", "condition", "arm", "requested")
                                },
                                "endpoint": endpoint,
                                "count": total,
                                "rate": total / size,
                                "unavailable": unknown,
                                "identification_bounds": [total / size, upper / size],
                                "marginal_95_interval": interval(total, size, 0.05),
                                "simultaneous_95_interval": interval(total, size, 0.05 / 192),
                                "simultaneous_failure_interval": [
                                    interval(total, size, 0.05 / 192)[0],
                                    interval(upper, size, 0.05 / 192)[1],
                                ],
                            }
                        )
                for neighbors in (5, 15):
                    a = {r["key"]: r for r in cell if r["arm"] == f"reference_{neighbors}"}
                    b = {r["key"]: r for r in cell if r["arm"] == f"spectral_{neighbors}"}
                    common = [
                        (a[k], b[k])
                        for k in a
                        if a[k]["status"] == b[k]["status"] == "available"
                        and a[k]["ari"] is not None
                    ]
                    for endpoint in endpoint_names(case):
                        positive = sum(
                            success(a[k], endpoint) and not success(b[k], endpoint) for k in a
                        )
                        negative = sum(
                            success(b[k], endpoint) and not success(a[k], endpoint) for k in a
                        )
                        size = len(a)

                        pairs.append(
                            {
                                "case": case,
                                "n": n,
                                "condition": condition,
                                "neighbors": neighbors,
                                "endpoint": endpoint,
                                "requested": size,
                                "reference_only": positive,
                                "legacy_only": negative,
                                "rate_difference": (positive - negative) / size,
                                "marginal_95_interval": difference_interval(
                                    positive, negative, size, 0.05
                                ),
                                "simultaneous_95_interval": difference_interval(
                                    positive, negative, size, 0.05 / 64
                                ),
                                "common_available": len(common),
                                "common_mean_ari_difference": float(
                                    np.mean([x["ari"] - y["ari"] for x, y in common])
                                )
                                if common
                                else None,
                            }
                        )
    if not config.fixture:
        require(
            len(rows) == 1920 and len(primary) == 192 and len(groups) == 192 and len(pairs) == 64,
            "Frozen report family accounting",
        )
    result = {
        "schema_version": 1,
        "fixture": config.fixture,
        "observations": len(tasks(config, "main")),
        "tables": 2 * len(tasks(config, "main")),
        "main_evaluations": frozen["counts"]["main"]["evaluations"],
        "point_outputs": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "groups": groups,
        "primary_rates": primary,
        "paired_comparisons": pairs,
        "primary_interval_family": 192,
        "paired_interval_family": 64,
        "interpretation": [
            "Feasibility only; ten datasets/cell, no general size/power guarantee",
            "Estimated Gaussian departure is not evidence that proposed K is correct",
            "No-supported-split K=1 does not establish one population or species",
            "Withheld results do not become K=1; paired maps are repeated measurements",
            "Conditional Monte Carlo intervals exclude null-fitting uncertainty",
            "No model, neighbor setting or default selected automatically",
        ],
    }
    write_json(rows, root / "point-assessments.json")
    write_json(result, root / "aggregate.json")
    return {
        "groups": len(groups),
        "primary_rates": len(primary),
        "point_outputs": len(rows),
        "fixture": config.fixture,
    }
