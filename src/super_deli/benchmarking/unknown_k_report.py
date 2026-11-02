"""All-request descriptive recovery, overpartitioning and exact rate intervals."""

from collections import Counter

import numpy as np
from scipy.stats import beta

from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_fit import recovery
from super_deli.benchmarking.unknown_k_config import ARMS, CASES, CONDITIONS, tasks
from super_deli.benchmarking.unknown_k_study import check_frozen
from super_deli.io.config import read_json, write_json


def interval(success, n, alpha):
    return [
        float(beta.ppf(alpha / 2, success, n - success + 1)) if success else 0.0,
        float(beta.ppf(1 - alpha / 2, success + 1, n - success)) if success < n else 1.0,
    ]


def report(root):
    f, config = check_frozen(root)
    require(read_json(root / "main-audit.json")["passed"], "Passed full audit required")
    rows = []
    for task in tasks(config, "main"):
        folder = root / "main" / task["key"]
        ref = arrays(folder / "generating-reference.npz")
        truth = ref.get("truth")
        for record in read_json(folder / "fits.json"):
            if record["kind"] != "point":
                continue
            arm, condition = record["arm"], record["condition"]
            out = folder / condition / arm
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
            if arm.startswith("spectral") and (out / "summary.json").exists():
                summary = read_json(out / "summary.json")
                row.update(
                    {
                        k: summary.get(k)
                        for k in ("proposed_k", "gap_lead", "components", "method", "cluster_sizes")
                    }
                )
            if row["status"] == "available":
                a = arrays(out / ("mixture.npz" if arm == "gmm" else "point.npz"))
                labels = a["labels"]
                row["coverage"] = float(np.mean(labels >= 0))
                row["noise_count"] = int(sum(labels < 0))
                row["cluster_sizes"] = [int(sum(labels == k)) for k in sorted(set(labels) - {-1})]
                if truth is not None:
                    metrics = recovery(labels, truth)
                    require(metrics["selected_k"] == record["selected_k"], "Returned label count")
                    row.update(metrics)
                    row["returned_correct_k"] = record["selected_k"] == row["generating_k"]
                    row["returned_correct_k_ari08"] = (
                        row["returned_correct_k"] and row["ari"] >= 0.8
                    )
            rows.append(row)
    groups = []
    primary = []
    for case in CASES:
        for n in sorted({t["n"] for t in tasks(config, "main")}):
            for condition in CONDITIONS:
                for arm in ARMS:
                    selected = [
                        r
                        for r in rows
                        if (r["case"], r["n"], r["condition"], r["arm"])
                        == (case, n, condition, arm)
                    ]
                    if not selected:
                        continue
                    done = [r for r in selected if r["status"] == "available"]
                    size = len(selected)
                    unknown = size - len(done)
                    vals = [r["ari"] for r in done if r["ari"] is not None]
                    g = {
                        "case": case,
                        "n": n,
                        "condition": condition,
                        "arm": arm,
                        "requested": size,
                        "available": len(done),
                        "statuses": dict(Counter(r["status"] for r in selected)),
                        "reasons": dict(
                            Counter(
                                r.get("reason") or r.get("error", "available") for r in selected
                            )
                        ),
                        "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in done)),
                        "mean_coverage_all_requests": float(
                            np.mean([r["coverage"] for r in selected])
                        ),
                        "available_mean_ari": float(np.mean(vals)) if vals else None,
                        "available_sd_ari": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                        "all_request_ari_bounds": [
                            (sum(vals) - unknown) / size,
                            (sum(vals) + unknown) / size,
                        ]
                        if vals or selected[0]["generating_k"] is not None
                        else None,
                        "component_partitions": sum(
                            r.get("method") == "components" and r["status"] == "available"
                            for r in selected
                        ),
                        "mean_assigned_ari": float(
                            np.mean(
                                [r["assigned_ari"] for r in done if r["assigned_ari"] is not None]
                            )
                        )
                        if any(r["assigned_ari"] is not None for r in done)
                        else None,
                    }
                    groups.append(g)
                    endpoints = []
                    if case in {"gaussian", "gaussian_8d"}:
                        endpoints = [("nontrivial_return", sum(r["selected_k"] > 1 for r in done))]
                    elif case in {"four_spherical", "four_unequal", "moons"}:
                        endpoints = [
                            (name, sum(r[name] for r in selected))
                            for name in ("returned_correct_k", "returned_correct_k_ari08")
                        ]
                    for name, success in endpoints:
                        upper = success + unknown if name == "nontrivial_return" else success
                        primary.append(
                            {
                                **{k: g[k] for k in ("case", "n", "condition", "arm", "requested")},
                                "endpoint": name,
                                "count": success,
                                "rate": success / size,
                                "unavailable": unknown,
                                "identification_bounds": [success / size, upper / size],
                                "marginal_95_interval": interval(success, size, 0.05),
                                "simultaneous_95_interval": interval(success, size, 0.05 / 128),
                                "simultaneous_failure_interval": [
                                    interval(success, size, 0.05 / 128)[0],
                                    interval(upper, size, 0.05 / 128)[1],
                                ],
                            }
                        )
    if not config.fixture:
        require(len(primary) == 128 and len(groups) == 128, "Primary family freeze")
    pairs = []
    for case in CASES:
        for n in sorted({r["n"] for r in rows}):
            for condition in CONDITIONS:
                for graph in ("spectral_5", "spectral_15"):
                    for baseline in ("gmm", "density"):
                        a = [
                            r
                            for r in rows
                            if (r["case"], r["n"], r["condition"], r["arm"])
                            == (case, n, condition, graph)
                        ]
                        b = {
                            r["key"]: r
                            for r in rows
                            if (r["case"], r["n"], r["condition"], r["arm"])
                            == (case, n, condition, baseline)
                        }
                        if not a:
                            continue
                        common = [
                            (r, b[r["key"]])
                            for r in a
                            if r["status"] == b[r["key"]]["status"] == "available"
                            and r["ari"] is not None
                        ]
                        diff = [x["ari"] - y["ari"] for x, y in common]
                        pairs.append(
                            {
                                "case": case,
                                "n": n,
                                "condition": condition,
                                "graph": graph,
                                "baseline": baseline,
                                "requested": len(a),
                                "common_available": len(common),
                                "common_mean_ari_difference": float(np.mean(diff))
                                if diff
                                else None,
                                "common_sd_ari_difference": float(np.std(diff, ddof=1))
                                if len(diff) > 1
                                else None,
                            }
                        )
    result = {
        "schema_version": 1,
        "fixture": config.fixture,
        "observations": len(tasks(config, "main")),
        "main_evaluations": f["counts"]["main"]["evaluations"],
        "point_workflows": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "groups": groups,
        "primary_rates": primary,
        "paired_comparisons": pairs,
        "primary_interval_family": 128,
        "interpretation": [
            "Experimental point-only selection, not a species test",
            "K=1 is heuristic selection, not statistical acceptance of one population",
            "Abstentions/failures are not correct returned partitions",
            "Gaussian nontrivial returns describe these laws, without a universal error guarantee",
            "Paired settings are repeated measurements; available-only metrics are conditional",
            "Neither graph setting nor baseline is selected automatically",
        ],
    }
    write_json(rows, root / "point-assessments.json")
    write_json(result, root / "aggregate.json")
    return {
        "groups": len(groups),
        "primary_rates": len(primary),
        "point_workflows": len(rows),
        "fixture": config.fixture,
    }
