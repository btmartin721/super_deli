"""All-request sensitivity summaries shared by frozen neural comparison designs."""

from collections import Counter

from super_deli.benchmarking.gaussian_graph_report import difference_interval
from super_deli.benchmarking.unknown_k_report import interval
from super_deli.benchmarking.view_signal_report import spread


def summarize_rates(rows, *, arms, joint, comparisons, primary_cases=None):
    groups, primary, paired = [], [], []
    for case, n in sorted({(r["case"], r["n"]) for r in rows}):
        cell = [r for r in rows if (r["case"], r["n"]) == (case, n)]
        is_primary = primary_cases is None or case in primary_cases
        for arm in arms:
            selected = [r for r in cell if r["arm"] == arm]
            done = [r for r in selected if r["status"] == "available"]
            size, unknown = len(selected), len(selected) - len(done)
            count = sum(r["endpoint"] for r in selected)
            endpoint = selected[0].get(
                "endpoint_name",
                "nontrivial_return" if case.startswith("gaussian_") else "k4_full_ari_ge_0_8",
            )
            group = {
                "case": case,
                "n": n,
                "arm": arm,
                "requested": size,
                "available": len(done),
                "failures": unknown,
                "endpoint_count": count if endpoint else None,
                "endpoint_rate_all_requests": count / size if endpoint else None,
                "endpoint": endpoint,
                "available_only_rate": count / len(done) if done and endpoint else None,
                "failure_bounds": [count / size, (count + unknown) / size] if endpoint else None,
                "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in done)),
            }
            for metric in ("coverage", "ari", "assigned_ari"):
                group[metric] = spread([r[metric] for r in done])
            for view in ("morphology", "ecology"):
                group["linear_" + view] = [
                    spread([r["linear_" + view]["r2"][i] for r in done]) for i in (0, 1)
                ]
            if arm in joint or any("standardized_mse_by_view" in r for r in done):
                group["standardized_mse_by_view"] = [
                    spread(
                        [
                            r["standardized_mse_by_view"][i]
                            for r in done
                            if "standardized_mse_by_view" in r
                        ]
                    )
                    for i in (0, 1)
                ]
            for metric in (
                "morphology_reference_ari",
                "ecology_reference_ari",
                "effective_rank",
                "neural_parameters",
                "point_neural_trainings",
                "workflow_seconds",
            ):
                if any(metric in r for r in selected):
                    group[metric] = spread([r.get(metric) for r in done])
            if arm in joint and is_primary:
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
        if not is_primary:
            continue
        for comparison in comparisons:
            equal = {r["key"]: r for r in cell if r["arm"] == comparison["reference"]}
            alternative = {r["key"]: r for r in cell if r["arm"] == comparison["alternative"]}
            common = [
                k for k in equal if equal[k]["status"] == alternative[k]["status"] == "available"
            ]
            positive = sum(alternative[k]["endpoint"] and not equal[k]["endpoint"] for k in common)
            negative = sum(equal[k]["endpoint"] and not alternative[k]["endpoint"] for k in common)
            size, unknown = len(equal), len(equal) - len(common)
            paired.append(
                {
                    "case": case,
                    "n": n,
                    **comparison["labels"],
                    "requested": size,
                    "common_available": len(common),
                    "unknown_pairs": unknown,
                    comparison.get("positive_key", "weighted_only"): positive,
                    comparison.get("negative_key", "equal_only"): negative,
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
