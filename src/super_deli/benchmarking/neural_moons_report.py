"""Small retrospective summaries; no seed, configuration or species selection."""

from pathlib import Path

from super_deli.benchmarking.neural_moons_config import POLICIES, REPRESENTATIONS
from super_deli.benchmarking.neural_moons_study import check_frozen
from super_deli.io.config import read_json, write_json


def report(root):
    import numpy as np

    from super_deli.benchmarking.neural_analysis import interval

    root = Path(root).resolve()
    _, config = check_frozen(root)
    audit = read_json(root / "main-audit.json")
    if not audit["passed"]:
        raise ValueError("Independent main audit required before reporting")
    binding = read_json(root / "archive-binding.json")
    rows, geometry = [], []
    for entry in binding["observations"]:
        base = {k: entry[k] for k in ("case", "n", "replicate")}
        folder = root / "main" / entry["key"]
        for space in REPRESENTATIONS:
            record = read_json(folder / space / "result.json")
            if record["status"] == "complete":
                geometry.append(
                    {
                        **base,
                        "representation": space,
                        "hierarchy_status": record["hierarchy_status"],
                        "graph": record["graph"],
                        "neighborhood": record["neighborhood"],
                        "branch_reference": record.get("branch_reference", {}),
                    }
                )
            for policy in POLICIES:
                result = record["policies"][policy]
                rows.append({**base, "representation": space, "policy": policy, **result})
    groups = []
    for case, n in sorted({(r["case"], r["n"]) for r in rows}):
        correct = {"gaussian": 1, "moons": 2, "four_spherical": 4}[case]
        for space in REPRESENTATIONS:
            for policy in POLICIES:
                cell = [
                    r
                    for r in rows
                    if r["case"] == case
                    and r["n"] == n
                    and r["representation"] == space
                    and r["policy"] == policy
                ]
                done = [r for r in cell if r["status"] == "complete"]
                count = sum(r["selected_k"] == correct for r in done)
                failed = len(cell) - len(done)
                group = {
                    "case": case,
                    "n": n,
                    "representation": space,
                    "policy": policy,
                    "requested": len(cell),
                    "successful": len(done),
                    "failed_or_unavailable": failed,
                    "correct_k": correct,
                    "correct_k_count": count,
                    "correct_k_rate_resolved": count / len(done) if done else None,
                    "failure_bounds": [count / len(cell), (count + failed) / len(cell)],
                    "descriptive_pointwise95_resolved": interval(count, len(done)),
                    "selected_k_counts": {
                        str(k): sum(r["selected_k"] == k for r in done)
                        for k in sorted({r["selected_k"] for r in done})
                    },
                }
                for metric in (
                    "ari",
                    "assigned_ari",
                    "coverage",
                    "noise_count",
                    "splits",
                    "merges",
                ):
                    values = [r[metric] for r in done if r[metric] is not None]
                    group["mean_" + metric] = float(np.mean(values)) if values else None
                    group["sd_" + metric] = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None
                    )
                selected = [r["selection"] for r in done if "selection" in r]
                group["root_selection_count"] = sum(r["root_selected"] for r in selected)
                groups.append(group)
    pairs = []
    for entry in binding["observations"]:
        cell = [r for r in rows if all(r[k] == entry[k] for k in ("case", "n", "replicate"))]
        for seed in (0, 101, 202):
            left = next(
                r
                for r in cell
                if r["representation"] == f"neural_{seed}_raw" and r["policy"] == "baseline"
            )
            right = next(
                r
                for r in cell
                if r["representation"] == f"neural_{seed}_standard" and r["policy"] == "baseline"
            )
            pairs.append(
                {
                    "case": entry["case"],
                    "n": entry["n"],
                    "replicate": entry["replicate"],
                    "seed": seed,
                    "status": "complete"
                    if left["status"] == right["status"] == "complete"
                    else "unavailable",
                    **(
                        {
                            "delta_ari": right["ari"] - left["ari"],
                            "delta_coverage": right["coverage"] - left["coverage"],
                            "raw_k": left["selected_k"],
                            "standard_k": right["selected_k"],
                        }
                        if left["status"] == right["status"] == "complete"
                        else {}
                    ),
                }
            )
    write_json(rows, root / "selection-assessments.json")
    write_json(geometry, root / "geometry-assessments.json")
    summary = {
        "schema_version": 1,
        "fixture": config.fixture,
        "observations": len(binding["observations"]),
        "attempted_density_fits": audit["attempted"],
        "selection_evaluations": audit["selection_evaluations"],
        "failed_density_fits": audit["failed"],
        "unavailable_hierarchies": audit["unavailable_hierarchies"],
        "groups": groups,
        "paired_latent_scaling": pairs,
        "interpretation": [
            "Retrospective diagnostic, not fresh performance confirmation",
            "Generating coordinates/labels are oracle evidence unavailable empirically",
            "Seeds and saved subsamples are repeated measurements",
            "Pointwise intervals are descriptive; no simultaneous or selection guarantee",
            "No neural retraining, new scientific datasets, p-values or species claim",
            "Metric/dimension changes do not isolate all noise/feature-construction effects",
        ],
    }
    write_json(summary, root / "aggregate.json")
    return {
        "observations": summary["observations"],
        "groups": len(groups),
        "failed_density_fits": audit["failed"],
        "unavailable_hierarchies": audit["unavailable_hierarchies"],
        "fixture": config.fixture,
    }
