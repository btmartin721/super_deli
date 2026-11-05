"""Retrospective all-seed comparisons with separate mechanism and model-selection evidence."""

from collections import Counter
from pathlib import Path

import numpy as np

from super_deli.benchmarking.coordinate_config import (
    CLUSTERERS,
    PRIMARY,
    PRIMARY_SPACES,
    SEEDS,
    SPACES,
    tasks,
)
from super_deli.benchmarking.coordinate_study import check_frozen
from super_deli.benchmarking.matched_study import inventory
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_fit import recovery
from super_deli.benchmarking.neural_summary import summarize_rates
from super_deli.benchmarking.view_signal_report import spread
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def measure(labels, reference, case):
    if "truth" not in reference:
        return dict(
            ari=None,
            assigned_ari=None,
            coverage=float(np.mean(labels >= 0)),
            selected_k=len(set(labels) - {-1}),
            endpoint=False,
            endpoint_name=None,
            **{
                role + "_reference_ari": recovery(labels, reference[role + "_truth"])["ari"]
                for role in ("morphology", "ecology")
            },
        )
    result = recovery(labels, reference["truth"])
    gaussian = case.startswith("gaussian_")
    result.update(
        endpoint=result["selected_k"] > 1
        if gaussian
        else result["selected_k"] == len(set(reference["truth"])) and result["ari"] >= 0.8,
        endpoint_name="nontrivial_return"
        if gaussian
        else "k2_full_ari_ge_0_8"
        if case == "moons"
        else "k4_full_ari_ge_0_8",
    )
    return result


def summarize_mechanisms(rows, kind):
    result = []
    keys = (
        ("case", "n", "space", "policy")
        if kind == "policy"
        else ("case", "n", "space", "clusterer")
    )
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected = [r for r in rows if tuple(r[k] for k in keys) == key]
        entry = {**dict(zip(keys, key, strict=True)), "requested": len(selected)}
        if kind == "policy":
            entry.update(
                selected_k_counts=dict(Counter(str(r["selected_k"]) for r in selected)),
                coverage=spread([r["coverage"] for r in selected]),
                ari=spread([r.get("ari") for r in selected]),
                endpoint_count=sum(r["endpoint"] for r in selected)
                if selected[0]["endpoint_name"]
                else None,
            )
        for metric in (
            "root_margin",
            "root_selected",
            "surviving_branches",
            "cross_neighbors_5",
            "cross_neighbors_15",
            "euclidean_separation_ratio",
            "mutual_separation_ratio",
            "minimum_best_branch_jaccard",
            "k4_candidate_ari",
            "k4_eligible",
            "k4_vs1_likelihood_gain",
            "k4_vs1_bic_penalty",
            "k4_vs1_bic_improvement",
        ):
            if any(metric in r for r in selected):
                entry[metric] = spread([r.get(metric) for r in selected])
        result.append(entry)
    return result


def assess(root, source, config):
    rows = []
    policies = []
    for task in tasks(config, "main"):
        folder = root / "main" / task["key"]
        ref = arrays(Path(source) / "main" / task["key"] / "generating-reference.npz")
        for fit in read_json(folder / "fits.json"):
            space, clusterer = fit["space"], fit["clusterer"]
            geometry = read_json(folder / space / "geometry.json")
            row = {
                **task,
                "arm": fit["arm"],
                "space": space,
                "clusterer": clusterer,
                "status": fit["status"],
                "selected_k": None,
                "endpoint": False,
                "endpoint_name": None
                if task["case"] == "discordant"
                else "nontrivial_return"
                if task["case"].startswith("gaussian_")
                else "k2_full_ari_ge_0_8"
                if task["case"] == "moons"
                else "k4_full_ari_ge_0_8",
                "coverage": 0.0,
                "ari": None,
                "assigned_ari": None,
                "error": fit.get("error"),
                "effective_rank": geometry["effective_rank"],
                "workflow_seconds": fit["seconds"],
                **{k: geometry[k] for k in ("linear_morphology", "linear_ecology")},
            }
            a = arrays(folder / space / clusterer / "fit.npz")
            if fit["status"] == "available":
                row.update(measure(a["labels"], ref, task["case"]))
            if clusterer == "hdbscan":
                common = geometry["references"].get("common")
                for policy, trace in fit["policies"].items():
                    item = {
                        **task,
                        "space": space,
                        "policy": policy,
                        **measure(a[policy + "_labels"], ref, task["case"]),
                        "root_margin": trace["root_margin"],
                        "root_selected": float(trace["root_selected"]),
                        "surviving_branches": trace["surviving_cluster_nodes"],
                    }
                    if common:
                        graph = common["graph"]
                        branch = common["branches"]["best_branch_jaccard"]
                        item.update(
                            cross_neighbors_5=graph["neighbor_impurity"]["5"],
                            cross_neighbors_15=graph["neighbor_impurity"]["15"],
                            euclidean_separation_ratio=graph.get("euclidean_separation_ratio"),
                            mutual_separation_ratio=graph.get("mutual_separation_ratio"),
                            minimum_best_branch_jaccard=min(branch) if branch else None,
                        )
                    policies.append(item)
            else:
                attempted = {
                    r["k"]: r for r in fit["candidate_attempts"] if r["status"] == "complete"
                }
                if 4 in attempted:
                    candidate = measure(a["k4_labels"], ref, task["case"])
                    row["k4_candidate_ari"] = candidate["ari"]
                    row["k4_eligible"] = float(np.bincount(a["k4_labels"], minlength=4).min() >= 5)
                    if 1 in attempted:
                        gain = 2 * (attempted[4]["log_likelihood"] - attempted[1]["log_likelihood"])
                        difference = attempted[1]["bic"] - attempted[4]["bic"]
                        row.update(
                            k4_vs1_likelihood_gain=gain,
                            k4_vs1_bic_penalty=gain - difference,
                            k4_vs1_bic_improvement=difference,
                        )
            rows.append(row)
    return rows, policies


def report(root, source):
    root = Path(root)
    frozen, config = check_frozen(root)
    audit = read_json(root / "main-audit.json")
    require(
        audit["passed"]
        and audit["freeze_sha256"] == fingerprint_file(root / "frozen.json").sha256
        and inventory(root / "main") == read_json(root / "main-inventory.json"),
        "Bound unchanged main audit",
    )
    rows, policies = assess(root, source, config)
    arms = tuple(s + "__" + c for s in SPACES for c in CLUSTERERS)
    primary = tuple(s + "__" + c for s in PRIMARY_SPACES for c in CLUSTERERS)
    comparisons = [
        dict(
            reference=f"separate_{seed}__hdbscan",
            alternative=f"separate_{seed}__{method}",
            labels=dict(
                neural_seed=seed,
                comparison=method,
                reference_arm=f"separate_{seed}__hdbscan",
                comparison_arm=f"separate_{seed}__{method}",
            ),
            positive_key="gmm_only",
            negative_key="density_only",
        )
        for seed in SEEDS
        for method in ("gmm_full", "gmm_shared")
    ]
    groups, rates, paired = summarize_rates(
        rows, arms=arms, joint=primary, comparisons=comparisons, primary_cases=PRIMARY
    )
    for group in groups:
        group.pop("standardized_mse_by_view", None)
    require(
        config.fixture or (len(groups), len(rates), len(paired)) == (480, 108, 72),
        "Frozen report grid",
    )
    phases = {}
    for phase in ("preflight", "main"):
        original = read_json(root / (phase + "-audit.json"))
        phases[phase] = {k: v for k, v in original.items() if k != "records"}
        for metric in ("candidate_densities_checked", "density_replays"):
            phases[phase][metric] = sum(r[metric] for r in original["records"])
    summary = dict(
        schema_version=1,
        fixture=config.fixture,
        observations=len(tasks(config, "main")),
        counts=frozen["counts"],
        audits=phases,
        primary_rates=len(rates),
        paired_comparisons=len(paired),
        failed_searches=sum(r["status"] != "available" for r in rows),
        neural_training=0,
        new_datasets=0,
        new_pvalues=0,
        defaults_changed=False,
        historical_pvalues_changed=False,
        interpretation=(
            "Retrospective diagnosis on previously examined observations; "
            "seeds/models/policies repeat the same data. No fresh confirmation, "
            "selected winning family, general size or species guarantee"
        ),
    )
    for name, value in (
        ("assessments", rows),
        ("policy-assessments", policies),
        ("all-groups", groups),
        ("primary-rates", rates),
        ("paired-comparisons", paired),
        ("hierarchy-groups", summarize_mechanisms(policies, "policy")),
        (
            "candidate-groups",
            summarize_mechanisms([r for r in rows if r["clusterer"] != "hdbscan"], "mixture"),
        ),
        ("summary", summary),
    ):
        write_json(value, root / (name + ".json"))
    return summary
