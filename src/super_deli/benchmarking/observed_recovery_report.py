"""Truth-after-fitting recovery summaries; no model fitting or setting selection."""

from pathlib import Path

from super_deli.benchmarking.observed_audit import change, reference, reference_update
from super_deli.benchmarking.observed_recovery_audit import arrays, file_hash
from super_deli.io.config import read_json, write_json


def interval(count, total):
    from scipy.stats import binomtest

    if not total:
        return None
    ci = binomtest(count, total).proportion_ci(method="exact")
    return [float(ci.low), float(ci.high)]


def assess(folder, data, origin, budget):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    summary = read_json(folder / "summary.json")
    result = dict(
        origin=origin,
        budget=budget,
        key=summary["task"]["key"],
        case=summary["task"]["case"],
        n=summary["original_n"],
        mask=summary["mask"],
        arm=summary["arm"],
        status=summary["status"],
        selected_k=summary["selected_k"],
        ari=summary.get("ari"),
        retention=len(summary["source_indices"]) / summary["original_n"],
        mean_log_density=summary.get("mean_log_density"),
        continuation_attempts=summary.get("continuation_attempts", 0),
        candidates=[],
        continued_starts=[],
    )
    if not (folder / "fit.npz").exists():
        result["error"] = summary["error"]
        return result
    a = arrays(folder / "fit.npz")
    x = (a["raw"] - a["normalization_means"]) / a["scales"]
    truth = data["truth"][summary["source_indices"]]
    result["min_coobserved"] = int(a["coobserved"].min())
    selected = next((c for c in summary["candidates"] if c["k"] == summary["selected_k"]), None)
    mask = data[f"eval_mask_{summary['mask']}"]
    observed = np.where(mask, np.nan, data["evaluation"])
    keep = ~np.isnan(observed).all(axis=1)
    evaluation = (observed[keep] - a["normalization_means"]) / a["scales"]
    for c in summary["candidates"]:
        row = {name: c[name] for name in ("k", "status")}
        row.update(
            converged_starts=sum(s["status"] == "converged" for s in c["starts"]),
            iteration_limited_starts=sum(
                s.get("error") == "Iteration limit reached" for s in c["starts"]
            ),
            max_iterations=max((s["iterations"] for s in c["starts"]), default=0),
        )
        if "bic" in c:
            row.update(
                {
                    name: c[name]
                    for name in (
                        "bic",
                        "log_likelihood",
                        "parameter_count",
                        "hard_cluster_sizes",
                        "active_bound_eigenvalues",
                    )
                }
            )
            row["ari"] = float(
                adjusted_rand_score(truth, a[f"k{c['k']}_membership"].argmax(axis=1))
            )
            p = tuple(a[f"k{c['k']}_{name}"] for name in ("weights", "means", "covariances"))
            density = reference(evaluation, *p)[0]
            density -= np.where(np.isnan(observed[keep]), 0, np.log(a["scales"])).sum(axis=1)
            row["external_mean_log_density"] = float(density.mean())
            eigen = np.asarray(c["eigenvalues"])
            row["max_condition_number"] = float((eigen[:, -1] / eigen[:, 0]).max())
            if selected is not None:
                row["bic_minus_selected"] = c["bic"] - selected["bic"]
                row["twice_ll_gain_vs_selected"] = 2 * (
                    c["log_likelihood"] - selected["log_likelihood"]
                )
                row["extra_bic_penalty"] = (
                    c["parameter_count"] - selected["parameter_count"]
                ) * float(np.log(len(x)))
        result["candidates"].append(row)
        for start in c["starts"]:
            if "original_iterations" not in start:
                continue
            prefix = f"k{c['k']}_s{start['start']}_"
            p = tuple(a[prefix + "final_" + name] for name in ("weights", "means", "covariances"))
            result["continued_starts"].append(
                dict(
                    k=c["k"],
                    start=start["start"],
                    status=start["status"],
                    iterations=start["iterations"],
                    error=start["error"],
                    independent_final_residual=float(change(p, reference_update(x, p, 1e-4))),
                    original_ll=float(a[prefix + "trace"][2000]),
                    final_ll=start["log_likelihood"],
                )
            )
    return result


def cells(rows):
    import numpy as np

    result = []
    keys = sorted(
        {(r["origin"], r["budget"], r["case"], r["n"], r["mask"], r["arm"]) for r in rows}
    )
    for key in keys:
        group = [
            r
            for r in rows
            if tuple(r[name] for name in ("origin", "budget", "case", "n", "mask", "arm")) == key
        ]
        success = [r for r in group if r["status"] == "complete"]
        rates = {}
        for name, event in (
            ("correct_k", lambda r: r["selected_k"] == (4 if r["case"] == "four_groups" else 1)),
            ("k_gt_1", lambda r: r["selected_k"] > 1),
        ):
            count = sum(event(r) for r in success)
            rates[name] = dict(
                count=count,
                successful=len(success),
                requested=len(group),
                conditional_exact95=interval(count, len(success)),
                failure_bounds=[
                    count / len(group),
                    (count + len(group) - len(success)) / len(group),
                ],
            )
        k4 = [next(c for c in r["candidates"] if c["k"] == 4) for r in group if r["candidates"]]

        def median(values):
            return float(np.median(values)) if values else None

        result.append(
            dict(zip(("origin", "budget", "case", "n", "mask", "arm"), key, strict=True))
            | dict(
                rates=rates,
                selected_k_counts={
                    str(k): sum(r["selected_k"] == k for r in success) for k in range(1, 5)
                },
                failures=len(group) - len(success),
                k4_eligible=sum(c["status"] == "eligible" for c in k4),
                median_selected_ari=median([r["ari"] for r in success]),
                median_k4_ari=median([c["ari"] for c in k4 if "ari" in c]),
                median_eligible_k4_ari=median([c["ari"] for c in k4 if c["status"] == "eligible"]),
                median_k4_bic_minus_selected=median(
                    [c["bic_minus_selected"] for c in k4 if "bic_minus_selected" in c]
                ),
                median_k4_twice_ll_gain=median(
                    [c["twice_ll_gain_vs_selected"] for c in k4 if "twice_ll_gain_vs_selected" in c]
                ),
                median_k4_extra_penalty=median(
                    [c["extra_bic_penalty"] for c in k4 if "extra_bic_penalty" in c]
                ),
                continued_starts=sum(r["continuation_attempts"] for r in group),
                median_external_density=median([r["mean_log_density"] for r in success]),
            )
        )
    return result


def oracle_assessment(source, oracle, data, origin):
    from sklearn.metrics import adjusted_rand_score

    base, fitted = read_json(source / "summary.json"), read_json(oracle / "summary.json")
    a = arrays(oracle / "fit.npz")
    c4 = next(c for c in base["candidates"] if c["k"] == 4)
    result = dict(
        origin=origin,
        key=base["task"]["key"],
        n=base["original_n"],
        mask=base["mask"],
        arm=base["arm"],
        status=fitted["status"],
        eligible=fitted.get("eligible", False),
        iterations=fitted["iterations"],
        excluded_from_selection=True,
        log_likelihood=fitted["log_likelihood"],
        baseline_k4_status=c4["status"],
    )
    if "log_likelihood" in c4:
        result["ll_gain_vs_baseline_k4"] = fitted["log_likelihood"] - c4["log_likelihood"]
    if "membership" in a:
        result["ari"] = float(
            adjusted_rand_score(
                data["truth"][base["source_indices"]], a["membership"].argmax(axis=1)
            )
        )
    return result


def build_report(root):
    root = Path(root)
    assert read_json(root / "archive-audit.json")["status"] == "passed"
    assert read_json(root / "main-audit.json")["status"] == "passed"
    frozen = read_json(root / "frozen-before-generation.json")
    archive = Path(frozen["archive"])
    rows, oracles = [], []
    for folder in sorted((archive / "main").iterdir()):
        data = arrays(folder / "inputs.npz")
        for summary in sorted(folder.glob("*/summary.json")):
            if not summary.parent.name.startswith("replay_"):
                rows.append(assess(summary.parent, data, "archived", "original"))
    for folder in sorted((root / "main").iterdir()):
        data = arrays(folder / "inputs.npz")
        for condition in sorted(p for p in folder.iterdir() if p.is_dir()):
            for budget, name in (("original", "baseline"), ("extended", "extended")):
                if (condition / name).exists():
                    rows.append(assess(condition / name, data, "fresh", budget))
            if (condition / "oracle").exists():
                oracles.append(
                    oracle_assessment(condition / "baseline", condition / "oracle", data, "fresh")
                )
    for folder in sorted((root / "archived-continuations").iterdir()):
        key, _ = folder.name.split("__", 1)
        rows.append(
            assess(folder, arrays(archive / "main" / key / "inputs.npz"), "archived", "extended")
        )
    for folder in sorted((root / "archived-oracles").iterdir()):
        key, condition = folder.name.split("__", 1)
        oracles.append(
            oracle_assessment(
                archive / "main" / key / condition,
                folder,
                arrays(archive / "main" / key / "inputs.npz"),
                "archived",
            )
        )
    assert len(rows) == 1683 and len(oracles) == 160
    paired = []
    for new in [r for r in rows if r["budget"] == "extended"]:
        old = next(
            r
            for r in rows
            if r["budget"] == "original"
            and all(r[k] == new[k] for k in ("origin", "key", "mask", "arm"))
        )
        paired.append(
            dict(
                origin=new["origin"],
                key=new["key"],
                mask=new["mask"],
                arm=new["arm"],
                original_status=old["status"],
                extended_status=new["status"],
                original_k=old["selected_k"],
                extended_k=new["selected_k"],
                continuation_attempts=new["continuation_attempts"],
            )
        )
    write_json(
        dict(points=rows, oracle_diagnostics=oracles, budget_pairs=paired),
        root / "detailed-assessments.json",
    )
    result = dict(
        schema_version=1,
        status="independently_audited",
        fresh_datasets=60,
        baseline_searches=252,
        archived_searches_audited=1224,
        preflight_searches=30,
        oracle_starts=160,
        archive_audit={
            k: v for k, v in read_json(root / "archive-audit.json").items() if k != "observations"
        },
        main_audit={
            k: v
            for k, v in read_json(root / "main-audit.json").items()
            if k not in ("observations", "archived_actions")
        },
        cells=cells(rows),
        paired_budget_counts=dict(
            pairs=len(paired),
            changed_k=sum(p["original_k"] != p["extended_k"] for p in paired),
            resolved=sum(
                p["original_status"] != "complete" and p["extended_status"] == "complete"
                for p in paired
            ),
        ),
        oracle_diagnostics=oracles,
        limitations=[
            "Pointwise intervals only; ten independent datasets per cell",
            "Masks, budgets, starts and paired references are repeated measures",
            "Known-parameter starts are diagnostic, not guaranteed global optima",
            "No default, p-value, model family or information-guard change",
            "No species or general performance guarantee",
        ],
    )
    write_json(result, root / "aggregate.json")
    # Full content hashes verify the original study was never rewritten.
    for name, identity in frozen["archive_manifest"].items():
        p = archive / name
        assert p.stat().st_size == identity["bytes"] and file_hash(p) == identity["sha256"]
    write_json(
        dict(
            status="passed",
            original_files=len(frozen["archive_manifest"]),
            original_content_unchanged=True,
        ),
        root / "archive-preservation.json",
    )
    return {k: v for k, v in result.items() if k not in ("cells", "oracle_diagnostics")}
