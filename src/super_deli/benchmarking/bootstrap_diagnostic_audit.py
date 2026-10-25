"""Independent no-refit validation of fitted maps, diagnostic gates, truth scores and provenance."""

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.bootstrap_diagnostic_metrics import workflow_folds
from super_deli.benchmarking.local_archives import materialize
from super_deli.benchmarking.predictive_audit import (
    independent_selection,
    independent_simulation,
    project_independent,
)
from super_deli.benchmarking.predictive_bootstrap_audit import (
    audit_draw,
    audit_geometry,
    audit_pool,
)
from super_deli.io.config import read_json, write_json
from super_deli.models.bootstrap_diagnostic_config import BootstrapDiagnosticConfig
from super_deli.models.predictive_study_config import PredictiveCell


def check_axis(matrix, labels, reported):
    import numpy as np

    if labels is None or not len(labels):
        if reported is not None:
            raise ValueError("Unavailable axis has a reported result")
        return
    x = matrix - matrix.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    basis = u[:, s > np.finfo(float).eps * max(x.shape) * max(float(s[0]), 1e-300)]
    for name, y in (("a", np.asarray(labels) // 2), ("b", np.asarray(labels) % 2)):
        y = y - y.mean()
        if not y @ y:
            if reported[name] is not None:
                raise ValueError("Degenerate contrast reported as estimable")
        else:
            projection = basis @ (basis.T @ y)
            np.testing.assert_allclose(
                reported[name], float(projection @ projection / (y @ y)), atol=1e-10
            )


def audit_gates(directory, indices, model, reported):
    import copy
    import math

    import numpy as np

    from super_deli.models.predictive_selection import screen_candidates

    directory = Path(directory)
    selection = read_json(directory / "summary.json")["selection"]
    if selection["status"] != "complete":
        if reported["gate"]["status"] != "unavailable":
            raise ValueError("Failure became available")
        return
    full = read_json(directory / "full/run/candidates.json")
    folds = workflow_folds(directory, indices)
    replay = screen_candidates(full, folds, n_units=len(indices), n_folds=5)
    if replay != selection:
        raise ValueError("Stored production selection differs")
    actual = reported["gate"]
    if actual["selected_k"] != selection["screened_k"]:
        raise ValueError("Diagnostic selected K differs")
    for c, row in zip(full, actual["candidates"], strict=True):
        k = c["k"]
        other = [next(x for x in f["candidates"] if x["k"] == k) for f in folds]
        if row["full_status"] != c["status"] or row["fold_statuses"] != [
            x["status"] for x in other
        ]:
            raise ValueError("Constraint status differs")
        expected = [len(set(map(int, indices)))] + [
            len(set(f["original_training_indices"])) for f in folds
        ]
        if expected != row["full_and_fold_distinct_counts"]:
            raise ValueError("Independent-unit denominators differ")
        if row["full_hard_sizes"] != c.get("hard_cluster_sizes") or row[
            "full_distinct_sizes"
        ] != c.get("distinct_unit_sizes"):
            raise ValueError("Size diagnostic differs")
        complete = all(x["status"] != "insufficient_samples" for x in [c, *other])
        if complete:
            total = []
            unweighted = []
            for f in folds:
                with np.load(
                    directory / f"fold_{f['fold']:03d}" / "validation.npz", allow_pickle=False
                ) as p:
                    difference = p[f"k{k}"] - p["k1"]
                ids = np.asarray(f["original_validation_indices"])
                total.extend(difference.tolist())
                for unit in np.unique(ids):
                    np.testing.assert_allclose(
                        difference[ids == unit], difference[ids == unit][0], atol=1e-12
                    )
                    unweighted.append(float(difference[ids == unit].mean()))
            gain = math.fsum(total) / len(total)
            equal = math.fsum(unweighted) / len(unweighted)
            np.testing.assert_allclose(
                [row["pooled_validation_gain"], row["equal_original_validation_gain"]],
                [gain, equal],
                atol=1e-12,
            )
    for policy, result in actual["counterfactuals"].items():
        full_copy, fold_copy = copy.deepcopy(full), copy.deepcopy(folds)
        if result["status"] == "unavailable":
            if not result.get("missing_candidates"):
                raise ValueError("Unavailable counterfactual lacks cause")
            continue
        for c in full_copy:
            k = c["k"]
            if policy == "remove_full_size" and c["status"] == "small_component":
                c["status"] = "eligible"
            if policy == "occurrence_sizes" and c["status"] != "insufficient_samples":
                c["status"] = (
                    "eligible"
                    if min(c["hard_cluster_sizes"]) >= model.min_cluster_size
                    else "small_component"
                )
            for f in fold_copy:
                candidate = next(x for x in f["candidates"] if x["k"] == k)
                if policy == "remove_fold_size" and candidate["status"] == "small_component":
                    candidate["status"] = "eligible"
                if policy == "occurrence_sizes" and candidate["status"] != "insufficient_samples":
                    candidate["status"] = (
                        "eligible"
                        if min(candidate["hard_cluster_sizes"]) >= model.min_cluster_size
                        else "small_component"
                    )
                if policy == "remove_gain" and str(k) in f.get("log_densities", {}):
                    f["log_densities"][str(k)] = [x + (k != 1) for x in f["log_densities"]["1"]]
        if policy == "equal_validation_weights":
            allow = []
            for c, row in zip(full, actual["candidates"], strict=True):
                if (
                    c["status"] == "eligible"
                    and all(s == "eligible" for s in row["fold_statuses"])
                    and (c["k"] == 1 or row["equal_original_validation_gain"] > 0)
                ):
                    allow.append(c)
            chosen = sorted(allow, key=lambda c: (c["aic"], c["k"]))[0]["k"]
        else:
            chosen = screen_candidates(full_copy, fold_copy, n_units=len(indices), n_folds=5)[
                "screened_k"
            ]
        if chosen != result["selected_k"]:
            raise ValueError("Counterfactual reconstruction differs")


def audit_metrics(directory, values, layout, indices, model, truth, reported):
    import numpy as np
    from scipy.special import entr

    from super_deli.benchmarking.predictive_density_audit import log_density
    from super_deli.benchmarking.recovery_analysis import _ari

    indices = np.asarray(indices, dtype=int)
    directory = Path(directory)
    audit_gates(directory, indices, model, reported)
    folds = workflow_folds(directory, indices)
    mapping = {
        "full": indices,
        **{
            f"fold_{f['fold']:03d}": np.asarray(f["original_training_indices"], dtype=int)
            for f in folds
        },
    }
    oob = np.asarray(sorted(set(range(len(layout["unit_ids"]))) - set(indices.tolist())), dtype=int)
    if reported["global_oob"] != oob.tolist():
        raise ValueError("OOB identities differ")
    for prefix, draw in mapping.items():
        run = directory / prefix / "run"
        if not (run / "gmm_candidates.npz").exists():
            continue
        actual = reported["searches"][prefix]
        if actual["n_occurrences"] != len(draw) or actual["n_distinct"] != len(set(draw.tolist())):
            raise ValueError("Search measure weights differ")
        if truth is not None:
            expected_truth = [len({int(u) for u in draw if truth[u] == k}) for k in range(4)]
            if actual["generating_distinct_sizes"] != expected_truth:
                raise ValueError("Generating-partition feasibility differs")
        if actual.get("status") == "unavailable":
            raise ValueError("Retained search metrics unavailable")
        y = None if truth is None else np.asarray(truth)[draw]
        raw_blocks = []
        for j, view in enumerate(layout["views"]):
            targets = np.asarray(view["target_ids"])
            means = np.array(
                [values[view["kind"]][targets == u].mean(0) for u in layout["unit_ids"]]
            )[draw]
            raw_blocks.append(means)
            v = actual["views"][j]
            check_axis(means, y, v["raw_mean_axis_r2"])
            with np.load(run / f"view_{j:03d}/parameters/pca.npz", allow_pickle=False) as p:
                standard = (means[:, p["kept_indices"]] - p["feature_means"]) / p["feature_scales"]
                check_axis(standard, y, v["standardized_axis_r2"])
                check_axis((standard - p["mean"]) @ p["components"].T, y, v["view_pca_axis_r2"])
                source_rows = [
                    r
                    for u in draw
                    for r, target in enumerate(targets)
                    if target == layout["unit_ids"][u]
                ]
                source_values = values[view["kind"]][source_rows][:, p["kept_indices"]]
                source_scores = (
                    (source_values - p["feature_means"]) / p["feature_scales"] - p["mean"]
                ) @ p["components"].T
                source_truth = (
                    None
                    if truth is None
                    else np.asarray(
                        [
                            truth[u]
                            for u in draw
                            for target in targets
                            if target == layout["unit_ids"][u]
                        ]
                    )
                )
                check_axis(source_scores, source_truth, v["source_projected_axis_r2"])
            with np.load(run / f"view_{j:03d}/parameters/profiles.npz", allow_pickle=False) as p:
                check_axis(p["profiles"], y, v["profile_axis_r2"])
        check_axis(np.column_stack(raw_blocks), y, actual["raw_joint_axis_r2"])
        check_axis(np.asarray(layout["counts"])[draw, None], y, actual["count_axis_r2"])
        with np.load(run / "fusion.npz", allow_pickle=False) as p:
            check_axis(p["fused"], y, actual["fused_axis_r2"])
        projected = project_independent(run, values, layout)
        check_axis(projected, truth, actual["projected_all_axis_r2"])
        check_axis(
            projected[oob],
            None if truth is None else np.asarray(truth)[oob],
            actual["projected_global_oob_axis_r2"],
        )
        with np.load(run / "gmm_candidates.npz", allow_pickle=False) as p:
            check_axis(p["scores"], y, actual["joint_axis_r2"])
            for c in actual["candidates"]:
                if c["status"] == "insufficient_samples":
                    continue
                k = c["k"]
                candidate = next(v for v in read_json(run / "candidates.json") if v["k"] == k)
                for field in ("status", "log_likelihood", "aic", "bic", "iterations", "converged"):
                    if c[field] != candidate[field]:
                        raise ValueError("Diagnostic candidate ledger differs")
                n, d = p["scores"].shape
                q = k - 1 + k * d + d * (d + 1) // 2
                if c["parameters"] != q:
                    raise ValueError("Diagnostic parameter count differs")
                one = next(v for v in read_json(run / "candidates.json") if v["k"] == 1)
                np.testing.assert_allclose(
                    c["twice_likelihood_gain_over_one"],
                    2 * (c["log_likelihood"] - one["log_likelihood"]),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    c["aic_penalty_difference_over_one"], 2 * (k - 1) * (d + 1), atol=1e-12
                )
                np.testing.assert_allclose(
                    c["bic_penalty_difference_over_one"], (k - 1) * (d + 1) * np.log(n), atol=1e-12
                )
                prob = p[f"k{k}_membership"]
                np.testing.assert_allclose(c["entropy"], entr(prob).sum() / len(prob), atol=1e-12)
                singular = np.linalg.svd(p[f"k{k}_covariances"], compute_uv=False)[:, ::-1]
                np.testing.assert_allclose(
                    c["covariance_eigenvalues"], singular, atol=1e-12, rtol=1e-10
                )
                if truth is not None:
                    np.testing.assert_allclose(
                        c["training_occurrence_ari"], _ari(y, p[f"k{k}_labels"]), atol=1e-12
                    )
                    weights, means, cov = [
                        p[f"k{k}_{v}"] for v in ("weights", "means", "covariances")
                    ]
                    terms = np.column_stack(
                        [
                            np.log(a) + log_density(projected, np.ones(1), m[None], v[None])
                            for a, m, v in zip(weights, means, cov, strict=True)
                        ]
                    )
                    labels = terms.argmax(1)
                    np.testing.assert_allclose(
                        c["all_original_ari"], _ari(np.asarray(truth), labels), atol=1e-12
                    )
                    if len(oob) > 1:
                        np.testing.assert_allclose(
                            c["global_oob_ari"],
                            _ari(np.asarray(truth)[oob], labels[oob]),
                            atol=1e-12,
                        )


def audit_point(directory, values, layout, model):
    import numpy as np

    from super_deli.workflows.predictive import fold_model

    directory = Path(directory)
    result = read_json(directory / "summary.json")
    if result["selection"]["status"] != "complete":
        return result["attempted_fits"]
    folds = workflow_folds(directory, list(range(len(layout["unit_ids"]))))
    checks = []
    audit_geometry(
        directory / "full/run", values, layout, np.arange(len(layout["unit_ids"])), model
    )
    full, _ = audit_pool(directory / "full/run", model, list(range(len(layout["unit_ids"]))))
    for f in folds:
        prefix = f"fold_{f['fold']:03d}"
        trained = fold_model(model, f["fold"])
        audit_geometry(
            directory / prefix / "run",
            values,
            layout,
            np.asarray(f["original_training_indices"]),
            trained,
        )
        scores = project_independent(directory / prefix / "run", values, layout)[
            f["original_validation_indices"]
        ]
        candidates, density = audit_pool(
            directory / prefix / "run", trained, f["original_training_indices"], scores
        )
        with np.load(directory / prefix / "validation.npz", allow_pickle=False) as p:
            np.testing.assert_allclose(p["scores"], scores, atol=1e-11, rtol=1e-11)
            for k, logs in density.items():
                np.testing.assert_allclose(p["k" + k], logs, atol=1e-8, rtol=1e-10)
        checks.append(dict(candidates=candidates, density=density))
    expected = independent_selection(full, checks, len(layout["unit_ids"]))
    for key in ("ordinary_aic_k", "screened_k", "descriptive_bic_k"):
        if result["selection"][key] != expected[key]:
            raise ValueError("Independent point selection differs")
    return 6


def audit_oracles(base, truth, model, assignments):
    import numpy as np
    from scipy.special import logsumexp

    from super_deli.benchmarking.predictive_density_audit import log_density
    from super_deli.workflows.predictive import fold_model

    rows = read_json(base / "oracles.json")
    for row in rows:
        directory = base / "oracles" / row["workflow"] / row["prefix"]
        if read_json(directory / "diagnostic.json") != {
            k: v for k, v in row.items() if k != "workflow"
        }:
            raise ValueError("Truth diagnostic ledger differs")
        if row["status"] != "complete":
            if not row.get("error"):
                raise ValueError("Unavailable truth diagnostic lacks reason")
            continue
        indices = np.asarray(row["original_indices"], dtype=int)
        labels = np.asarray(truth)[indices]
        trained = model if row["prefix"] == "full" else fold_model(model, int(row["prefix"][-3:]))
        with (
            np.load(
                base / "baseline" / row["workflow"] / row["prefix"] / "run/gmm_candidates.npz",
                allow_pickle=False,
            ) as baseline,
            np.load(directory / "mixture.npz", allow_pickle=False) as fit,
            np.load(directory / "initialization.npz", allow_pickle=False) as initial,
        ):
            scores = fit["scores"]
            np.testing.assert_array_equal(scores, baseline["scores"])
            means = np.array([scores[labels == k].mean(0) for k in range(4)])
            residual = scores - means[labels]
            cov = residual.T @ residual / len(scores) + trained.reg_covar * np.eye(scores.shape[1])
            np.testing.assert_allclose(
                initial["weights"], np.bincount(labels, minlength=4) / len(labels), atol=1e-14
            )
            np.testing.assert_allclose(initial["means"], means, atol=1e-13)
            np.testing.assert_allclose(
                initial["covariances"], np.repeat(cov[None], 4, axis=0), atol=1e-13
            )
            joint = np.column_stack(
                [
                    np.log(a) + log_density(scores, np.ones(1), m[None], v[None])
                    for a, m, v in zip(
                        fit["weights"], fit["means"], fit["covariances"], strict=True
                    )
                ]
            )
            logs = logsumexp(joint, axis=1)
            prob = np.exp(joint - logs[:, None])
            predicted = joint.argmax(1)
            np.testing.assert_allclose(fit["membership"], prob, atol=1e-10)
            np.testing.assert_array_equal(fit["labels"], predicted)
            sizes = [len(set(indices[predicted == k].tolist())) for k in range(4)]
            if row["distinct_unit_sizes"] != sizes or row["eligible_original_ids"] != (
                min(sizes) >= model.min_cluster_size
            ):
                raise ValueError("Original-ID oracle eligibility differs")
            q = 3 + 4 * scores.shape[1] + scores.shape[1] * (scores.shape[1] + 1) // 2
            np.testing.assert_allclose(
                [row["log_likelihood"], row["aic"], row["bic"]],
                [logs.sum(), -2 * logs.sum() + 2 * q, -2 * logs.sum() + q * np.log(len(scores))],
                atol=1e-7,
                rtol=1e-10,
            )
    return len(rows)


def _audit_observation(task):
    import numpy as np

    from super_deli.benchmarking.bootstrap_diagnostic_gates import unique_support

    source, raw, declaration, row, scratch = task
    c = BootstrapDiagnosticConfig.from_dict(raw)
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    with materialize(
        Path(source) / row["archive"], row, scratch=scratch, min_free_bytes=c.min_free_bytes
    ) as base:
        values, layout, truth = independent_simulation(cell, cell.n_units, declaration["data_seed"])
        layout = json.loads(json.dumps(layout))
        if read_json(base / "layout.json") != layout:
            raise ValueError("Raw layout differs")
        with np.load(base / "values.npz", allow_pickle=False) as p:
            for k, v in values.items():
                np.testing.assert_allclose(p[k], v, atol=4e-14, rtol=4e-14)
        with np.load(base / "truth.npz", allow_pickle=False) as p:
            np.testing.assert_array_equal(p["labels"], truth)
        workflows = read_json(base / "workflows.json")
        draws = read_json(base / "draws.json")
        assignments = read_json(base / "fold_assignments.json")
        from super_deli.models.predictive_selection import fold_assignments

        if assignments != fold_assignments(
            layout["unit_ids"], replace(c.screening, random_seed=declaration["split_seed"])
        ):
            raise ValueError("Frozen folds differ")
        for i, (draw, seed) in enumerate(zip(draws, c.draw_seeds(declaration), strict=True)):
            expected = (
                np.random.default_rng(seed).integers(cell.n_units, size=cell.n_units).tolist()
            )
            if draw != dict(
                replicate=i, seed=seed, indices=expected, support=unique_support(expected)
            ):
                raise ValueError("Paired draw/support stream differs")
        checked = 0
        evaluated = truth if cell.role == "alternative" else None
        for workflow in workflows:
            arm, i = workflow["arm"], workflow["replicate"]
            directory = base / workflow["prefix"]
            model = replace(c.model, n_init=c.extended_starts) if arm == "extended" else c.model
            expected = (
                list(range(cell.n_units))
                if i == -1
                else draws[i]["support" if arm == "unique" else "indices"]
            )
            if workflow["indices"] != expected:
                raise ValueError("Control weights or original identities differ")
            summary = read_json(directory / "summary.json")
            if i == -1:
                checked += audit_point(directory, values, layout, model)
            else:
                selection = summary["selection"]
                record = dict(
                    status="failed" if selection["status"] != "complete" else "complete",
                    selected_k=selection.get("screened_k"),
                )
                if record["status"] == "complete":
                    with np.load(directory / "original_predictions.npz", allow_pickle=False) as p:
                        record["labels"] = p["labels"].tolist()
                if (directory / "full/record.json").exists():
                    checked += audit_draw(
                        directory, values, layout, np.asarray(expected), model, assignments, record
                    )
            name = "point" if i == -1 else f"draw_{i:04d}"
            reported = read_json(base / "diagnostics" / arm / (name + ".json"))
            audit_metrics(directory, values, layout, expected, model, evaluated, reported)
        count = (
            audit_oracles(base, truth, c.model, assignments) if cell.role == "alternative" else 0
        )
        return dict(
            cell=row["cell"],
            repetition=row["repetition"],
            audited_searches=checked,
            audited_oracles=count,
            **{
                k: row[k]
                for k in (
                    "failed_fits",
                    "unattempted_fits",
                    "withheld_workflows",
                    "unavailable_diagnostics",
                    "failed_oracles",
                    "unattempted_oracles",
                )
            },
        )


def audit_observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _audit_observation(task)


def audit_study(source, *, output_dir, scratch, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four workers")
    source, output = Path(source), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    c = BootstrapDiagnosticConfig.from_dict(read_json(source / "configuration.json"))
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    lookup = {(r["cell"], r["repetition"]): r for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(c.plan()):
        raise ValueError("Incomplete or duplicate observations")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        audited = list(
            pool.map(
                audit_observation,
                [
                    (str(source), c.to_dict(), d, lookup[d["cell"], d["repetition"]], str(scratch))
                    for d in c.plan()
                ],
            )
        )
    write_json(audited, output / "observations.json")
    result = dict(
        status="passed",
        observations=len(audited),
        **{k: sum(r[k] for r in audited) for k in audited[0] if k not in {"cell", "repetition"}},
    )
    write_json(result, output / "summary.json")
    return result
