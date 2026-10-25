"""Post-fit signal, constraint and optimization diagnostics; generating truth is evaluation only."""

from pathlib import Path

from super_deli.benchmarking.bootstrap_diagnostic_gates import gates, truth_sizes
from super_deli.io.config import read_json, write_json

PREFIXES = ("full",) + tuple(f"fold_{i:03d}" for i in range(5))


def axis_information(matrix, labels):
    import numpy as np

    from super_deli.benchmarking.recovery_metrics import axis_r2

    if labels is None or not len(labels):
        return None
    labels = np.asarray(labels)
    # Degenerate generating contrasts are unavailable rather than division by zero.
    result = {}
    for name, axis in (("a", labels // 2), ("b", labels % 2)):
        if np.unique(axis).size < 2:
            result[name] = None
        else:
            # Reuse the existing metric with both axes available in a harmless
            # recoding; the requested contrast is unchanged.
            x = np.column_stack([np.ones(len(matrix)), matrix])
            y = 2 * axis - 1
            residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
            result[name] = float(1 - residual @ residual / ((y - y.mean()) @ (y - y.mean())))
    if all(v is not None for v in result.values()):
        reference = axis_r2(matrix, labels)
        np.testing.assert_allclose(list(result.values()), list(reference.values()), atol=1e-12)
    return result


def matched_changes(reference, labels):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if reference is None or not len(reference):
        return None
    a, b = np.asarray(reference), np.asarray(labels)
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=int)
    np.add.at(table, (a, b), 1)
    i, j = linear_sum_assignment(-table)
    return int(len(a) - table[i, j].sum())


def workflow_folds(directory, indices):
    import numpy as np

    directory = Path(directory)
    if (directory / "folds.json").exists():
        return read_json(directory / "folds.json")
    rows = []
    assignments = read_json(directory / "fold_assignments.json")
    for f in assignments:
        prefix = f"fold_{f['fold']:03d}"
        row = read_json(directory / prefix / "validation_record.json")
        row.update(
            original_training_indices=[indices[j] for j in f["training_indices"]],
            original_validation_indices=[indices[j] for j in f["validation_indices"]],
        )
        if row["status"] == "complete":
            row["candidates"] = read_json(directory / prefix / "run/candidates.json")
            with np.load(directory / prefix / "validation.npz", allow_pickle=False) as p:
                row["log_densities"] = {
                    str(c["k"]): p["k" + str(c["k"])].tolist()
                    for c in row["candidates"]
                    if c["status"] != "insufficient_samples"
                }
        rows.append(row)
    return rows


def search_metrics(run, values, layout, indices, truth, global_oob):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.io.predictive_inputs import project_exported
    from super_deli.models.predictive_selection import gaussian_predictions

    run = Path(run)
    indices = np.asarray(indices, dtype=int)
    labels = None if truth is None else np.asarray(truth)[indices]
    raw_blocks = []
    view_metrics = []
    for i, view in enumerate(layout["views"]):
        targets = np.asarray(view["target_ids"])
        means = np.stack([values[view["kind"]][targets == u].mean(0) for u in layout["unit_ids"]])
        matrix = means[indices]
        path = run / f"view_{i:03d}/parameters"
        with (
            np.load(path / "pca.npz", allow_pickle=False) as p,
            np.load(path / "profiles.npz", allow_pickle=False) as profile,
        ):
            standardized = (matrix[:, p["kept_indices"]] - p["feature_means"]) / p["feature_scales"]
            source_rows = np.concatenate(
                [np.flatnonzero(targets == layout["unit_ids"][u]) for u in indices]
            )
            source_matrix = values[view["kind"]][source_rows][:, p["kept_indices"]]
            source_scores = (
                (source_matrix - p["feature_means"]) / p["feature_scales"] - p["mean"]
            ) @ p["components"].T
            source_labels = (
                None
                if labels is None
                else np.concatenate(
                    [
                        np.repeat(labels[j], np.sum(targets == layout["unit_ids"][u]))
                        for j, u in enumerate(indices)
                    ]
                )
            )
            view_scores = (standardized - p["mean"]) @ p["components"].T
            raw_blocks.append(matrix)
            view_metrics.append(
                dict(
                    kind=view["kind"],
                    raw_mean_axis_r2=axis_information(matrix, labels),
                    standardized_axis_r2=axis_information(standardized, labels),
                    profile_axis_r2=axis_information(profile["profiles"], labels),
                    view_pca_axis_r2=axis_information(view_scores, labels),
                    source_projected_axis_r2=axis_information(source_scores, source_labels),
                    profile_scale=float(profile["scale"]),
                    n_training_occurrences=len(indices),
                    n_distinct_localities=len(set(indices.tolist())),
                )
            )
    with np.load(run / "fusion.npz", allow_pickle=False) as p:
        fused_r2 = axis_information(p["fused"], labels)
        joint_variance = p["explained_variance_ratio"].tolist()
    projected = project_exported(run, values, layout)
    raw = np.column_stack(raw_blocks)
    n_original = len(layout["unit_ids"])
    oob = np.asarray(global_oob, dtype=int)
    candidates = read_json(run / "candidates.json")
    metrics = []
    with np.load(run / "gmm_candidates.npz", allow_pickle=False) as p:
        scores = p["scores"]
        n, d = scores.shape
        q_one = d + d * (d + 1) // 2
        one = next(c for c in candidates if c["k"] == 1)
        for c in candidates:
            k = c["k"]
            if c["status"] == "insufficient_samples":
                metrics.append(dict(k=k, status=c["status"]))
                continue
            probability = p[f"k{k}_membership"]
            predicted = p[f"k{k}_labels"]
            _, all_probability = gaussian_predictions(projected, p, k)
            all_labels = all_probability.argmax(1)
            eigen = np.linalg.eigvalsh(p[f"k{k}_covariances"])
            q = k - 1 + k * d + d * (d + 1) // 2
            metrics.append(
                dict(
                    k=k,
                    status=c["status"],
                    parameters=q,
                    log_likelihood=c["log_likelihood"],
                    aic=c["aic"],
                    bic=c["bic"],
                    twice_likelihood_gain_over_one=2
                    * (c["log_likelihood"] - one["log_likelihood"]),
                    aic_penalty_difference_over_one=2 * (q - q_one),
                    bic_penalty_difference_over_one=(q - q_one) * float(np.log(n)),
                    occurrence_sizes=c["hard_cluster_sizes"],
                    distinct_sizes=c["distinct_unit_sizes"],
                    entropy=float(
                        -np.sum(probability * np.log(np.maximum(probability, 1e-300))) / n
                    ),
                    covariance_eigenvalues=eigen.tolist(),
                    covariance_condition_numbers=(eigen[:, -1] / eigen[:, 0]).tolist(),
                    iterations=c["iterations"],
                    converged=c["converged"],
                    training_occurrence_ari=float(adjusted_rand_score(labels, predicted))
                    if labels is not None
                    else None,
                    all_original_ari=float(adjusted_rand_score(truth, all_labels))
                    if truth is not None
                    else None,
                    global_oob_ari=float(
                        adjusted_rand_score(np.asarray(truth)[oob], all_labels[oob])
                    )
                    if truth is not None and len(oob) > 1
                    else None,
                    global_oob_count=len(oob),
                    all_original_count=n_original,
                )
            )
        joint_r2 = axis_information(scores, labels)
    return dict(
        n_occurrences=len(indices),
        n_distinct=len(set(indices.tolist())),
        generating_distinct_sizes=truth_sizes(truth, indices.tolist()),
        count_axis_r2=axis_information(np.asarray(layout["counts"])[indices, None], labels),
        raw_joint_axis_r2=axis_information(raw, labels),
        views=view_metrics,
        fused_axis_r2=fused_r2,
        joint_axis_r2=joint_r2,
        joint_variance_ratio=joint_variance,
        projected_all_axis_r2=axis_information(projected, truth),
        projected_global_oob_axis_r2=axis_information(
            projected[oob], None if truth is None else np.asarray(truth)[oob]
        ),
        candidates=metrics,
    )


def diagnose_workflow(directory, values, layout, indices, model, truth=None, point_labels=None):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    directory = Path(directory)
    indices = np.asarray(indices, dtype=int)
    summary = read_json(directory / "summary.json")
    original_oob = sorted(set(range(len(layout["unit_ids"]))) - set(indices.tolist()))
    folds = (
        workflow_folds(directory, indices)
        if (directory / "folds.json").exists() or (directory / "fold_assignments.json").exists()
        else []
    )
    full = (
        read_json(directory / "full/run/candidates.json")
        if (directory / "full/run/candidates.json").exists()
        else []
    )
    gate = gates(
        full,
        folds,
        indices.tolist(),
        summary["selection"],
        minimum=model.min_cluster_size,
        truth=truth,
    )
    searches = {}
    mapping = {
        "full": indices.tolist(),
        **{f"fold_{f['fold']:03d}": f["original_training_indices"] for f in folds},
    }
    for prefix, draw in mapping.items():
        run = directory / prefix / "run"
        if (run / "gmm_candidates.npz").exists() and (run / "candidates.json").exists():
            searches[prefix] = search_metrics(run, values, layout, draw, truth, original_oob)
        else:
            searches[prefix] = dict(status="unavailable", error="No retained candidate parameters")
    predicted = None
    path = directory / "original_predictions.npz"
    if not path.exists() and (directory / "fold_assignments.json").exists():
        path = directory / "selected_memberships.npz"
    if summary["selection"]["status"] == "complete" and path.exists():
        with np.load(path, allow_pickle=False) as p:
            predicted = p["labels"]
    return dict(
        selection=summary["selection"],
        gate=gate,
        searches=searches,
        n_original=len(layout["unit_ids"]),
        draw_indices=indices.tolist(),
        global_oob=original_oob,
        original_multiplicities=np.bincount(indices, minlength=len(layout["unit_ids"])).tolist(),
        selected_ari=float(adjusted_rand_score(truth, predicted))
        if truth is not None and predicted is not None
        else None,
        point_partition_ari=float(adjusted_rand_score(point_labels, predicted))
        if point_labels is not None and predicted is not None
        else None,
        point_assignment_changes=matched_changes(point_labels, predicted)
        if predicted is not None
        else None,
    )


def truth_diagnostics(workflow, truth, indices, model, assignments, destination):
    import numpy as np

    from super_deli.benchmarking.grouped_recovery_metrics import truth_initialized
    from super_deli.workflows.predictive import fold_model

    destination = Path(destination)
    destination.mkdir()
    indices = np.asarray(indices, dtype=int)
    rows = []
    for prefix in PREFIXES:
        if prefix != "full" and len(assignments) != 5:
            target = destination / prefix
            target.mkdir()
            row = dict(
                prefix=prefix,
                original_indices=[],
                attempted=False,
                status="unattempted",
                error="Original fold partition unavailable",
                role="Truth-initialized optimization only; never selection",
            )
            write_json(row, target / "diagnostic.json")
            rows.append(row)
            continue
        draw = (
            indices
            if prefix == "full"
            else indices[
                [
                    j
                    for j, u in enumerate(indices)
                    if u in set(assignments[int(prefix[-3:])]["training_indices"])
                ]
            ]
        )
        trained = model if prefix == "full" else fold_model(model, int(prefix[-3:]))
        run = Path(workflow) / prefix / "run"
        row = dict(
            prefix=prefix,
            original_indices=draw.tolist(),
            attempted=False,
            role="Truth-initialized optimization only; never selection",
        )
        target = destination / prefix
        if not (run / "gmm_candidates.npz").exists():
            row.update(status="unattempted", error="Baseline score search unavailable")
        elif set(np.asarray(truth)[draw].tolist()) != {0, 1, 2, 3}:
            row.update(status="unattempted", error="Generating group absent from training support")
        else:
            with np.load(run / "gmm_candidates.npz", allow_pickle=False) as p:
                scores = p["scores"].copy()
            try:
                fitted = truth_initialized(scores, np.asarray(truth)[draw], trained, target)
                row.update(attempted=True, status=fitted["status"])
                if fitted["status"] == "complete":
                    with np.load(target / "mixture.npz", allow_pickle=False) as p:
                        labels = p["labels"]
                        n, d = p["scores"].shape
                        sizes = [len(set(draw[labels == k].tolist())) for k in range(4)]
                        # Original helper's occurrence-size eligibility is retained
                        # separately, never used for bootstrap eligibility.
                        q = 3 + 4 * d + d * (d + 1) // 2
                        ll = (q * np.log(n) - fitted["bic"]) / 2
                        row.update(
                            distinct_unit_sizes=sizes,
                            eligible_original_ids=min(sizes) >= model.min_cluster_size,
                            legacy_occurrence_eligibility=fitted["eligible"],
                            log_likelihood=float(ll),
                            aic=float(-2 * ll + 2 * q),
                            bic=fitted["bic"],
                            iterations=fitted["iterations"],
                        )
                else:
                    row["error"] = fitted["error"]
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row.update(attempted=True, status="failed", error=str(exc))
        target.mkdir(exist_ok=True)
        write_json(row, target / "diagnostic.json")
        rows.append(row)
    write_json(rows, destination / "summary.json")
    return rows
