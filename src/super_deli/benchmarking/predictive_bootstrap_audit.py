"""No-refit audit of expanded empirical measures, fixed-ID folds and retained stability."""

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.local_archives import materialize
from super_deli.benchmarking.predictive_audit import (
    independent_selection,
    independent_simulation,
    project_independent,
    raw_subset,
)
from super_deli.benchmarking.predictive_density_audit import log_density
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy
from super_deli.models.predictive_study_config import PredictiveCell


def audit_geometry(run, values, layout, draw, model):
    import numpy as np

    blocks = []
    unique = sorted(set(map(int, draw)))
    subset, sublayout = raw_subset(values, layout, unique)
    order = [layout["unit_ids"][i] for i in draw]
    weights = np.asarray([v.weight for v in model.views], dtype=float)
    weights /= weights.sum()
    for i, view in enumerate(layout["views"]):
        path = run / f"view_{i:03d}"
        rows = read_json(path / "source_rows.json")
        for key in ("observation_ids", "unit_ids", "target_ids", "feature_names"):
            if rows[key] != sublayout["views"][i][key]:
                raise ValueError("Training identities or member counts differ")
        np.testing.assert_allclose(
            np.load(path / "source_values.npy"), subset[view["kind"]], atol=4e-14, rtol=4e-14
        )
        targets = np.asarray(view["target_ids"])
        means = np.stack([values[view["kind"]][targets == u].mean(0) for u in order])
        with (
            np.load(path / "parameters/pca.npz", allow_pickle=False) as p,
            np.load(path / "parameters/profiles.npz", allow_pickle=False) as profile,
        ):
            kept = p["kept_indices"]
            np.testing.assert_allclose(p["feature_means"], means.mean(0)[kept], atol=1e-12)
            np.testing.assert_allclose(p["feature_scales"], means.std(0)[kept], atol=1e-12)
            standardized = (means[:, kept] - p["feature_means"]) / p["feature_scales"]
            np.testing.assert_allclose(p["mean"], standardized.mean(0), atol=1e-12)
            covariance = np.cov(standardized, rowvar=False)
            c = p["components"]
            np.testing.assert_allclose(c @ c.T, np.eye(len(c)), atol=1e-12)
            np.testing.assert_allclose(
                c @ covariance, p["explained_variance"][:, None] * c, atol=1e-11, rtol=1e-11
            )
            np.testing.assert_allclose(
                p["explained_variance"],
                np.linalg.eigvalsh(covariance)[::-1][: len(c)],
                atol=1e-11,
                rtol=1e-11,
            )
            scores = (standardized - p["mean"]) @ c.T
            np.testing.assert_allclose(profile["profiles"], scores, atol=1e-11, rtol=1e-11)
            center = scores.mean(0)
            scale = math.sqrt(float(np.sum((scores - center) ** 2) / (len(scores) - 1)))
            np.testing.assert_allclose(profile["mean"], center, atol=1e-12)
            np.testing.assert_allclose(profile["scale"], scale, atol=1e-12)
            np.testing.assert_allclose(profile["normalized_weight"], weights[i], atol=1e-14)
            blocks.append((scores - center) / scale * math.sqrt(weights[i]))
    fused = np.column_stack(blocks)
    with np.load(run / "fusion.npz", allow_pickle=False) as p:
        np.testing.assert_allclose(p["fused"], fused, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(p["mean"], fused.mean(0), atol=1e-12)
        c, covariance = p["components"], np.cov(fused, rowvar=False)
        np.testing.assert_allclose(c @ c.T, np.eye(len(c)), atol=1e-12)
        np.testing.assert_allclose(
            c @ covariance, p["explained_variance"][:, None] * c, atol=1e-11, rtol=1e-11
        )
        np.testing.assert_allclose(
            p["explained_variance"],
            np.linalg.eigvalsh(covariance)[::-1][: len(c)],
            atol=1e-11,
            rtol=1e-11,
        )
        expected = (fused - p["mean"]) @ c.T
    with np.load(run / "gmm_candidates.npz", allow_pickle=False) as p:
        np.testing.assert_allclose(p["scores"], expected, atol=1e-11, rtol=1e-11)
    return expected


def audit_pool(run, model, original_indices, heldout=None):
    import numpy as np
    from scipy.special import logsumexp

    candidates = read_json(run / "candidates.json")
    densities = {}
    with np.load(run / "gmm_candidates.npz", allow_pickle=False) as p:
        scores = p["scores"]
        n, d = scores.shape
        for c in candidates:
            k = c["k"]
            if c["status"] == "insufficient_samples":
                if len(set(original_indices)) >= k * model.min_cluster_size:
                    raise ValueError("False distinct-unit infeasibility")
                continue
            w, mu, cov = [p[f"k{k}_{s}"] for s in ("weights", "means", "covariances")]
            np.testing.assert_allclose(w.sum(), 1, atol=1e-12)
            if model.covariance_type == "tied":
                np.testing.assert_array_equal(cov, np.repeat(cov[:1], k, axis=0))
            joint = np.column_stack(
                [
                    np.log(a) + log_density(scores, np.ones(1), m[None], v[None])
                    for a, m, v in zip(w, mu, cov, strict=True)
                ]
            )
            logs = logsumexp(joint, axis=1)
            probabilities = np.exp(joint - logs[:, None])
            labels = probabilities.argmax(1)
            np.testing.assert_allclose(p[f"k{k}_membership"], probabilities, atol=1e-10)
            np.testing.assert_array_equal(p[f"k{k}_labels"], labels)
            sizes = np.bincount(labels, minlength=k).tolist()
            distinct = [
                len({original_indices[j] for j in np.flatnonzero(labels == a)}) for a in range(k)
            ]
            if sizes != c["hard_cluster_sizes"] or distinct != c["distinct_unit_sizes"]:
                raise ValueError("Hard/distinct size accounting differs")
            status = "eligible" if min(distinct) >= model.min_cluster_size else "small_component"
            if status != c["status"] or not c["converged"]:
                raise ValueError("Candidate eligibility/convergence differs")
            q = k - 1 + k * d + (1 if model.covariance_type == "tied" else k) * d * (d + 1) // 2
            np.testing.assert_allclose(
                [c["log_likelihood"], c["aic"], c["bic"]],
                [logs.sum(), -2 * logs.sum() + 2 * q, -2 * logs.sum() + q * np.log(n)],
                atol=1e-7,
                rtol=1e-10,
            )
            if heldout is not None:
                densities[str(k)] = log_density(heldout, w, mu, cov)
    return candidates, densities


def audit_draw(base, values, layout, draw, model, assignments, record):
    import numpy as np

    from super_deli.workflows.predictive import fold_model

    full_record = read_json(base / "full/record.json")
    audited = 1
    if full_record["status"] != "complete":
        if record["status"] == "complete" or not full_record.get("error"):
            raise ValueError("Failed full search misreported")
        return audited
    audit_geometry(base / "full/run", values, layout, draw, model)
    full, _ = audit_pool(base / "full/run", model, draw.tolist())
    retained = read_json(base / "folds.json")
    independent = []
    failed = False
    for assignment, actual in zip(assignments, retained, strict=True):
        i = assignment["fold"]
        keep = set(assignment["validation_indices"])
        test = [j for j, u in enumerate(draw) if u in keep]
        train = [j for j, u in enumerate(draw) if u not in keep]
        if (
            actual["training_indices"] != train
            or actual["validation_indices"] != test
            or actual["original_training_indices"] != draw[train].tolist()
            or actual["original_validation_indices"] != draw[test].tolist()
            or set(draw[train]) & set(draw[test])
        ):
            raise ValueError("Original locality fold leakage or multiplicity loss")
        audited += 1
        prefix = f"fold_{i:03d}"
        if actual["status"] != "complete":
            if not actual.get("error"):
                raise ValueError("Failed fold lacks a reason")
            failed = True
            continue
        trained = fold_model(model, i)
        if read_json(base / prefix / "model.json") != json.loads(json.dumps(trained.to_dict())):
            raise ValueError("Fold optimizer settings differ")
        audit_geometry(base / prefix / "run", values, layout, draw[train], trained)
        scores = project_independent(base / prefix / "run", values, layout)[draw[test]]
        candidates, density = audit_pool(
            base / prefix / "run", trained, draw[train].tolist(), scores
        )
        with np.load(base / prefix / "validation.npz", allow_pickle=False) as p:
            np.testing.assert_allclose(p["scores"], scores, atol=1e-11, rtol=1e-11)
            for k, logs in density.items():
                np.testing.assert_allclose(p["k" + k], logs, atol=1e-8, rtol=1e-10)
                np.testing.assert_allclose(actual["log_densities"][k], logs, atol=1e-8, rtol=1e-10)
        independent.append({"candidates": candidates, "density": density})
    result = read_json(base / "summary.json")
    if failed:
        if result["selection"]["status"] == "complete" or record["status"] == "complete":
            raise ValueError("Unresolved fold became an available draw")
        return audited
    expected = independent_selection(full, independent, len(draw))
    actual = result["selection"]
    for key in ("ordinary_aic_k", "descriptive_bic_k", "screened_k"):
        if expected[key] != actual[key]:
            raise ValueError("Screen selection differs")
    for a, b in zip(expected["candidates"], actual["candidates"], strict=True):
        for key in ("k", "reason", "passes", "fold_statuses"):
            if a[key] != b[key]:
                raise ValueError("Screen gate differs")
        if a["pooled_validation_gain"] is not None:
            np.testing.assert_allclose(
                a["pooled_validation_gain"], b["pooled_validation_gain"], atol=1e-8, rtol=1e-10
            )
    scores = project_independent(base / "full/run", values, layout)
    k = expected["screened_k"]
    with (
        np.load(base / "selected_model.npz", allow_pickle=False) as selected,
        np.load(base / "full/run/gmm_candidates.npz", allow_pickle=False) as pool,
        np.load(base / "original_predictions.npz", allow_pickle=False) as predicted,
    ):
        order = selected["component_order"]
        if str(selected["covariance_type"]) != "tied":
            raise ValueError("Selected family differs")
        for key in ("weights", "means", "covariances"):
            np.testing.assert_array_equal(selected[key], pool[f"k{k}_{key}"][order])
        joint = np.column_stack(
            [
                np.log(a) + log_density(scores, np.ones(1), m[None], v[None])
                for a, m, v in zip(
                    selected["weights"], selected["means"], selected["covariances"], strict=True
                )
            ]
        )
        logs = np.logaddexp.reduce(joint, axis=1)
        probability = np.exp(joint - logs[:, None])
        np.testing.assert_allclose(predicted["scores"], scores, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(predicted["membership"], probability, atol=1e-10)
        np.testing.assert_array_equal(predicted["labels"], probability.argmax(1))
        if record["labels"] != predicted["labels"].tolist() or record["selected_k"] != k:
            raise ValueError("Retained original-unit partition differs")
    return audited


def audit_stability(base, result):
    import numpy as np
    from scipy.stats import binomtest

    records = result["replicates"]
    n = len(result["unit_ids"])
    r = len(records)
    point = np.load(base / "point/selected_memberships.npz")["labels"]
    point_k = result["point"]["selection"]["screened_k"]
    counts = np.zeros((n, n), int)
    oob_counts = counts.copy()
    requested = counts.copy()
    success = counts.copy()
    exact = np.zeros(point_k, int)
    ks = {}
    complete = 0
    for row in records:
        missing = set(range(n)) - set(row["training_indices"])
        oob = np.zeros((n, n), bool)
        for j in missing:
            oob[j, list(missing)] = True
        requested += oob
        if row["status"] != "complete":
            continue
        labels = np.asarray(row["labels"])
        same = labels[:, None] == labels[None, :]
        counts += same
        oob_counts += same & oob
        success += oob
        complete += 1
        k = row["selected_k"]
        ks[k] = ks.get(k, 0) + 1
        for j in range(point_k):
            g = set(np.flatnonzero(point == j))
            exact[j] += any(g == set(np.flatnonzero(labels == a)) for a in range(k))
    with np.load(base / "stability.npz", allow_pickle=False) as p:
        for name, a, b, c, minimum in (
            ("all", counts, r, complete, 1),
            ("oob", oob_counts, requested, success, 10),
            ("exact_cluster", exact, r, complete, 1),
            (
                "selected_k",
                np.array([ks.get(k, 0) for k in range(1, len(p["selected_k_counts"]) + 1)]),
                r,
                complete,
                1,
            ),
        ):
            a, b, c = np.broadcast_arrays(a, b, c)
            for key, expected in [("counts", a), ("requested", b), ("successful", c)]:
                np.testing.assert_array_equal(p[name + "_" + key], expected)
            low = np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
            high = np.divide(a + b - c, b, out=np.full(a.shape, np.nan), where=b > 0)
            conditional = np.divide(a, c, out=np.full(a.shape, np.nan), where=c >= minimum)
            freq = np.where((b == c) & (c >= minimum), low, np.nan)
            for key, expected in [
                ("failure_lower", low),
                ("failure_upper", high),
                ("frequency", freq),
                ("conditional_frequency", conditional),
            ]:
                np.testing.assert_allclose(p[name + "_" + key], expected, equal_nan=True)
            np.testing.assert_allclose(
                p[name + "_conditional_mcse"],
                np.sqrt(conditional * (1 - conditional) / np.maximum(c, 1)),
                equal_nan=True,
            )
            # Deduplicate equal binomial count pairs; exact independent SciPy interface.
            for x, y in set(zip(a[c >= minimum].flat, c[c >= minimum].flat, strict=True)):
                interval = binomtest(int(x), int(y)).proportion_ci(method="exact")
                mask = (a == x) & (c == y) & (c >= minimum)
                np.testing.assert_allclose(
                    p[name + "_conditional_mc_low"][mask], interval.low, atol=1e-11
                )
                np.testing.assert_allclose(
                    p[name + "_conditional_mc_high"][mask], interval.high, atol=1e-11
                )
    if {str(k): v for k, v in ks.items()} != {
        str(k): v for k, v in result["stability"]["selected_k_counts"].items()
    }:
        raise ValueError("Selected-K totals differ")


def _audit_observation(task):
    import numpy as np

    source, raw, declaration, row, scratch = task
    c = PredictiveBootstrapStudy.from_dict(raw)
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    with materialize(
        Path(source) / row["archive"], row, scratch=scratch, min_free_bytes=c.min_free_bytes
    ) as base:
        expected, layout, truth = independent_simulation(
            cell, cell.n_units, declaration["data_seed"]
        )
        with np.load(base / "values.npz", allow_pickle=False) as p:
            for key, value in expected.items():
                np.testing.assert_allclose(p[key], value, atol=4e-14, rtol=4e-14)
        layout = json.loads(json.dumps(layout))
        if read_json(base / "generating_layout.json") != layout:
            raise ValueError("Raw layout differs")
        with np.load(base / "truth.npz", allow_pickle=False) as p:
            np.testing.assert_array_equal(p["labels"], truth)
        result = read_json(base / "workflow/summary.json")
        if result["point"]["selection"]["status"] != "complete":
            raise ValueError("Validation point unavailable; retain for explicit audit review")
        assignment = read_json(base / "workflow/point/fold_assignments.json")
        from super_deli.models.predictive_selection import fold_assignments

        screening = replace(c.screening, random_seed=declaration["split_seed"])
        if assignment != fold_assignments(layout["unit_ids"], screening):
            raise ValueError("Fixed folds differ")
        # Point geometry/density audits share independent primitives but use no bootstrap refits.
        searches = 0
        for prefix in ["full"] + [f"fold_{i:03d}" for i in range(5)]:
            indices = (
                np.arange(cell.n_units)
                if prefix == "full"
                else np.asarray(assignment[int(prefix[-3:])]["training_indices"])
            )
            trained = c.model
            if prefix != "full":
                from super_deli.workflows.predictive import fold_model

                trained = fold_model(c.model, int(prefix[-3:]))
            audit_geometry(
                base / "workflow/point" / prefix / "run", expected, layout, indices, trained
            )
            audit_pool(base / "workflow/point" / prefix / "run", trained, indices.tolist())
            searches += 1
        point_base = base / "workflow/point"
        point_folds = []
        for fold in assignment:
            prefix = f"fold_{fold['fold']:03d}"
            scores = project_independent(point_base / prefix / "run", expected, layout)[
                fold["validation_indices"]
            ]
            from super_deli.workflows.predictive import fold_model

            candidates, density = audit_pool(
                point_base / prefix / "run",
                fold_model(c.model, fold["fold"]),
                fold["training_indices"],
                scores,
            )
            with np.load(point_base / prefix / "validation.npz", allow_pickle=False) as p:
                np.testing.assert_allclose(p["scores"], scores, atol=1e-11, rtol=1e-11)
                for key, logs in density.items():
                    np.testing.assert_allclose(p["k" + key], logs, atol=1e-8, rtol=1e-10)
            point_folds.append({"candidates": candidates, "density": density})
        point_choice = independent_selection(
            read_json(point_base / "full/run/candidates.json"), point_folds, cell.n_units
        )
        for key in ("ordinary_aic_k", "descriptive_bic_k", "screened_k"):
            if point_choice[key] != result["point"]["selection"][key]:
                raise ValueError("Point selection differs from independent gate")
        seeds = PredictiveBootstrapConfig(c.n_bootstrap, declaration["bootstrap_seed"]).draw_seeds()
        for i, (replicate, seed) in enumerate(zip(result["replicates"], seeds, strict=True)):
            draw = np.random.default_rng(seed).integers(cell.n_units, size=cell.n_units)
            if (
                replicate["replicate"] != i
                or replicate["draw_seed"] != seed
                or replicate["training_indices"] != draw.tolist()
                or replicate["multiplicities"] != np.bincount(draw, minlength=cell.n_units).tolist()
            ):
                raise ValueError("Bootstrap seed or original identity/multiplicity differs")
            searches += audit_draw(
                base / "workflow" / replicate["prefix"],
                expected,
                layout,
                draw,
                c.model,
                assignment,
                replicate,
            )
        audit_stability(base / "workflow", result)
        if any(
            result[k] != row[k]
            for k in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")
        ):
            raise ValueError("Outer search accounting differs")
        return {
            "cell": row["cell"],
            "repetition": row["repetition"],
            "audited_searches": searches,
            **{
                k: row[k]
                for k in (
                    "successful_fits",
                    "failed_fits",
                    "unattempted_fits",
                    "failed_draws",
                    "unattempted_draws",
                    "successful_draws",
                )
            },
        }


def audit_study(source, *, output_dir, scratch, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four workers")
    source = Path(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    c = PredictiveBootstrapStudy.from_dict(read_json(source / "configuration.json"))
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    lookup = {(r["cell"], r["repetition"]): r for r in rows}
    plan = c.plan()
    if len(lookup) != len(rows) or len(rows) != len(plan):
        raise ValueError("Incomplete observation ledger")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        audited = list(
            pool.map(
                audit_observation,
                [
                    (str(source), c.to_dict(), d, lookup[d["cell"], d["repetition"]], str(scratch))
                    for d in plan
                ],
            )
        )
    write_json(audited, output / "observations.json")
    result = {
        "status": "passed",
        "observations": len(audited),
        **{k: sum(r[k] for r in audited) for k in audited[0] if k not in {"cell", "repetition"}},
    }
    write_json(result, output / "summary.json")
    return result


def audit_observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _audit_observation(task)
