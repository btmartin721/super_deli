"""Independent diagnostics audits; no model fitting or production changes."""

from dataclasses import replace

from super_deli.benchmarking.aic_diagnostic_audit import check_pool
from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.predictive_audit import (
    audit_search,
    independent_selection,
    project_independent,
    raw_subset,
)
from super_deli.benchmarking.recovery_analysis import _ari, _axis_check
from super_deli.io.config import read_json


def close(actual, expected):
    import numpy as np

    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Diagnostic fields differ")
        for k, v in expected.items():
            close(actual[k], v)
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError("Diagnostic lengths differ")
        for a, b in zip(actual, expected, strict=True):
            close(a, b)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
    elif actual != expected:
        raise ValueError("Diagnostic value differs")


def audit_gates(full, folds, n, reported):
    import numpy as np

    if any(f["status"] != "complete" for f in folds):
        if reported is not None and reported["status"] != "withheld":
            raise ValueError("Invalid available gate diagnosis")
        return
    records = []
    allowed = {name: [] for name in ("ordinary", "eligibility_only", "gain_only", "actual")}
    for c in full:
        k = c["k"]
        states = [next(x["status"] for x in f["candidates"] if x["k"] == k) for f in folds]
        deltas = [f["density"][str(k)] - f["density"]["1"] for f in folds if str(k) in f["density"]]
        total = np.concatenate(deltas) if deltas else np.array([])
        gain = float(total.sum() / n) if len(total) == n else None
        eligible = c["status"] == "eligible"
        every = all(s == "eligible" for s in states)
        positive = gain is not None and gain > 0
        passes = {
            "ordinary": eligible,
            "eligibility_only": eligible and every,
            "gain_only": eligible and (k == 1 or positive),
            "actual": eligible and every and (k == 1 or positive),
        }
        for name, value in passes.items():
            if value:
                allowed[name].append(c)
        records.append(
            {
                "k": k,
                "full_status": c["status"],
                "fold_statuses": states,
                "fold_ineligible_count": sum(s != "eligible" for s in states),
                "pooled_gain_all_fitted_folds": gain,
                "fold_mean_gains": [float(v.mean()) for v in deltas],
                "minimum_unit_gain": float(total.min()) if len(total) else None,
                "maximum_unit_gain": float(total.max()) if len(total) else None,
                "negative_units": int(np.sum(total < 0)),
                "passes": passes,
            }
        )
    if reported["status"] != "complete":
        raise ValueError("Complete gate diagnosis withheld")
    close(reported["candidates"], records)
    for name, cs in allowed.items():
        if reported[name + "_k"] != min(cs, key=lambda c: (c["aic"], c["k"]))["k"]:
            raise ValueError("Counterfactual selection differs")


def audit_signal(run, labels, reported):
    import numpy as np

    units = read_json(run / "summary.json")["unit_ids"]
    position = {u: i for i, u in enumerate(units)}
    blocks = []
    for i, kind in enumerate(("morphological", "ecological")):
        base = run / f"view_{i:03d}"
        rows = read_json(base / "source_rows.json")
        raw = np.load(base / "source_values.npy", allow_pickle=False)
        target = np.array([position[u] for u in rows["target_ids"]])
        counts = np.bincount(target, minlength=len(units))
        means = np.array([raw[target == j].sum(0) / counts[j] for j in range(len(units))])
        v = reported["upstream"][kind]
        close(v["counts"], counts.tolist())
        close(v["raw_group_mean"], means.tolist())
        close(v["counts_by_label"], [counts[labels == j].tolist() for j in range(4)])
        _axis_check(means, labels, v["raw_group_axis_r2"])
        if np.ptp(counts):
            _axis_check(counts[:, None], labels, v["count_axis_r2"])
        else:
            close(v["count_axis_r2"], {"a": 0.0, "b": 0.0})
        with np.load(base / "parameters/pca.npz", allow_pickle=False) as p:
            transformed = (raw[:, p["kept_indices"]] - p["feature_means"]) / p["feature_scales"]
            scores = transformed @ p["components"].T - p["mean"] @ p["components"].T
            close(v["source_pca_components"], p["components"].tolist())
            close(v["source_variance_ratio"], p["explained_variance_ratio"].tolist())
        _axis_check(scores, labels[target], v["source_axis_r2"])
        profiles = np.array([scores[target == j].mean(0) for j in range(len(units))])
        _axis_check(profiles, labels, v["profile_axis_r2"])
        with np.load(base / "parameters/profiles.npz", allow_pickle=False) as p:
            np.testing.assert_allclose(profiles, p["profiles"], atol=1e-11)
            blocks.append((profiles - p["mean"]) / p["scale"] * np.sqrt(p["normalized_weight"]))
    _axis_check(np.concatenate(blocks, axis=1), labels, reported["fused_axis_r2"])


def external_densities(pool, candidates, held):
    """Audit saved densities by triangular solves, avoiding eigenvalue cancellation."""
    import numpy as np
    from scipy.linalg import cholesky, solve_triangular
    from scipy.special import logsumexp

    density = {}
    scores = held["scores"]
    for candidate in candidates:
        if candidate["status"] == "insufficient_samples":
            continue
        k = candidate["k"]
        terms = []
        for weight, mean, covariance in zip(
            pool[f"k{k}_weights"], pool[f"k{k}_means"], pool[f"k{k}_covariances"], strict=True
        ):
            factor = cholesky(covariance, lower=True)
            solved = solve_triangular(factor, (scores - mean).T, lower=True)
            logdet = 2 * np.log(np.diag(factor)).sum()
            terms.append(
                np.log(weight)
                - 0.5 * (scores.shape[1] * np.log(2 * np.pi) + logdet + (solved * solved).sum(0))
            )
        values = logsumexp(np.column_stack(terms), axis=1)
        np.testing.assert_allclose(held[f"k{k}"], values, atol=1e-8, rtol=1e-10)
        density[str(k)] = values
    return density


def audit_diagnostics(
    workflow, truth, tests, test_layout, model, destination, full=None, folds=None
):
    import numpy as np
    from scipy.linalg import orth
    from scipy.special import xlogy

    m = read_json(destination / "metrics.json")
    if full is not None:
        audit_gates(
            full,
            folds,
            len(truth)
            if truth is not None
            else len(read_json(workflow / "full/run/summary.json")["unit_ids"]),
            m["gates"],
        )
    assignments = (
        read_json(workflow / "fold_assignments.json")
        if (workflow / "fold_assignments.json").exists()
        else []
    )
    by = {f"fold_{r['fold']:03d}": r for r in assignments}
    fullmaps = {}
    for row in m["spaces"]:
        prefix = row["prefix"]
        run = workflow / prefix / "run"
        if row["status"] != "complete":
            if (run / "gmm_candidates.npz").exists():
                raise ValueError("Available space marked unavailable")
            continue
        cs = read_json(run / "candidates.json")
        projected = tests if test_layout is None else project_independent(run, tests, test_layout)
        with np.load(destination / (prefix + "_external.npz"), allow_pickle=False) as held:
            np.testing.assert_allclose(held["scores"], projected, atol=1e-10)
            check_pool(
                (run / "gmm_candidates.npz").read_bytes(),
                cs,
                counts=None,
                config=model,
            )
            with np.load(run / "gmm_candidates.npz", allow_pickle=False) as pool:
                density = external_densities(pool, cs, held)
        labels = (
            None
            if truth is None
            else truth
            if prefix == "full"
            else truth[by[prefix]["training_indices"]]
        )
        with np.load(run / "gmm_candidates.npz", allow_pickle=False) as pool:
            n, d = pool["scores"].shape
            one = next(c for c in cs if c["k"] == 1)
            diff = pool["scores"] - pool["k1_means"][0]
            distances = np.einsum("ij,ij->i", diff, diff @ np.linalg.inv(pool["k1_covariances"][0]))
            close(row["one_component_mahalanobis_squared"], distances.tolist())
            close(
                row["extreme_unit_indices"],
                np.argsort(-distances, kind="stable")[: min(10, n)].tolist(),
            )
            for c in row["candidates"]:
                original = next(x for x in cs if x["k"] == c["k"])
                k = c["k"]
                for key, value in original.items():
                    close(c[key], value)
                if c["status"] == "insufficient_samples":
                    continue
                eigen = np.linalg.eigvalsh(pool[f"k{k}_covariances"])
                q = k - 1 + k * d + k * d * (d + 1) // 2
                expected = {
                    "parameters": q,
                    "parameters_per_unit": q / n,
                    "twice_loglikelihood_gain_over_one": 2
                    * (c["log_likelihood"] - one["log_likelihood"]),
                    "aic_penalty_over_one": 2 * (q - (d + d * (d + 1) // 2)),
                    "bic_penalty_over_one": float(np.log(n) * (q - (d + d * (d + 1) // 2))),
                    "minimum_eigenvalues": eigen[:, 0].tolist(),
                    "minimum_eigenvalue_over_regularizer": float(eigen.min() / model.reg_covar),
                    "condition_numbers": (eigen[:, -1] / eigen[:, 0]).tolist(),
                    "weights": pool[f"k{k}_weights"].tolist(),
                    "mean_assignment_entropy": float(
                        -xlogy(pool[f"k{k}_membership"], pool[f"k{k}_membership"]).sum() / n
                    ),
                    "component_mean_mahalanobis_squared": [
                        float(distances[pool[f"k{k}_labels"] == j].mean())
                        if np.any(pool[f"k{k}_labels"] == j)
                        else None
                        for j in range(k)
                    ],
                }
                gain = density[str(k)] - density["1"]
                expected.update(
                    heldout_mean_logdensity_gain_over_one=float(gain.mean()),
                    heldout_mcse_conditional_on_training=float(
                        gain.std(ddof=1) / np.sqrt(len(gain))
                    ),
                )
                if labels is not None:
                    pred = pool[f"k{k}_labels"]
                    expected["ari"] = _ari(labels, pred)
                    expected["generating_by_component"] = [
                        [int(np.sum((labels == a) & (pred == b))) for b in range(k)]
                        for a in range(4)
                    ]
                for key, value in expected.items():
                    close(c[key], value)
            if labels is not None:
                _axis_check(pool["scores"], labels, row["joint_axis_r2"])
                close(row["generating_counts"], np.bincount(labels, minlength=4).tolist())
        if test_layout is not None:
            for i, maprow in enumerate(row["maps"]):
                with np.load(run / f"view_{i:03d}/parameters/pca.npz", allow_pickle=False) as p:
                    matrix = np.zeros((len(p["components"]), 4))
                    matrix[:, p["kept_indices"]] = p["components"] / p["feature_scales"]
                    close(maprow["loadings_raw_coordinates"], matrix.tolist())
                    close(
                        maprow["explained_variance_ratio"], p["explained_variance_ratio"].tolist()
                    )
                    if prefix == "full":
                        fullmaps[i] = matrix
                    singular = np.linalg.svd(
                        orth(fullmaps[i].T).T @ orth(matrix.T), compute_uv=False
                    )
                    np.testing.assert_allclose(
                        np.cos(maprow["angles_from_full_radians"]),
                        np.clip(singular, 0, 1)[::-1],
                        atol=1e-12,
                    )
            if labels is not None:
                audit_signal(run, labels, row)
    return m


def audit_partition(base, values, layout, truth, tests, test_layout, model, seed, n_folds):
    import csv
    import json

    import numpy as np

    workflow = base / "workflow"
    summary = read_json(workflow / "summary.json")
    config = read_json(workflow / "configuration.json")
    close(config["predictive_model"], json.loads(json.dumps(model.to_dict())))
    if config["screening"]["random_seed"] != seed:
        raise ValueError("Partition seed differs")
    full_record, full, _ = audit_search(workflow, "full", model, values, layout)
    if full_record["status"] != "complete":
        if summary["selection"]["status"] != "withheld":
            raise ValueError("Failed full fit has selection")
        return summary
    units = layout["unit_ids"]
    parts = np.array_split(np.random.default_rng(seed).permutation(len(units)), n_folds)
    assignments = read_json(workflow / "fold_assignments.json")
    folds = []
    for i, part in enumerate(parts):
        valid = sorted(map(int, part))
        train = [j for j in range(len(units)) if j not in set(valid)]
        expected = {
            "fold": i,
            "training_indices": train,
            "validation_indices": valid,
            "training_unit_ids": [units[j] for j in train],
            "validation_unit_ids": [units[j] for j in valid],
        }
        close(assignments[i], expected)
        v, vl = raw_subset(values, layout, valid)
        t, tl = raw_subset(values, layout, train)
        prefix = f"fold_{i:03d}"
        directory = workflow / prefix
        close(read_json(directory / "training_layout.json"), tl)
        close(read_json(directory / "validation_layout.json"), vl)
        trained = replace(
            model,
            random_seed=derived_seed(model.random_seed, "predictive_fold", i, "model") % 2**32,
        )
        projection = (
            project_independent(directory / "run", v, vl)
            if (directory / "validation.npz").exists()
            else None
        )
        record, cs, density = audit_search(workflow, prefix, trained, t, tl, projection)
        validation = read_json(directory / "validation_record.json")
        for key, value in expected.items():
            close(validation[key], value)
        folds.append({"status": validation["status"], "candidates": cs, "density": density})
    if all(f["status"] == "complete" for f in folds):
        expected = independent_selection(full, folds, len(units))
        for key, value in expected.items():
            close(summary["selection"][key], value)
    elif summary["selection"]["status"] != "withheld":
        raise ValueError("Incomplete screen has selection")
    audit_diagnostics(workflow, truth, tests, test_layout, model, base / "diagnostics", full, folds)
    performance = read_json(base / "performance.json")
    with np.load(base / "external_predictions.npz", allow_pickle=False) as held:
        np.testing.assert_allclose(
            held["scores"],
            project_independent(workflow / "full/run", tests, test_layout),
            atol=1e-10,
        )
        _, density = check_pool(
            (workflow / "full/run/gmm_candidates.npz").read_bytes(),
            full,
            counts=None,
            config=model,
            heldout=held,
        )
    with np.load(workflow / "full/run/gmm_candidates.npz", allow_pickle=False) as pool:
        for name, key in (
            ("ordinary", "ordinary_aic_k"),
            ("screened", "screened_k"),
            ("bic", "descriptive_bic_k"),
        ):
            k = summary["selection"].get(key)
            p = performance[name]
            if k is None:
                if p is not None:
                    raise ValueError("Withheld selection has prediction")
                continue
            if p["k"] != k:
                raise ValueError("Performance selection differs")
            gain = density[str(k)] - density["1"]
            close(p["external_mean_logdensity_gain"], float(gain.mean()))
            close(
                p["external_mcse_conditional_on_training"],
                float(gain.std(ddof=1) / np.sqrt(len(gain))),
            )
            close(
                p["mean_assignment_entropy"],
                float(
                    -np.sum(
                        pool[f"k{k}_membership"]
                        * np.log(np.maximum(pool[f"k{k}_membership"], 1e-300))
                    )
                    / len(units)
                ),
            )
            if truth is not None:
                close(p["ari"], _ari(truth, pool[f"k{k}_labels"]))
        k = summary["selection"].get("screened_k")
        if k is not None:
            probability = pool[f"k{k}_membership"]
            labels = probability.argmax(1)
            order = sorted(range(k), key=lambda j: int(np.flatnonzero(labels == j)[0]))
            with np.load(workflow / "selected_memberships.npz", allow_pickle=False) as p:
                np.testing.assert_array_equal(p["membership"], probability[:, order])
                np.testing.assert_array_equal(p["labels"], probability[:, order].argmax(1))
            with np.load(workflow / "selected_model.npz", allow_pickle=False) as p:
                np.testing.assert_array_equal(p["component_order"], order)
                for key in ("weights", "means", "covariances"):
                    np.testing.assert_array_equal(p[key], pool[f"k{k}_{key}"][order])
            with (workflow / "selected_assignments.csv").open() as handle:
                exported = list(csv.reader(handle))[1:]
            close([r[0] for r in exported], list(units))
            np.testing.assert_allclose(
                [[float(x) for x in r[2:]] for r in exported], probability[:, order]
            )
            close([int(r[1]) for r in exported], (probability[:, order].argmax(1) + 1).tolist())
    return summary
