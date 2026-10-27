"""Fit-free independent grouped geometry, density, selection and denominator audits."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.grouped_analysis_study import compare_replay, model_arms, tasks
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


class PointGeometryPaths:
    """Read native score storage through the established independent geometry audit.

    Ordinary BIC stores winning scores in mixture.npz; AIC additionally exports
    gmm_candidates.npz. This read adapter does not modify or synthesize evidence.
    """

    def __init__(self, run):
        self.run = Path(run)

    def __truediv__(self, name):
        if name == "gmm_candidates.npz" and not (self.run / name).exists():
            return self.run / "mixture.npz"
        return self.run / name


def independent_raw(task):
    from super_deli.benchmarking.grouped_power_audit import independent_values
    from super_deli.benchmarking.predictive_audit import independent_simulation

    if task["case"] == "unequal_variance":
        values, layout, labels = independent_simulation(
            SimpleNamespace(law="unequal_variance", signal=3), task["n"], task["data_seed"]
        )
        return values, layout["counts"], labels
    return independent_values(
        SimpleNamespace(
            law=task["case"],
            n_units=task["n"],
            has_groups=task["case"] == "complementary",
            signal=3,
        ),
        task["data_seed"],
    )


def audit_winner(run, model):
    import numpy as np
    from scipy.special import logsumexp

    from super_deli.benchmarking.predictive_density_audit import log_density

    with np.load(run / "mixture.npz", allow_pickle=False) as p:
        joint = np.column_stack(
            [
                np.log(w) + log_density(p["scores"], np.ones(1), m[None], c[None])
                for w, m, c in zip(p["weights"], p["means"], p["covariances"], strict=True)
            ]
        )
        logs = logsumexp(joint, axis=1)
        probability = np.exp(joint - logs[:, None])
        np.testing.assert_allclose(p["membership"], probability, atol=1e-10)
        np.testing.assert_array_equal(p["labels"], probability.argmax(1))
        k = len(p["weights"])
        n, d = p["scores"].shape
    eligible = [c for c in read_json(run / "candidates.json") if c["status"] == "eligible"]
    chosen = min(eligible, key=lambda c: (c[model.selection_criterion], c["k"]))
    assert chosen["k"] == k
    q = k - 1 + k * d + (1 if model.covariance_type == "tied" else k) * d * (d + 1) // 2
    np.testing.assert_allclose(
        chosen["bic"], -2 * logs.sum() + q * np.log(n), atol=1e-7, rtol=1e-10
    )
    if "aic" in chosen:
        np.testing.assert_allclose(chosen["aic"], -2 * logs.sum() + 2 * q, atol=1e-7, rtol=1e-10)
        np.testing.assert_allclose(chosen["log_likelihood"], logs.sum(), atol=1e-7, rtol=1e-10)
    return k


def audit_screen(point, values, layout, arm):
    import numpy as np

    from super_deli.benchmarking.predictive_audit import (
        independent_selection,
        project_independent,
        raw_subset,
    )
    from super_deli.benchmarking.predictive_bootstrap_audit import audit_geometry, audit_pool
    from super_deli.benchmarking.scaled_fold_audit import selection_check

    n = len(layout["unit_ids"])
    model = arm.model
    assignments = read_json(point / "fold_assignments.json")
    shuffled = np.random.default_rng(arm.screen.random_seed).permutation(n)
    splits = [sorted(map(int, s)) for s in np.array_split(shuffled, arm.screen.n_folds)]
    assert [a["validation_indices"] for a in assignments] == splits
    audit_geometry(point / "full/run", values, layout, np.arange(n), model)
    full, _ = audit_pool(point / "full/run", model, list(range(n)))
    folds = []
    checked = 1
    for a in assignments:
        i = a["fold"]
        train = a["training_indices"]
        test = a["validation_indices"]
        assert not set(train) & set(test) and set(train) | set(test) == set(range(n))
        trained = replace(
            model,
            random_seed=derived_seed(model.random_seed, "predictive_fold", i, "model") % 2**32,
        )
        fold = point / f"fold_{i:03d}"
        assert read_json(fold / "model.json") == json.loads(json.dumps(trained.to_dict()))
        record = read_json(fold / "record.json")
        checked += 1
        if record["status"] != "complete":
            assert record.get("error")
            continue
        audit_geometry(fold / "run", values, layout, np.asarray(train), trained)
        testvalues, testlayout = raw_subset(values, layout, test)
        projected = project_independent(fold / "run", testvalues, testlayout)
        candidates, density = audit_pool(fold / "run", trained, train, projected)
        with np.load(fold / "validation.npz") as p:
            np.testing.assert_allclose(p["scores"], projected, atol=1e-11, rtol=1e-11)
            for key, logs in density.items():
                np.testing.assert_allclose(p["k" + key], logs, atol=1e-8, rtol=1e-10)
        folds.append({"candidates": candidates, "density": density})
    actual = read_json(point / "summary.json")["selection"]
    if len(folds) != arm.screen.n_folds:
        assert actual["status"] != "complete"
        return checked
    expected = independent_selection(full, folds, n)
    selection_check(expected, actual)
    if actual["status"] == "complete":
        with (
            np.load(point / "full/run/gmm_candidates.npz") as p,
            np.load(point / "selected_model.npz") as chosen,
            np.load(point / "selected_memberships.npz") as membership,
        ):
            k = actual["screened_k"]
            order = chosen["component_order"]
            for key in ("weights", "means", "covariances"):
                np.testing.assert_array_equal(chosen[key], p[f"k{k}_{key}"][order])
            np.testing.assert_array_equal(
                membership["labels"], p[f"k{k}_membership"][:, order].argmax(1)
            )
    return checked


def audit_observation(directory, task):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.benchmarking.predictive_bootstrap_audit import (
        audit_geometry,
        audit_pool,
        audit_stability,
    )
    from super_deli.benchmarking.scaled_fold_audit import audit_workflow

    d = Path(directory)
    out = d / "analysis"
    summary = read_json(out / "summary.json")
    for name, digest in summary["artifacts"].items():
        assert fingerprint_file(out / name).sha256 == digest, name
    record = read_json(d / "validation.json")
    assert {k: record["task"][k] for k in task} == task
    manifest = read_json(out / "manifest.json")
    config = model_arms(task["draws"])
    assert manifest["analysis"] == json.loads(json.dumps(config.to_dict()))
    values, counts, truth = independent_raw(task)
    layout = read_json(d / "layout.json")
    assert list(counts) == layout["counts"] == [g["morphology_count"] for g in summary["groups"]]
    assert all(g["ecology_count"] == 1 for g in summary["groups"])
    for i, view in enumerate(layout["views"]):
        table = read_json(out / f"inputs/ingestion/modality_{i:03d}/table.json")
        np.testing.assert_allclose(values[view["kind"]], table["values"], atol=4e-14, rtol=4e-14)
        for key in ("unit_ids", "observation_ids"):
            assert table[key] == view[key]
    if (d / "truth.npz").exists():
        with np.load(d / "truth.npz") as p:
            np.testing.assert_array_equal(p["labels"], truth)
    n = task["n"]
    points = {}
    searches = 0
    failures = []
    for arm, row in zip(config.arms, summary["models"], strict=True):
        base = out / "arms" / arm.name
        if row["status"] != "complete":
            failures.append({"arm": arm.name, "status": row["status"], "error": row["error"]})
        if arm.kind == "ordinary":
            if not (base / "summary.json").exists():
                continue
            audit_geometry(PointGeometryPaths(base), values, layout, np.arange(n), arm.model)
            if (base / "gmm_candidates.npz").exists():
                audit_pool(base, arm.model, list(range(n)))
            assert audit_winner(base, arm.model) == row["selected_k"]
            searches += 1
        else:
            point = base / "point" if arm.bootstrap is not None else base
            searches += audit_screen(point, values, layout, arm)
            if arm.bootstrap is not None:
                result = read_json(base / "summary.json")
                assignments = read_json(point / "fold_assignments.json")
                for retained, seed in zip(
                    result["replicates"], arm.bootstrap.draw_seeds(), strict=True
                ):
                    draw = np.random.default_rng(seed).integers(n, size=n)
                    assert (
                        retained["draw_seed"] == seed
                        and retained["training_indices"] == draw.tolist()
                    )
                    assert retained["multiplicities"] == np.bincount(draw, minlength=n).tolist()
                    checked, labels = audit_workflow(
                        base / retained["prefix"],
                        values,
                        layout,
                        draw,
                        arm.model,
                        assignments,
                        False,
                    )
                    searches += checked
                    if retained["status"] == "complete":
                        np.testing.assert_array_equal(labels, retained["labels"])
                    else:
                        assert retained.get("error")
                audit_stability(base, result)
        if (base / "comparison-point.json").exists():
            points[arm.name] = read_json(base / "comparison-point.json")
        if task["replay"]:
            compare_replay(arm, base, d / "replays" / arm.name)
            searches += 6 if arm.kind == "predictive" else 1
    for row in summary["comparisons"]:
        if row["status"] == "unavailable":
            continue
        a, b = points[row["left"]], points[row["right"]]
        assert row["joint_resolved"] == n and row["joint_pairs"] == n * (n - 1) // 2
        np.testing.assert_allclose(
            row["adjusted_rand"], adjusted_rand_score(a["labels"], b["labels"]), atol=1e-12
        )
        discordant = sum(
            (a["labels"][i] == a["labels"][j]) != (b["labels"][i] == b["labels"][j])
            for i, j in combinations(range(n), 2)
        )
        assert row["discordant_pairs"] == discordant
    return {
        "key": task["key"],
        "searches": searches,
        "failures": failures,
        "models": [
            {
                "case": task["case"],
                "n": n,
                "dataset": task["dataset"],
                **{
                    k: r.get(k)
                    for k in (
                        "name",
                        "status",
                        "selected_k",
                        "ordinary_k",
                        "screen_status",
                        "stability_status",
                    )
                },
            }
            for r in summary["models"]
        ],
    }


def audit_study(directory, output_file, workers=1):
    from concurrent.futures import ProcessPoolExecutor

    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four audit workers")
    root = Path(directory)
    phase = read_json(root / "plan.json")["phase"]
    jobs = [(str(root / t["key"]), t) for t in tasks(phase)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_audit_job, jobs))
    total = sum(r["searches"] for r in rows)
    expected = 364 if phase == "preflight" else 2568
    failures = [r for r in rows if r["failures"]]
    result = {
        "status": "passed" if total == expected and not failures else "audited_incomplete",
        "phase": phase,
        "observations": len(rows),
        "searches": total,
        "planned_searches": expected,
        "failures": failures,
        "models": [m for r in rows for m in r["models"]],
        "comparisons": 15 * len(rows),
        "scope": (
            "Independent raw/PCA/density/fold/selection/multiplicity/denominator "
            "checks; no new fits"
        ),
    }
    write_json(result, output_file)
    return result


def _audit_job(args):
    return audit_observation(*args)
