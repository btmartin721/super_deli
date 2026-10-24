"""Independent no-refit checks of simulation, fold leakage, prediction and selection."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from super_deli.benchmarking.aic_diagnostic_audit import check_pool, independent_raw
from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.grouped_likelihood_analysis import audit_candidates
from super_deli.benchmarking.grouped_power_audit import independent_values
from super_deli.benchmarking.grouped_study import grouped_layout
from super_deli.benchmarking.local_archives import materialize
from super_deli.benchmarking.recovery_analysis import _ari
from super_deli.benchmarking.recovery_inputs import DirectoryArchive
from super_deli.io.config import read_json
from super_deli.models.predictive_study_config import PredictiveCell, PredictiveStudyConfig


def independent_simulation(cell, n, seed):
    import numpy as np

    if cell.law == "gaussian_scores":
        return np.random.default_rng(seed).standard_normal((n, 4)) * 0.5, None, None
    if cell.law in {"independent_gaussian", "correlated_gaussian", "zero_conditional_gaussian"}:
        v, layout, _, _ = independent_raw(cell.law, n, seed)
        return v, layout, np.zeros(n, dtype=int)
    law = "complementary" if cell.law == "unequal_variance" else cell.law
    v, counts, truth = independent_values(
        SimpleNamespace(
            law=law,
            n_units=n,
            has_groups=law == "complementary",
            signal=3 if cell.law == "unequal_variance" else cell.signal,
        ),
        seed,
    )
    if cell.law == "unequal_variance":
        effect = np.array([1, 0.7, 0.5, 0.3])
        centers = np.linspace(-2, 2, 8) + 3 * np.column_stack(
            ((2 * (truth // 2) - 1)[:, None] * effect, (2 * (truth % 2) - 1)[:, None] * effect)
        )
        target = np.repeat(np.arange(n), counts)
        scale = np.array([0.5, 0.8, 1.2, 1.8])[truth]
        v = {
            "morphological": centers[target, :4]
            + (v["morphological"] - centers[target, :4]) * scale[target, None],
            "ecological": centers[:, 4:] + (v["ecological"] - centers[:, 4:]) * scale[:, None],
        }
    return v, grouped_layout(counts, "locality"), truth


def project_independent(run, values, layout):
    import numpy as np

    blocks = []
    for i, view in enumerate(layout["views"]):
        targets = np.asarray(view["target_ids"])
        means = np.stack([values[view["kind"]][targets == u].mean(0) for u in layout["unit_ids"]])
        path = Path(run) / f"view_{i:03d}/parameters"
        with (
            np.load(path / "pca.npz", allow_pickle=False) as p,
            np.load(path / "profiles.npz", allow_pickle=False) as s,
        ):
            matrix = p["components"].T / p["feature_scales"][:, None]
            offset = p["feature_means"] @ matrix + p["mean"] @ p["components"].T
            scores = means[:, p["kept_indices"]] @ matrix - offset
            blocks.append(
                (scores - s["mean"]) * math.sqrt(float(s["normalized_weight"])) / s["scale"]
            )
    with np.load(Path(run) / "fusion.npz", allow_pickle=False) as p:
        return np.column_stack(blocks) @ p["components"].T - p["mean"] @ p["components"].T


def raw_subset(values, layout, indices):
    import copy

    import numpy as np

    result = copy.deepcopy(layout)
    chosen = [layout["unit_ids"][i] for i in sorted(indices)]
    keep = set(chosen)
    result["unit_ids"] = chosen
    result["counts"] = [
        c for c, u in zip(layout["counts"], layout["unit_ids"], strict=True) if u in keep
    ]
    result["specimens"] = [s for s in layout["specimens"] if s["locality_id"] in keep]
    arrays = {}
    for view in result["views"]:
        mask = np.isin(view["target_ids"], chosen)
        arrays[view["kind"]] = values[view["kind"]][mask]
        for name in ("unit_ids", "observation_ids", "target_ids"):
            view[name] = [u for u, take in zip(view[name], mask, strict=True) if take]
    return arrays, result


def audit_search(base, prefix, model, values, layout, projection=None):
    import numpy as np

    directory = base / prefix
    if read_json(directory / "model.json") != json.loads(json.dumps(model.to_dict())):
        raise ValueError("Retained search configuration differs")
    record = read_json(directory / "record.json")
    if record["status"] != "complete":
        if not record.get("error"):
            raise ValueError("Failure lacks retained reason")
        return record, None, None
    run = directory / "run"
    candidates = read_json(run / "candidates.json")
    seed = int(np.random.SeedSequence(model.random_seed).spawn(1)[0].generate_state(1)[0])
    if record["point_seed"] != seed:
        raise ValueError("Search initialization seed differs")
    if layout is not None:
        actual = {}
        for i, view in enumerate(layout["views"]):
            a = np.load(run / f"view_{i:03d}/source_values.npy", allow_pickle=False)
            np.testing.assert_allclose(a, values[view["kind"]], atol=4e-14, rtol=4e-14)
            actual[view["kind"]] = a
        audit_candidates(
            DirectoryArchive(base), record, SimpleNamespace(model=model), actual, layout
        )
    else:
        with np.load(run / "gmm_candidates.npz", allow_pickle=False) as p:
            np.testing.assert_array_equal(p["scores"], values)
    if projection is not None:
        with np.load(directory / "validation.npz", allow_pickle=False) as held:
            np.testing.assert_allclose(held["scores"], projection, atol=1e-11, rtol=1e-11)
            choice, density = check_pool(
                (run / "gmm_candidates.npz").read_bytes(),
                candidates,
                counts=None,
                config=model,
                heldout=held,
            )
    else:
        choice, density = check_pool(
            (run / "gmm_candidates.npz").read_bytes(), candidates, counts=None, config=model
        )
    if record["selected_k"] != choice["aic_k"]:
        raise ValueError("Retained ordinary AIC choice differs")
    return record, candidates, density


def independent_selection(full, folds, n):
    valid = [c for c in full if c["status"] == "eligible"]
    accepted = []
    decisions = []
    for c in full:
        k = c["k"]
        states = [next(x["status"] for x in f["candidates"] if x["k"] == k) for f in folds]
        gain = None
        if c["status"] == "eligible" and all(s == "eligible" for s in states):
            total = math.fsum(
                float(a - b)
                for f in folds
                for a, b in zip(f["density"][str(k)], f["density"]["1"], strict=True)
            )
            gain = total / n
        passes = k == 1 or (gain is not None and gain > 0)
        reason = (
            "baseline"
            if k == 1
            else "positive_gain"
            if passes
            else "ineligible_full_or_fold"
            if gain is None
            else "nonpositive_gain"
        )
        decisions.append(
            {
                "k": k,
                "full_status": c["status"],
                "fold_statuses": states,
                "pooled_validation_gain": gain,
                "passes": passes,
                "reason": reason,
            }
        )
        if passes:
            accepted.append(c)
    return {
        "ordinary_aic_k": min(valid, key=lambda c: (c["aic"], c["k"]))["k"],
        "descriptive_bic_k": min(valid, key=lambda c: (c["bic"], c["k"]))["k"],
        "screened_k": min(accepted, key=lambda c: (c["aic"], c["k"]))["k"],
        "candidates": decisions,
    }


def _audit(task):
    import numpy as np

    source, declaration, row, raw_config, scratch = task
    config = PredictiveStudyConfig.from_dict(raw_config)
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    if any(row[k] != v for k, v in declaration.items()):
        raise ValueError("Ledger declaration differs")
    relative = f"{cell.name}/dataset_{declaration['repetition']:04d}.tar.gz"
    if row["archive"] != relative or row["manifest"] != relative + ".manifest.json":
        raise ValueError("Archive identity differs")
    with materialize(
        Path(source) / relative, row, scratch=scratch, min_free_bytes=config.min_free_bytes
    ) as base:
        if read_json(base / "declaration.json") != declaration:
            raise ValueError("Archived declaration differs")
        if any(row.get(k) != v for k, v in read_json(base / "record.json").items()):
            raise ValueError("Archived and outer ledger differ")
        values, layout, truth = independent_simulation(cell, cell.n_units, declaration["data_seed"])
        tests, test_layout, test_truth = independent_simulation(
            cell, config.n_test, declaration["test_seed"]
        )
        direct = cell.law == "gaussian_scores"
        if direct:
            with np.load(base / "inputs.npz", allow_pickle=False) as p:
                np.testing.assert_array_equal(p["training"], values)
                np.testing.assert_array_equal(p["external"], tests)
            units = [f"score_{i:05d}" for i in range(len(values))]
        else:
            layout = json.loads(json.dumps(layout))
            if read_json(base / "generating_layout.json") != layout:
                raise ValueError("Generating identities/counts differ")
            with np.load(base / "values.npz", allow_pickle=False) as p:
                for k in values:
                    np.testing.assert_allclose(p[k], values[k], atol=4e-14, rtol=4e-14)
                    values[k] = p[k].copy()
            with np.load(base / "external_values.npz", allow_pickle=False) as p:
                for k in tests:
                    np.testing.assert_allclose(p[k], tests[k], atol=4e-14, rtol=4e-14)
                    tests[k] = p[k].copy()
            with np.load(base / "truth.npz", allow_pickle=False) as p:
                np.testing.assert_array_equal(p["training"], truth)
                np.testing.assert_array_equal(p["external"], test_truth)
            # Reconstruct held-out identifiers without using the writer's identity helper.
            test_layout = json.loads(json.dumps(test_layout))
            test_layout["unit_ids"] = ["heldout_" + u for u in test_layout["unit_ids"]]
            for s in test_layout["specimens"]:
                for k in s:
                    s[k] = "heldout_" + s[k]
            for view in test_layout["views"]:
                for k in ("observation_ids", "unit_ids", "target_ids"):
                    view[k] = ["heldout_" + u for u in view[k]]
            if read_json(base / "external_layout.json") != test_layout:
                raise ValueError("External identities differ")
            units = layout["unit_ids"]
        workflow = base / "workflow"
        raw = read_json(workflow / "configuration.json")
        screen = {**config.screening.to_dict(), "random_seed": declaration["split_seed"]}
        if (
            raw["predictive_model"] != json.loads(json.dumps(config.model.to_dict()))
            or raw["screening"] != screen
        ):
            raise ValueError("Screen/model config differs")
        full_record, full, _ = audit_search(workflow, "full", config.model, values, layout)
        if full_record["status"] != "complete":
            if (
                row["selection"]["status"] != "withheld"
                or row["selection"].get("screened_k") is not None
            ):
                raise ValueError("Failed full fit has a screened result")
            return row
        if not direct and read_json(workflow / "layout.json") != layout:
            raise ValueError("Ingested source layout differs")
        shuffled = np.random.default_rng(declaration["split_seed"]).permutation(len(units))
        parts = np.array_split(shuffled, config.screening.n_folds)
        assignments = read_json(workflow / "fold_assignments.json")
        folds = []
        records = [full_record]
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
            if assignments[i] != expected:
                raise ValueError("Fold membership/identity differs")
            prefix = f"fold_{i:03d}"
            model = replace(
                config.model,
                random_seed=derived_seed(config.model.random_seed, "predictive_fold", i, "model")
                % 2**32,
            )
            if direct:
                training, training_layout = values[train], None
                projection = values[valid]
            else:
                training, training_layout = raw_subset(values, layout, train)
                validation, validation_layout = raw_subset(values, layout, valid)
                if (
                    read_json(workflow / prefix / "training_layout.json") != training_layout
                    or read_json(workflow / prefix / "validation_layout.json") != validation_layout
                ):
                    raise ValueError("Whole-locality raw subset differs")
                projection = (
                    project_independent(workflow / prefix / "run", validation, validation_layout)
                    if (workflow / prefix / "validation.npz").exists()
                    else None
                )
            record, candidates, density = audit_search(
                workflow, prefix, model, training, training_layout, projection
            )
            records.append(record)
            validation_record = read_json(workflow / prefix / "validation_record.json")
            if any(validation_record[k] != v for k, v in expected.items()):
                raise ValueError("Validation record disagrees with split")
            folds.append(
                {
                    "status": validation_record["status"],
                    "candidates": candidates,
                    "density": density,
                }
            )
        selection = row["selection"]
        if all(f["status"] == "complete" for f in folds):
            expected = independent_selection(full, folds, len(units))
            if selection["status"] != "complete":
                raise ValueError("Complete fits unexpectedly withheld")
            for k in ("ordinary_aic_k", "descriptive_bic_k", "screened_k"):
                if expected[k] != selection[k]:
                    raise ValueError("Screened/ordinary selection differs")
            for a, b in zip(selection["candidates"], expected["candidates"], strict=True):
                for k in a:
                    if k == "pooled_validation_gain" and a[k] is not None:
                        np.testing.assert_allclose(a[k], b[k], atol=1e-10, rtol=1e-10)
                    elif a[k] != b[k]:
                        raise ValueError("Screen candidate decision differs")
        elif selection["status"] != "withheld" or selection.get("screened_k") is not None:
            raise ValueError("Incomplete screening is not withheld")
        projection = (
            tests if direct else project_independent(workflow / "full/run", tests, test_layout)
        )
        with np.load(base / "external_predictions.npz", allow_pickle=False) as external:
            np.testing.assert_allclose(external["scores"], projection, atol=1e-11, rtol=1e-11)
            _, density = check_pool(
                (workflow / "full/run/gmm_candidates.npz").read_bytes(),
                full,
                counts=None,
                config=config.model,
                heldout=external,
            )
        with np.load(workflow / "full/run/gmm_candidates.npz", allow_pickle=False) as pool:
            for name, key in (
                ("ordinary", "ordinary_aic_k"),
                ("screened", "screened_k"),
                ("bic", "descriptive_bic_k"),
            ):
                k = selection.get(key)
                metric = row["performance"][name]
                if k is None:
                    if metric is not None:
                        raise ValueError("Unavailable choice has performance")
                    continue
                if metric["k"] != k:
                    raise ValueError("Performance choice differs")
                gain = density[str(k)] - density["1"]
                np.testing.assert_allclose(
                    metric["external_mean_logdensity_gain"], gain.mean(), atol=1e-10
                )
                np.testing.assert_allclose(
                    metric["external_mcse_conditional_on_training"],
                    gain.std(ddof=1) / np.sqrt(len(gain)),
                    atol=1e-10,
                )
                if cell.law in {"complementary", "unequal_variance"}:
                    np.testing.assert_allclose(
                        metric["ari"], _ari(truth, pool[f"k{k}_labels"]), atol=1e-12
                    )
            k = selection.get("screened_k")
            if k is not None:
                probability = pool[f"k{k}_membership"]
                labels = probability.argmax(1)
                order = sorted(range(k), key=lambda j: int(np.flatnonzero(labels == j)[0]))
                with np.load(workflow / "selected_model.npz", allow_pickle=False) as p:
                    np.testing.assert_array_equal(p["component_order"], order)
                    for key in ("weights", "means", "covariances"):
                        np.testing.assert_array_equal(p[key], pool[f"k{k}_{key}"][order])
                with np.load(workflow / "selected_memberships.npz", allow_pickle=False) as p:
                    np.testing.assert_array_equal(p["membership"], probability[:, order])
                    np.testing.assert_array_equal(p["labels"], probability[:, order].argmax(1))
                with (workflow / "selected_assignments.csv").open() as h:
                    exported = list(csv.reader(h))[1:]
                if [r[0] for r in exported] != units:
                    raise ValueError("Selected assignment IDs differ")
                np.testing.assert_allclose(
                    [[float(x) for x in r[2:]] for r in exported], probability[:, order]
                )
            elif (workflow / "selected_model.npz").exists():
                raise ValueError("Withheld result exported a selected model")
        for key, value in {
            "attempted_fits": len(records),
            "successful_fits": sum(r["status"] == "complete" for r in records),
            "failed_fits": sum(r["status"] != "complete" for r in records),
        }.items():
            if row[key] != value:
                raise ValueError("Search accounting differs")
        return row


def audit_observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _audit(task)
