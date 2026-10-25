"""Independent no-refit audits of fold exposure, fitted maps, gates and stability."""

import itertools
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from super_deli.benchmarking.local_archives import materialize
from super_deli.benchmarking.predictive_audit import (
    independent_selection,
    independent_simulation,
    project_independent,
)
from super_deli.benchmarking.predictive_bootstrap_audit import (
    audit_geometry,
    audit_pool,
    audit_stability,
)
from super_deli.benchmarking.predictive_density_audit import log_density
from super_deli.benchmarking.recovery_analysis import _ari
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_study_config import PredictiveCell
from super_deli.models.scaled_fold_study import ScaledFoldStudy


def exposure(model, indices, train, scaled):
    """Use divmod rather than the production ceiling expression."""
    if not scaled:
        return None, model.min_cluster_size
    u, t = len(set(map(int, indices))), len(set(map(int, train)))
    q, r = divmod(model.min_cluster_size * t, u)
    minimum = max(2, q + bool(r)) if 0 < t < u else None
    result = dict(
        schema_version=1,
        policy="proportional_distinct",
        full_distinct=u,
        training_distinct=t,
        full_minimum=model.min_cluster_size,
        training_minimum=minimum,
        conditioning="Sampling IDs only; copies do not add independent localities",
    )
    return result, minimum


def selection_check(expected, actual):
    import numpy as np

    for key in ("ordinary_aic_k", "descriptive_bic_k", "screened_k"):
        if actual[key] != expected[key]:
            raise ValueError("Independent selection differs")
    for a, b in zip(expected["candidates"], actual["candidates"], strict=True):
        for key in ("k", "reason", "passes", "fold_statuses"):
            if a[key] != b[key]:
                raise ValueError("Independent predictive gate differs")
        if a["pooled_validation_gain"] is not None:
            np.testing.assert_allclose(
                a["pooled_validation_gain"], b["pooled_validation_gain"], atol=1e-8, rtol=1e-10
            )


def prediction_check(directory, values, layout, k, point):
    import numpy as np
    from scipy.special import logsumexp

    scores = project_independent(directory / "full/run", values, layout)
    with (
        np.load(directory / "selected_model.npz", allow_pickle=False) as p,
        np.load(directory / "full/run/gmm_candidates.npz", allow_pickle=False) as pool,
    ):
        if str(p["covariance_type"]) != "tied":
            raise ValueError("Selected covariance family differs")
        for key in ("weights", "means", "covariances"):
            np.testing.assert_array_equal(p[key], pool[f"k{k}_{key}"][p["component_order"]])
        joint = np.column_stack(
            [
                np.log(a) + log_density(scores, np.ones(1), m[None], c[None])
                for a, m, c in zip(p["weights"], p["means"], p["covariances"], strict=True)
            ]
        )
    logs = logsumexp(joint, axis=1)
    prob = np.exp(joint - logs[:, None])
    labels = prob.argmax(1)
    with np.load(
        directory / ("selected_memberships.npz" if point else "original_predictions.npz"),
        allow_pickle=False,
    ) as p:
        np.testing.assert_allclose(p["membership"], prob, atol=1e-10)
        np.testing.assert_array_equal(p["labels"], labels)
        if not point:
            np.testing.assert_allclose(p["scores"], scores, atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(p["log_density"], logs, atol=1e-8, rtol=1e-10)
    return labels


def audit_workflow(directory, values, layout, indices, model, assignments, scaled, point=False):
    import numpy as np

    from super_deli.workflows.predictive import fold_model

    directory = Path(directory)
    indices = np.asarray(indices, dtype=int)
    if (directory / "summary.json").exists():
        summary = read_json(directory / "summary.json")
    else:
        failure_path = directory.parent / (directory.name + ".json")
        if not failure_path.exists():
            raise ValueError("Missing workflow summary and failure record")
        failure = read_json(failure_path)
        if failure["status"] == "complete" or not failure.get("error"):
            raise ValueError("Unresolved workflow lacks retained failure evidence")
        summary = dict(selection=dict(status="withheld", error=failure["error"]))
        if scaled:
            summary["fold_size_policy"] = dict(schema_version=1, policy="proportional_distinct")
    if scaled and summary.get("fold_size_policy") != {
        "schema_version": 1,
        "policy": "proportional_distinct",
    }:
        raise ValueError("Policy provenance differs")
    if not scaled and "fold_size_policy" in summary:
        raise ValueError("Legacy summary gained a policy field")
    full_record = (
        read_json(directory / "full/record.json")
        if (directory / "full/record.json").exists()
        else None
    )
    if full_record is None or full_record["status"] != "complete":
        if summary["selection"]["status"] == "complete":
            raise ValueError("Full failure became available")
        return int(full_record is not None), None
    if read_json(directory / "full/model.json") != json.loads(json.dumps(model.to_dict())):
        raise ValueError("Full model or minimum changed")
    audit_geometry(directory / "full/run", values, layout, indices, model)
    full, _ = audit_pool(directory / "full/run", model, indices.tolist())
    folds = []
    checked = 1
    failed = False
    if not point:
        retained = read_json(directory / "folds.json")
    for assignment in assignments:
        i = assignment["fold"]
        prefix = f"fold_{i:03d}"
        target = directory / prefix
        keep = set(assignment["validation_indices"])
        train = [j for j, u in enumerate(indices) if u not in keep]
        test = [j for j, u in enumerate(indices) if u in keep]
        if set(indices[train]) & set(indices[test]):
            raise ValueError("Original-ID fold leakage")
        path = target / "record.json"
        if not path.exists():
            failed = True
            continue
        record = read_json(path)
        checked += 1
        actual = (
            read_json(target / "validation_record.json")
            if point
            else next(v for v in retained if v["fold"] == i)
        )
        for key, value in (("training_indices", train), ("validation_indices", test)):
            if actual[key] != value:
                raise ValueError("Fold occurrence partition differs")
        if not point:
            for key, value in (
                ("original_training_indices", indices[train].tolist()),
                ("original_validation_indices", indices[test].tolist()),
            ):
                if actual[key] != value:
                    raise ValueError("Fold identity or multiplicity differs")
        support, minimum = exposure(model, indices, indices[train], scaled)
        if scaled:
            saved = read_json(target / "fold_support.json")
            if minimum is None:
                if saved["training_minimum"] is not None or not saved.get("error"):
                    raise ValueError("Empty fold threshold was repaired")
            elif saved != support:
                raise ValueError("Independent fold threshold differs")
            if actual.get("fold_support") != saved:
                raise ValueError("Fold support ledger differs")
        elif (target / "fold_support.json").exists():
            raise ValueError("Legacy evidence changed")
        if record["status"] != "complete" or actual["status"] != "complete":
            if not record.get("error", actual.get("error")):
                raise ValueError("Failure lacks explanation")
            failed = True
            if record["status"] != "complete":
                continue
        trained = fold_model(model, i)
        if read_json(target / "model.json") != json.loads(json.dumps(trained.to_dict())):
            raise ValueError("Fold optimizer or size settings differ")
        audit_geometry(target / "run", values, layout, indices[train], trained)
        scores = project_independent(target / "run", values, layout)[indices[test]]
        effective = SimpleNamespace(
            min_cluster_size=minimum, covariance_type=trained.covariance_type
        )
        if scaled:
            names = (
                "max_k",
                "n_init",
                "max_iter",
                "tol",
                "reg_covar",
                "selection_criterion",
                "max_working_bytes",
            )
            expected_settings = dict(
                schema_version=1,
                workflow="predictive_fold_mixture_v1",
                min_cluster_size=minimum,
                **{name: getattr(trained, name) for name in names},
            )
            if read_json(target / "run/mixture_settings.json") != expected_settings:
                raise ValueError("Effective mixture settings differ")
        candidates, density = audit_pool(target / "run", effective, indices[train].tolist(), scores)
        if actual["status"] == "complete":
            with np.load(target / "validation.npz", allow_pickle=False) as p:
                np.testing.assert_allclose(p["scores"], scores, atol=1e-11, rtol=1e-11)
                for key, logs in density.items():
                    np.testing.assert_allclose(p["k" + key], logs, atol=1e-8, rtol=1e-10)
            folds.append(dict(candidates=candidates, density=density))
    if failed:
        if summary["selection"]["status"] == "complete":
            raise ValueError("Numerical failure became a selection")
        return checked, None
    if summary["selection"]["status"] != "complete":
        if not summary["selection"].get("error"):
            raise ValueError("Withheld outcome lacks a reason")
        return checked, None
    expected = independent_selection(full, folds, len(indices))
    selection_check(expected, summary["selection"])
    return checked, prediction_check(directory, values, layout, expected["screened_k"], point)


def changes(a, b):
    """Independent exhaustive matching for at most four components."""
    if a is None or b is None:
        return None
    left, right = sorted(set(a.tolist())), sorted(set(b.tolist()))
    n = max(len(left), len(right))
    table = [[0] * n for _ in range(n)]
    for x, y in zip(a, b, strict=True):
        table[left.index(x)][right.index(y)] += 1
    return len(a) - max(
        sum(table[i][j] for i, j in enumerate(order)) for order in itertools.permutations(range(n))
    )


def common_audit(base, result, common, labels, k, scratch):
    from super_deli.workflows.predictive import save_arrays

    # Adapt immutable reference evidence to the established independent auditor.
    with TemporaryDirectory(prefix="scaled-common-audit-", dir=scratch) as name:
        target = Path(name)
        (target / "point").mkdir()
        save_arrays(target / "point/selected_memberships.npz", labels=labels)
        shutil.copyfile(base / "common_stability.npz", target / "stability.npz")
        view = dict(result, stability=common, point=dict(selection=dict(screened_k=k)))
        audit_stability(target, view)


def paired_parameters(base, results):
    import numpy as np

    names = ["point"] + [r["prefix"] for r in results["legacy"]["replicates"]]
    comparisons = 0
    for name in names:
        choices = []
        for arm in ("legacy", "scaled"):
            path = base / arm / name / "summary.json"
            choices.append(read_json(path)["selection"] if path.exists() else None)
        if all(v is not None and v["status"] == "complete" for v in choices):
            allowed = [{v["k"] for v in choice["candidates"] if v["passes"]} for choice in choices]
            if not allowed[0] <= allowed[1]:
                raise ValueError("Scaled eligibility removed a passing legacy candidate")
            candidates = read_json(base / "legacy" / name / "full/run/candidates.json")
            values = {v["k"]: v.get("aic") for v in candidates}
            if values[choices[1]["screened_k"]] > values[choices[0]["screened_k"]] + 1e-9:
                raise ValueError("Expanded candidate set increased selected full-data AIC")
        for prefix in ["full"] + [f"fold_{i:03d}" for i in range(5)]:
            a = base / "legacy" / name / prefix / "run/gmm_candidates.npz"
            b = base / "scaled" / name / prefix / "run/gmm_candidates.npz"
            if not a.exists() or not b.exists():
                continue
            with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
                # Thresholds change eligibility/feasibility, never input scores or
                # the fitted parameters of candidates feasible under both rules.
                for key in set(x.files) & set(y.files):
                    np.testing.assert_array_equal(
                        x[key],
                        y[key],
                        err_msg=f"Paired candidate values changed at {name}/{prefix}/{key}",
                    )
            comparisons += 1
    return comparisons


def _audit_observation(task):
    import numpy as np

    from super_deli.models.predictive_selection import fold_assignments

    source, raw, declaration, row, scratch = task
    c = ScaledFoldStudy.from_dict(raw)
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    with materialize(
        Path(source) / row["archive"], row, scratch=scratch, min_free_bytes=c.min_free_bytes
    ) as base:
        values, layout, truth = independent_simulation(cell, cell.n_units, declaration["data_seed"])
        layout = json.loads(json.dumps(layout))
        if read_json(base / "generating_layout.json") != layout:
            raise ValueError("Raw layout differs")
        with np.load(base / "values.npz", allow_pickle=False) as p:
            for key, value in values.items():
                np.testing.assert_allclose(p[key], value, atol=4e-14, rtol=4e-14)
        with np.load(base / "truth.npz", allow_pickle=False) as p:
            np.testing.assert_array_equal(p["labels"], truth)
        assignments = fold_assignments(
            layout["unit_ids"], replace(c.screening, random_seed=declaration["split_seed"])
        )
        seeds = PredictiveBootstrapConfig(c.n_bootstrap, declaration["bootstrap_seed"]).draw_seeds()
        evaluated = read_json(base / "evaluation.json")
        reference = None
        checked = 0
        results = {}
        for arm in ("legacy", "scaled"):
            directory = base / arm
            result = read_json(directory / "summary.json")
            results[arm] = result
            point = result["point"]
            point_labels = None
            if (directory / "point/summary.json").exists():
                count, point_labels = audit_workflow(
                    directory / "point",
                    values,
                    layout,
                    np.arange(cell.n_units),
                    c.model,
                    assignments,
                    arm == "scaled",
                    True,
                )
                checked += count
            if point["selection"]["status"] == "complete":
                if read_json(directory / "point/fold_assignments.json") != assignments:
                    raise ValueError("Original folds changed")
            if arm == "legacy":
                reference = point_labels
            actual = evaluated[arm]
            if actual["point_k"] != point["selection"].get("screened_k") or actual[
                "point_common_assignment_changes"
            ] != changes(reference, point_labels):
                raise ValueError("Point evaluation differs")
            group_truth = cell.law in {"complementary", "unequal_variance"}
            if point_labels is not None and group_truth:
                np.testing.assert_allclose(
                    actual["point_truth_ari"], _ari(truth, point_labels), atol=1e-12
                )
            for i, record in enumerate(result["replicates"]):
                draw = np.random.default_rng(seeds[i]).integers(cell.n_units, size=cell.n_units)
                if (
                    record["draw_seed"] != seeds[i]
                    or record["training_indices"] != draw.tolist()
                    or record["multiplicities"]
                    != np.bincount(draw, minlength=cell.n_units).tolist()
                ):
                    raise ValueError("Paired sampling identities or seeds differ")
                oob = np.flatnonzero(np.bincount(draw, minlength=cell.n_units) == 0)
                if record["oob_indices"] != oob.tolist():
                    raise ValueError("OOB identities differ")
                count, labels = audit_workflow(
                    directory / record["prefix"],
                    values,
                    layout,
                    draw,
                    c.model,
                    assignments,
                    arm == "scaled",
                )
                checked += count
                if (labels is not None) != (record["status"] == "complete"):
                    raise ValueError("Failure converted into a valid outcome")
                e = actual["draws"][i]
                if labels is not None:
                    if record["labels"] != labels.tolist():
                        raise ValueError("Original-unit partition differs")
                    if e["own_assignment_changes"] != changes(point_labels, labels) or e[
                        "common_assignment_changes"
                    ] != changes(reference, labels):
                        raise ValueError("Matched assignment changes differ")
                    if group_truth:
                        np.testing.assert_allclose(e["truth_ari"], _ari(truth, labels), atol=1e-12)
                        if len(oob) > 1:
                            np.testing.assert_allclose(
                                e["oob_truth_ari"], _ari(truth[oob], labels[oob]), atol=1e-12
                            )
            if point_labels is not None and len(result["replicates"]) == c.n_bootstrap:
                audit_stability(directory, result)
            for key in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits"):
                if result[key] != row["arms"][arm][key]:
                    raise ValueError("Outer search counts differ")
        paired = paired_parameters(base, results)
        common = read_json(base / "common_stability.json")
        if (base / "common_stability.npz").exists():
            common_audit(
                base,
                results["scaled"],
                common,
                reference,
                results["legacy"]["point"]["selection"]["screened_k"],
                scratch,
            )
        elif common["status"] != "unavailable" or not common.get("error"):
            raise ValueError("Common reference loss unreported")
        if checked != row["attempted_fits"]:
            raise ValueError("Audited search denominator differs")
        return dict(
            cell=row["cell"],
            repetition=row["repetition"],
            audited_searches=checked,
            paired_search_parameter_comparisons=paired,
            **{
                k: row[k]
                for k in (
                    "successful_fits",
                    "failed_fits",
                    "unattempted_fits",
                    "successful_draws",
                    "failed_draws",
                    "unattempted_draws",
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
    c = ScaledFoldStudy.from_dict(read_json(source / "configuration.json"))
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    lookup = {(r["cell"], r["repetition"]): r for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(c.plan()):
        raise ValueError("Missing or duplicate observations")
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
