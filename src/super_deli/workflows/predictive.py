"""Opt-in grouped AIC screening with complete training-fold pipeline refits."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.config import derived_seed
from super_deli.io.config import read_json, write_json
from super_deli.io.predictive_inputs import load_sources, project_exported, subset_raw
from super_deli.models.predictive_config import PredictiveScreenConfig


def save_arrays(path, **arrays):
    import numpy as np

    with Path(path).open("xb") as handle:
        np.savez(handle, **arrays)


def fold_model(model, i):
    return replace(
        model, random_seed=derived_seed(model.random_seed, "predictive_fold", i, "model") % 2**32
    )


def fold_with_support(model, i, full_ids, training_ids, policy):
    trained = fold_model(model, i)
    if policy is None:
        return trained, None
    evidence = policy.evidence(model.min_cluster_size, full_ids, training_ids)
    return trained, evidence


def checked_policy(policy, model, screening):
    if policy is None:
        return None
    from super_deli.models.predictive_fold_size import PredictiveFoldSizePolicy

    if not isinstance(policy, PredictiveFoldSizePolicy):
        raise ValueError("An explicit PredictiveFoldSizePolicy is required")
    policy = PredictiveFoldSizePolicy.from_dict(policy.to_dict())
    policy.validate_model(model, screening)
    return policy


def raw_search(base, prefix, values, layout, model):
    from super_deli.models.grouped_null import write_raw_views

    directory = Path(base) / prefix
    directory.mkdir()
    dataset = write_raw_views(values, layout, directory / "raw")
    return dataset_search(base, prefix, dataset, directory / "raw", model)


def dataset_search(base, prefix, dataset, source, model):
    from super_deli.workflows.grouped_likelihood import point_fitter
    from super_deli.workflows.integrative import run_integrative

    directory = Path(base) / prefix
    directory.mkdir(exist_ok=True)
    write_json(model.to_dict(), directory / "model.json")
    record = point_fitter(directory, dataset, source, prefix, model=model, runner=run_integrative)
    write_json(record, directory / "record.json")
    return record


def score_search(base, prefix, scores, units, model):
    import numpy as np

    from super_deli.models.likelihood_selection import candidate_statistics
    from super_deli.models.mixture import select_gaussian_mixture

    directory = Path(base) / prefix
    directory.mkdir()
    run = directory / "run"
    run.mkdir()
    write_json(model.to_dict(), directory / "model.json")
    seed = int(np.random.SeedSequence(model.random_seed).spawn(1)[0].generate_state(1)[0])
    try:
        fitted = select_gaussian_mixture(
            scores, model, seed, covariance_type="full", unit_indices=np.arange(len(scores))
        )
        save_arrays(run / "gmm_candidates.npz", scores=scores, **fitted.candidate_arrays)
        write_json(fitted.candidates, run / "candidates.json")
        record = {
            "status": "complete",
            "prefix": prefix,
            "point_seed": seed,
            "n_retained": len(units),
            **candidate_statistics(fitted.candidates),
        }
        write_json(
            {"unit_ids": units, "selected_k": fitted.selected_k, "point_seed": seed},
            run / "summary.json",
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record = {"status": "failed", "prefix": prefix, "error": str(exc)}
    write_json(record, directory / "record.json")
    return record


def predict_pool(run, scores, destination):
    import numpy as np

    from super_deli.models.predictive_selection import gaussian_predictions

    logs = {}
    candidates = read_json(Path(run) / "candidates.json")
    with np.load(Path(run) / "gmm_candidates.npz", allow_pickle=False) as pool:
        for c in candidates:
            if c["status"] != "insufficient_samples":
                logs[str(c["k"])] = gaussian_predictions(scores, pool, c["k"])[0]
    save_arrays(destination, scores=scores, **{"k" + k: v for k, v in logs.items()})
    return {k: v.tolist() for k, v in logs.items()}


def write_selected(base, selection, units, covariance_type="full"):
    import numpy as np

    k = selection["screened_k"]
    if k is None:
        return
    with np.load(Path(base) / "full/run/gmm_candidates.npz", allow_pickle=False) as pool:
        probability = pool[f"k{k}_membership"]
        raw_labels = probability.argmax(1)
        order = sorted(range(k), key=lambda j: int(np.flatnonzero(raw_labels == j)[0]))
        probability = probability[:, order]
        save_arrays(
            Path(base) / "selected_model.npz",
            component_order=np.asarray(order),
            **({"covariance_type": np.asarray("tied")} if covariance_type == "tied" else {}),
            **{name: pool[f"k{k}_{name}"][order] for name in ("weights", "means", "covariances")},
        )
    save_arrays(
        Path(base) / "selected_memberships.npz",
        membership=probability,
        labels=probability.argmax(1),
    )
    with (Path(base) / "selected_assignments.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("unit_id", "cluster_id", *(f"component_{j + 1}" for j in range(k))))
        writer.writerows(
            (u, int(p.argmax()) + 1, *p) for u, p in zip(units, probability, strict=True)
        )


def finish_screen(
    base, values, layout, model, config, full_record, *, direct=False, fold_size_policy=None
):
    import numpy as np

    from super_deli.models.predictive_selection import fold_assignments, screen_candidates

    units = list(layout) if direct else list(layout["unit_ids"])
    assignments = fold_assignments(units, config)
    write_json(assignments, Path(base) / "fold_assignments.json")
    fits = [full_record]
    folds = []
    for assignment in assignments:
        i = assignment["fold"]
        prefix = f"fold_{i:03d}"
        train, test = assignment["training_indices"], assignment["validation_indices"]
        trained, exposure = fold_with_support(
            model, i, units, [units[j] for j in train], fold_size_policy
        )
        if direct:
            record = score_search(base, prefix, values[train], [units[j] for j in train], trained)
        else:
            training, training_layout = subset_raw(values, layout, train)
            validation, validation_layout = subset_raw(values, layout, test)
            if exposure is None:
                record = raw_search(base, prefix, training, training_layout, trained)
                write_json(training_layout, Path(base) / prefix / "training_layout.json")
            else:
                from super_deli.workflows.predictive_resampling import occurrence_search

                record = occurrence_search(
                    base,
                    prefix,
                    values,
                    layout,
                    np.asarray(train, dtype=int),
                    trained,
                    fold_minimum=exposure["training_minimum"],
                    fold_support=exposure,
                )
            write_json(validation_layout, Path(base) / prefix / "validation_layout.json")
        if exposure is not None:
            if read_json(Path(base) / prefix / "fold_support.json") != exposure:
                raise ValueError("Frozen fold-support record differs")
        fits.append(record)
        row = {
            **assignment,
            "prefix": prefix,
            "status": record["status"],
            "model": trained.to_dict(),
        }
        if exposure is not None:
            row["fold_support"] = exposure
        if record["status"] == "complete":
            try:
                scores = (
                    values[test]
                    if direct
                    else project_exported(
                        Path(base) / prefix / "run", validation, validation_layout
                    )
                )
                row["log_densities"] = predict_pool(
                    Path(base) / prefix / "run", scores, Path(base) / prefix / "validation.npz"
                )
                row["candidates"] = read_json(Path(base) / prefix / "run/candidates.json")
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row.update(status="failed", error=str(exc))
        else:
            row["error"] = record["error"]
        write_json(
            {k: v for k, v in row.items() if k != "log_densities"},
            Path(base) / prefix / "validation_record.json",
        )
        folds.append(row)
    selection = screen_candidates(
        read_json(Path(base) / "full/run/candidates.json"),
        folds,
        n_units=len(units),
        n_folds=config.n_folds,
    )
    write_selected(base, selection, units, model.covariance_type)
    return selection, fits


def summary_for(base, selection, records, config, fold_size_policy=None):
    result = {
        "schema_version": 1,
        "workflow": "predictive_screened_aic_v1",
        "selection": selection,
        "screen_config": config.to_dict(),
        "searches": records,
        "planned_fits": 1 + config.n_folds,
        "attempted_fits": len(records),
        "successful_fits": sum(r["status"] == "complete" for r in records),
        "failed_fits": sum(r["status"] != "complete" for r in records),
        "unattempted_fits": 1 + config.n_folds - len(records),
        "scope": "Opt-in predictive selection only; significance and stability are unavailable",
    }
    if fold_size_policy is not None:
        result.update(
            workflow="predictive_screened_aic_scaled_fold_v1",
            fold_size_policy=fold_size_policy.to_dict(),
        )
    write_json(result, Path(base) / "summary.json")
    return result


def run_predictive(
    dataset, *, model_config, screen_config, base_dir, output_dir, fold_size_policy=None
):
    from threadpoolctl import threadpool_limits

    from super_deli.models.mixture import FIT_LOCK

    screen_config = PredictiveScreenConfig.from_dict(screen_config.to_dict())
    screen_config.validate_model(model_config)
    fold_size_policy = checked_policy(fold_size_policy, model_config, screen_config)
    base = Path(output_dir).absolute()
    base.mkdir(exist_ok=False)
    write_json(
        {
            "dataset": dataset.to_dict(),
            "predictive_model": model_config.to_dict(),
            "screening": screen_config.to_dict(),
            **(
                {"fold_size_policy": fold_size_policy.to_dict()}
                if fold_size_policy is not None
                else {}
            ),
        },
        base / "configuration.json",
    )
    with FIT_LOCK, threadpool_limits(limits=1):
        record = dataset_search(base, "full", dataset, base_dir, model_config)
        if record["status"] != "complete":
            return summary_for(
                base,
                {"status": "withheld", "screened_k": None, "error": "Full-data fit failed"},
                [record],
                screen_config,
                fold_size_policy,
            )
        try:
            values, layout = load_sources(dataset, model_config, base / "full/run")
            write_json(layout, base / "layout.json")
            save_arrays(base / "source_values.npz", **values)
            selection, records = finish_screen(
                base,
                values,
                layout,
                model_config,
                screen_config,
                record,
                fold_size_policy=fold_size_policy,
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            write_json({"error": str(exc)}, base / "workflow_failure.json")
            raise
    return summary_for(base, selection, records, screen_config, fold_size_policy)


def _run_score_screen(scores, *, model, screen_config, output_dir):
    base = Path(output_dir)
    base.mkdir(exist_ok=False)
    units = [f"score_{i:05d}" for i in range(len(scores))]
    write_json(
        {
            "predictive_model": model.to_dict(),
            "screening": screen_config.to_dict(),
            "direct_scores": True,
        },
        base / "configuration.json",
    )
    record = score_search(base, "full", scores, units, model)
    if record["status"] != "complete":
        return summary_for(
            base,
            {"status": "withheld", "screened_k": None, "error": "Full-data fit failed"},
            [record],
            screen_config,
        )
    selection, records = finish_screen(
        base, scores, units, model, screen_config, record, direct=True
    )
    return summary_for(base, selection, records, screen_config)


def run_score_screen(scores, *, model, screen_config, output_dir):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.models.mixture import FIT_LOCK

    screen_config = PredictiveScreenConfig.from_dict(screen_config.to_dict())
    screen_config.validate_model(model)
    if (
        not isinstance(scores, np.ndarray)
        or scores.ndim != 2
        or scores.shape[1] != model.n_pcs
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Finite score columns must match the declared PCA dimension")
    if model.covariance_type != "full":
        raise ValueError("Direct-score screening remains full-family only")
    with FIT_LOCK, threadpool_limits(limits=1):
        return _run_score_screen(
            scores, model=model, screen_config=screen_config, output_dir=output_dir
        )
