"""Identity-preserving locality occurrence fits using the existing integrative engine."""

from pathlib import Path

from super_deli.io.config import read_json, write_json
from super_deli.io.predictive_inputs import project_exported, subset_raw
from super_deli.workflows.predictive import fold_model, predict_pool, save_arrays, write_selected


def occurrence_search(base, prefix, values, layout, indices, model):
    """Reingest distinct raw members, then fit the explicitly expanded empirical measure."""
    import numpy as np

    from super_deli.io.integrative_artifacts import write_integration_parameters
    from super_deli.models.grouped_null import write_raw_views
    from super_deli.models.integrative import IntegrativeFit, _fit_once, _prepare
    from super_deli.models.unit_bootstrap import UnitStability
    from super_deli.workflows.integrative import _load_view

    if (
        not isinstance(indices, np.ndarray)
        or indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(layout["unit_ids"]))
    ):
        raise ValueError("Nonempty integer original-locality indices required")
    directory = Path(base) / prefix
    directory.mkdir()
    write_json(model.to_dict(), directory / "model.json")
    seed = int(np.random.SeedSequence(model.random_seed).spawn(1)[0].generate_state(1)[0])
    write_json({"original_indices": indices.tolist(), "point_seed": seed}, directory / "draw.json")
    run = directory / "run"
    run.mkdir()
    try:
        selected = sorted(set(map(int, indices)))
        raw, sublayout = subset_raw(values, layout, selected)
        write_json(sublayout, directory / "training_layout.json")
        declaration = write_raw_views(raw, sublayout, directory / "raw")
        sources = {s.name: s for s in declaration.modalities}
        inputs = []
        for i, view in enumerate(model.views):
            destination = run / f"view_{i:03d}"
            destination.mkdir()
            data, _, _ = _load_view(
                declaration, sources[view.modality], view, model, directory / "raw", destination
            )
            inputs.append(data)
        inputs, units, excluded = _prepare(inputs, model)
        if excluded or set(units) != set(sublayout["unit_ids"]):
            raise ValueError("Bootstrap training must retain every selected locality")
        positions = {u: i for i, u in enumerate(units)}
        draw = np.asarray([positions[layout["unit_ids"][j]] for j in indices], dtype=int)
        pipeline = _fit_once(inputs, units, model, seed, draw)
        occurrence_ids = tuple(units[i] for i in draw)
        fit = IntegrativeFit(
            pipeline.mixture,
            pipeline,
            UnitStability("not_computed", 0, 0, (), {}, (), (), {}),
            occurrence_ids,
            (),
            inputs,
            model,
            seed,
        )
        views = write_integration_parameters(fit, run)
        save_arrays(
            run / "gmm_candidates.npz", scores=pipeline.scores, **pipeline.mixture.candidate_arrays
        )
        write_json(pipeline.mixture.candidates, run / "candidates.json")
        write_json(
            {
                "unit_ids": occurrence_ids,
                "distinct_unit_ids": units,
                "views": views,
                "point_seed": seed,
            },
            run / "summary.json",
        )
        record = {
            "status": "complete",
            "prefix": prefix,
            "point_seed": seed,
            "n_occurrences": len(indices),
            "n_distinct": len(units),
        }
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record = {"status": "failed", "prefix": prefix, "point_seed": seed, "error": str(exc)}
    write_json(record, directory / "record.json")
    return record


def occurrence_screen(base, values, layout, indices, model, assignments):
    """Keep original-ID folds fixed; expand validation occurrences rather than fold averages."""
    import numpy as np

    from super_deli.models.predictive_selection import gaussian_predictions, screen_candidates

    base = Path(base)
    base.mkdir()
    fits = [occurrence_search(base, "full", values, layout, indices, model)]
    folds = []
    if fits[0]["status"] == "complete":
        for original in assignments:
            i = original["fold"]
            prefix = f"fold_{i:03d}"
            test_set = set(original["validation_indices"])
            test = [j for j, u in enumerate(indices) if u in test_set]
            train = [j for j, u in enumerate(indices) if u not in test_set]
            row = {
                "fold": i,
                "training_indices": train,
                "validation_indices": test,
                "original_training_indices": indices[train].tolist(),
                "original_validation_indices": indices[test].tolist(),
                "prefix": prefix,
            }
            if not train or not test:
                record = {
                    "status": "failed",
                    "prefix": prefix,
                    "error": "Empty sampled training or validation fold",
                }
                (base / prefix).mkdir()
                write_json(record, base / prefix / "record.json")
            else:
                record = occurrence_search(
                    base, prefix, values, layout, indices[train], fold_model(model, i)
                )
            fits.append(record)
            row.update(status=record["status"])
            if record["status"] == "complete":
                try:
                    # Only distinct held-out raw members enter projection; expand afterwards.
                    original_test = sorted(set(map(int, indices[test])))
                    raw, sublayout = subset_raw(values, layout, original_test)
                    distinct_scores = project_exported(base / prefix / "run", raw, sublayout)
                    positions = {u: j for j, u in enumerate(sublayout["unit_ids"])}
                    scores = distinct_scores[
                        [positions[layout["unit_ids"][u]] for u in indices[test]]
                    ]
                    row["log_densities"] = predict_pool(
                        base / prefix / "run", scores, base / prefix / "validation.npz"
                    )
                    row["candidates"] = read_json(base / prefix / "run/candidates.json")
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    row.update(status="failed", error=str(exc))
            else:
                row["error"] = record["error"]
            folds.append(row)
            write_json(row, base / prefix / "validation_record.json")
        selection = screen_candidates(
            read_json(base / "full/run/candidates.json"),
            folds,
            n_units=len(indices),
            n_folds=len(assignments),
        )
    else:
        selection = {
            "status": "withheld",
            "screened_k": None,
            "error": "Full bootstrap search failed",
        }
    write_json(folds, base / "folds.json")
    if selection["status"] == "complete":
        write_selected(base, selection, [layout["unit_ids"][i] for i in indices], "tied")
        scores = project_exported(base / "full/run", values, layout)
        k = selection["screened_k"]
        with np.load(base / "full/run/gmm_candidates.npz", allow_pickle=False) as pool:
            logs, probabilities = gaussian_predictions(scores, pool, k)
        with np.load(base / "selected_model.npz", allow_pickle=False) as fitted:
            probabilities = probabilities[:, fitted["component_order"]]
        save_arrays(
            base / "original_predictions.npz",
            scores=scores,
            membership=probabilities,
            labels=probabilities.argmax(1),
            log_density=logs,
        )
    summary = {
        "selection": selection,
        "searches": fits,
        "planned_fits": 6,
        "attempted_fits": len(fits),
        "successful_fits": sum(r["status"] == "complete" for r in fits),
        "failed_fits": sum(r["status"] != "complete" for r in fits),
        "unattempted_fits": 6 - len(fits),
        "validation_failures": sum(f["status"] != "complete" for f in folds),
    }
    write_json(summary, base / "summary.json")
    return summary
