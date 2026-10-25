"""Whole-locality bootstrap of the complete fixed-fold shared predictive workflow."""

from pathlib import Path

from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.workflows.predictive import run_predictive, save_arrays


def run_predictive_bootstrap(
    dataset, *, model_config, screen_config, bootstrap_config, base_dir, output_dir
):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.models.mixture import FIT_LOCK
    from super_deli.models.predictive_config import PredictiveScreenConfig
    from super_deli.models.predictive_stability import summarize_partitions
    from super_deli.workflows.predictive_resampling import occurrence_screen

    c = PredictiveBootstrapConfig.from_dict(bootstrap_config.to_dict())
    screen_config = PredictiveScreenConfig.from_dict(screen_config.to_dict())
    c.validate_model(model_config, screen_config)
    n = len(dataset.localities)
    if 320 * n * n > model_config.max_pairwise_bytes:
        raise ValueError("Predictive stability exceeds max_pairwise_bytes")
    base = Path(output_dir).absolute()
    base.mkdir(exist_ok=False)
    write_json(
        {
            "dataset": dataset.to_dict(),
            "model": model_config.to_dict(),
            "screening": screen_config.to_dict(),
            "bootstrap": c.to_dict(),
        },
        base / "configuration.json",
    )
    point = run_predictive(
        dataset,
        model_config=model_config,
        screen_config=screen_config,
        base_dir=base_dir,
        output_dir=base / "point",
    )
    planned = (1 + c.n_bootstrap) * 6
    if point["selection"]["status"] != "complete":
        result = {
            "status": "unavailable",
            "error": "Point screen unavailable",
            "point": point,
            "planned_fits": planned,
            **{k: point[k] for k in ("attempted_fits", "successful_fits", "failed_fits")},
            "unattempted_fits": planned - point["attempted_fits"],
            "replicates": [],
        }
        write_json(result, base / "summary.json")
        return result
    layout = read_json(base / "point/layout.json")
    units = layout["unit_ids"]
    with np.load(base / "point/source_values.npz", allow_pickle=False) as raw:
        values = {key: raw[key] for key in raw.files}
    with np.load(base / "point/selected_memberships.npz", allow_pickle=False) as p:
        point_labels = p["labels"]
    assignments = read_json(base / "point/fold_assignments.json")
    rows = []
    with FIT_LOCK, threadpool_limits(limits=1):
        for i, seed in enumerate(c.draw_seeds()):
            draw = np.random.default_rng(seed).integers(n, size=n)
            prefix = f"draw_{i:04d}"
            row = {
                "replicate": i,
                "draw_seed": seed,
                "training_indices": draw.tolist(),
                "multiplicities": np.bincount(draw, minlength=n).tolist(),
                "oob_indices": np.flatnonzero(np.bincount(draw, minlength=n) == 0).tolist(),
                "prefix": prefix,
                "model_seed": model_config.random_seed,
                "status": "failed",
            }
            try:
                summary = occurrence_screen(
                    base / prefix, values, layout, draw, model_config, assignments
                )
                row.update(
                    {
                        k: summary[k]
                        for k in (
                            "attempted_fits",
                            "successful_fits",
                            "failed_fits",
                            "unattempted_fits",
                        )
                    }
                )
                if summary["selection"]["status"] != "complete":
                    raise ValueError(summary["selection"].get("error", "Unavailable draw screen"))
                with np.load(base / prefix / "original_predictions.npz", allow_pickle=False) as p:
                    row.update(
                        status="complete",
                        labels=p["labels"].tolist(),
                        selected_k=summary["selection"]["screened_k"],
                    )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row["error"] = str(exc)
                if "attempted_fits" not in row:
                    fits = [read_json(p) for p in (base / prefix).glob("*/record.json")]
                    row.update(
                        attempted_fits=len(fits),
                        successful_fits=sum(r["status"] == "complete" for r in fits),
                        failed_fits=sum(r["status"] != "complete" for r in fits),
                        unattempted_fits=6 - len(fits),
                    )
            write_json(row, base / (prefix + ".json"))
            rows.append(row)
    stability, arrays = summarize_partitions(point_labels, point["selection"]["screened_k"], rows)
    save_arrays(base / "stability.npz", **arrays)
    write_json(stability, base / "stability.json")
    result = {
        "schema_version": 1,
        "workflow": "shared_predictive_locality_bootstrap_v1",
        "status": stability["status"],
        "point": point,
        "stability": stability,
        "unit_ids": units,
        "replicates": rows,
        "planned_fits": planned,
        **{
            k: point[k] + sum(r[k] for r in rows)
            for k in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")
        },
        "conditioning": "Original locality folds and model/fold seeds held fixed; "
        "whole-locality draws vary; members/counts fixed",
        "interpretation": "All-unit stability includes resubstitution; OOB support may "
        "be sparse. Neither is a species probability or p-value.",
    }
    write_json(result, base / "summary.json")
    return result
