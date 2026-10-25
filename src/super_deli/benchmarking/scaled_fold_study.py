"""Independent raw datasets, paired complete refits, and own/common-reference stability."""

import contextlib
import shutil
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.bounded_study import run_bounded_study
from super_deli.benchmarking.local_archives import compact_directory
from super_deli.benchmarking.predictive_study import simulate
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig

COUNTS = ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")


def run_arm(dataset, base, c, screen, bootstrap, arm):
    import numpy as np

    from super_deli.workflows.predictive_bootstrap import run_predictive_bootstrap

    target = base / arm
    options = {} if arm == "legacy" else dict(fold_size_policy=c.fold_size_policy)
    try:
        return run_predictive_bootstrap(
            dataset,
            model_config=c.model,
            screen_config=screen,
            bootstrap_config=bootstrap,
            base_dir=base / "observed",
            output_dir=target,
            **options,
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, OSError) as exc:
        # Preserve partial fits. Missing records remain unattempted, never K=1.
        target.mkdir(exist_ok=True)
        fits = [read_json(p) for p in target.glob("*/*/record.json")]
        point = (
            read_json(target / "point/summary.json")
            if (target / "point/summary.json").exists()
            else dict(selection=dict(status="withheld", screened_k=None, error=str(exc)))
        )
        if not (target / "point/summary.json").exists() and (target / "point").exists():
            partial = [read_json(p) for p in (target / "point").glob("*/record.json")]
            point.update(
                planned_fits=6,
                attempted_fits=len(partial),
                successful_fits=sum(r["status"] == "complete" for r in partial),
                failed_fits=sum(r["status"] != "complete" for r in partial),
                unattempted_fits=6 - len(partial),
            )
            if arm == "scaled":
                point["fold_size_policy"] = c.fold_size_policy.to_dict()
            write_json(point, target / "point/summary.json")
        records = [read_json(p) for p in sorted(target.glob("draw_*.json"))]
        summary = dict(
            status="unavailable",
            error=str(exc),
            point=point,
            replicates=records,
            planned_fits=6 * (1 + c.n_bootstrap),
            attempted_fits=len(fits),
            successful_fits=sum(r["status"] == "complete" for r in fits),
            failed_fits=sum(r["status"] != "complete" for r in fits),
            unattempted_fits=6 * (1 + c.n_bootstrap) - len(fits),
        )
        write_json(dict(error=str(exc)), target / "execution_failure.json")
        if not (target / "summary.json").exists():
            write_json(summary, target / "summary.json")
        return summary


def evaluate(base, results, truth, group_truth):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.benchmarking.bootstrap_diagnostic_metrics import matched_changes

    reference = None
    if results["legacy"]["point"]["selection"]["status"] == "complete":
        with np.load(base / "legacy/point/selected_memberships.npz", allow_pickle=False) as p:
            reference = p["labels"].copy()
    rows = {}
    for arm, result in results.items():
        point = None
        if result["point"]["selection"]["status"] == "complete":
            with np.load(base / arm / "point/selected_memberships.npz", allow_pickle=False) as p:
                point = p["labels"].copy()
        draws = []
        for row in result["replicates"]:
            complete = row["status"] == "complete"
            labels = np.asarray(row["labels"]) if complete else None
            oob = np.asarray(row["oob_indices"], dtype=int)
            gate = None
            path = base / arm / row["prefix"] / "summary.json"
            if path.exists():
                selection = read_json(path)["selection"]
                if selection["status"] == "complete":
                    gate = next(v for v in selection["candidates"] if v["k"] == 4)
            draws.append(
                dict(
                    replicate=row["replicate"],
                    status=row["status"],
                    k=row.get("selected_k"),
                    truth_ari=float(adjusted_rand_score(truth, labels))
                    if complete and group_truth
                    else None,
                    oob_truth_ari=float(adjusted_rand_score(truth[oob], labels[oob]))
                    if complete and group_truth and len(oob) > 1
                    else None,
                    oob_count=len(oob),
                    own_assignment_changes=matched_changes(point, labels) if complete else None,
                    common_assignment_changes=matched_changes(reference, labels)
                    if complete
                    else None,
                    k4_gate=gate,
                )
            )
        rows[arm] = dict(
            point_k=result["point"]["selection"].get("screened_k"),
            point_truth_ari=float(adjusted_rand_score(truth, point))
            if point is not None and group_truth
            else None,
            point_common_assignment_changes=matched_changes(reference, point)
            if point is not None
            else None,
            draws=draws,
        )
    write_json(rows, base / "evaluation.json")
    return rows


def _observation(task):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views
    from super_deli.models.predictive_stability import summarize_partitions
    from super_deli.models.predictive_study_config import PredictiveCell
    from super_deli.models.scaled_fold_study import ScaledFoldStudy
    from super_deli.workflows.predictive import save_arrays

    source, raw, declaration = task
    root, c = Path(source), ScaledFoldStudy.from_dict(raw)
    if shutil.disk_usage(root).free < c.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked scratch space")
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    base = root / ".working" / f"{cell.name}_{declaration['repetition']:04d}"
    base.mkdir(parents=True, exist_ok=False)
    write_json(declaration, base / "declaration.json")
    with (
        (base / "stdout.log").open("x") as out,
        (base / "stderr.log").open("x") as err,
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        values, layout, truth = simulate(cell, cell.n_units, declaration["data_seed"])
        save_arrays(base / "values.npz", **values)
        save_arrays(base / "truth.npz", labels=truth)
        write_json(layout, base / "generating_layout.json")
        dataset = write_raw_views(values, layout, base / "observed")
        screen = replace(c.screening, random_seed=declaration["split_seed"])
        bootstrap = PredictiveBootstrapConfig(c.n_bootstrap, declaration["bootstrap_seed"])
        results = {
            arm: run_arm(dataset, base, c, screen, bootstrap, arm) for arm in ("legacy", "scaled")
        }
        # References/truth are evaluated only after both ordinary arms finish.
        common = dict(
            status="unavailable",
            requested=c.n_bootstrap,
            error="Legacy point or complete draw declarations unavailable",
        )
        reference = results["legacy"]["point"]["selection"]
        if (
            reference["status"] == "complete"
            and len(results["scaled"]["replicates"]) == c.n_bootstrap
        ):
            with np.load(base / "legacy/point/selected_memberships.npz", allow_pickle=False) as p:
                labels = p["labels"].copy()
            common, arrays = summarize_partitions(
                labels, reference["screened_k"], results["scaled"]["replicates"]
            )
            save_arrays(base / "common_stability.npz", **arrays)
        write_json(common, base / "common_stability.json")
        evaluation = evaluate(
            base, results, truth, cell.law in {"complementary", "unequal_variance"}
        )
    arms = {
        arm: dict(
            **{k: result[k] for k in COUNTS},
            point_selection=result["point"]["selection"],
            stability=result.get("stability"),
            successful_draws=sum(r["status"] == "complete" for r in result["replicates"]),
            failed_draws=sum(r["status"] != "complete" for r in result["replicates"]),
            unattempted_draws=c.n_bootstrap - len(result["replicates"]),
            point_truth_ari=evaluation[arm]["point_truth_ari"],
        )
        for arm, result in results.items()
    }
    record = dict(
        declaration,
        arms=arms,
        **{
            k: sum(r[k] for r in arms.values())
            for k in (*COUNTS, "successful_draws", "failed_draws", "unattempted_draws")
        },
    )
    write_json(record, base / "record.json")
    relative = f"{cell.name}/dataset_{declaration['repetition']:04d}.tar.gz"
    packed = compact_directory(base, root / relative)
    shutil.rmtree(base)
    return {**record, **packed, "archive": relative, "manifest": relative + ".manifest.json"}


def observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _observation(task)


def run_study(config, *, output_dir, workers=1, preflight=None):
    return run_bounded_study(
        config, output_dir=output_dir, worker=observation, workers=workers, preflight=preflight
    )
