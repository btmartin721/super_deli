"""Train each representation once; retain all fixed clustering heads and failures."""

import time
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.coordinate_fit import mixture
from super_deli.benchmarking.matched_data import write_dataset
from super_deli.benchmarking.moons_hierarchy import retained_hierarchy
from super_deli.benchmarking.neural_analysis import arrays
from super_deli.benchmarking.separate_gmm_config import ARMS, HEADS, NATIVE, model_for
from super_deli.benchmarking.separate_gmm_data import generate
from super_deli.benchmarking.separate_gmm_observer import evaluation_path, retain_points
from super_deli.benchmarking.view_weight_observer import retain_density, retain_training
from super_deli.io.config import load_config, read_json, write_json
from super_deli.models.density_scores import fit_density_scores
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.tabular_neural import point_arrays
from super_deli.workflows.separate_gmm import run_separate_gmm


def record_stop(path, row):
    if not Path(path).exists():
        write_json(row, path)


def save_head(path, row, a):
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "fit.npz", **a)
    write_json(row, path / "execution.json")


def density(scores):
    from super_deli.benchmarking.coordinate_config import model_for as density_config

    captured = {}

    def factory(**kwargs):
        captured["estimator"] = HDBSCAN(**kwargs)
        return captured["estimator"]

    with FIT_LOCK, threadpool_limits(limits=1):
        labels, strength = fit_density_scores(
            scores, density_config("separate_0"), estimator_class=factory
        )
    linkage, condensed = retained_hierarchy(captured["estimator"], len(scores), 5)
    return (
        dict(status="available", selected_k=len(set(labels) - {-1})),
        dict(labels=labels, strength=strength, linkage=linkage, condensed=condensed),
    )


def complete_heads(out, index, scores, *, raw=False, pca_point=None, trace=None):
    rows = []
    for head in ("gmm_shared",) if raw else HEADS:
        target = evaluation_path(out, index, head)
        if target.is_dir() and (target / "execution.json").exists():
            row = read_json(target / "execution.json")
        elif scores is None:
            row = dict(status="unattempted", selected_k=None, error="No complete representation")
            save_head(target, row, {})
        else:
            started = time.monotonic()
            try:
                if head == "hdbscan":
                    if pca_point is None:
                        row, a = density(scores)
                    else:
                        row = dict(status="available", selected_k=pca_point.selected_k)
                        a = dict(
                            labels=pca_point.labels,
                            strength=pca_point.strength,
                            linkage=trace["fit_0_linkage"],
                            condensed=trace["fit_0_condensed"],
                        )
                else:
                    row, a = mixture(scores, "full" if head == "gmm_full" else "tied")
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row, a = (dict(status="failed", selected_k=None, error=str(exc)), {})
            row["seconds"] = time.monotonic() - started
            save_head(target, row, a)
        rows.append(dict(index=index, head=head, **row))
    return rows


def run_arm(task, data_dir, out, arm):
    from super_deli.benchmarking import separate_controls
    from super_deli.models import tabular_neural

    seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
    draws = task["draws"] if arm in NATIVE else 0
    c = model_for(arm, seed, draws)
    started = time.monotonic()
    guard = {}
    events = []
    rows = []
    status, error = ("available", None)
    if arm in NATIVE:
        try:
            with retain_training(
                out, paired=False, module=tabular_neural, guard_state=guard, record_times=True
            ) as events:
                with retain_points(out):
                    run_separate_gmm(
                        load_config(data_dir / "dataset.json"),
                        model_config=c,
                        base_dir=data_dir,
                        output_dir=out,
                    )
        except Exception as exc:
            status, error = ("failed", str(exc))
        for i in range(1 + draws):
            path = evaluation_path(out, i, "coordinates") / "point.npz"
            scores = None
            if path.exists():
                a = arrays(path)
                if all(role + "__scores" in a for role in ("morphology", "ecology")):
                    scores = np.column_stack(
                        [a[role + "__scores"] for role in ("morphology", "ecology")]
                    )
            rows.extend(complete_heads(out, i, scores))
    else:
        out.mkdir()
        from super_deli.io.tabular import ingest_tabular
        from super_deli.models.matched_preprocessing import align_tables

        config = model_for("separate_0", seed)
        tables = ingest_tabular(
            load_config(data_dir / "dataset.json"),
            base_dir=data_dir,
            output_dir=out / "ingestion",
            modality_names=["morphology", "ecology"],
        ).tables
        x, alignment = align_tables(tables, config)
        write_json(alignment, out / "cohort.json")
        if arm == "pca4":
            with retain_density(out, [], paired=False, module=separate_controls, guard_state=guard):
                point = separate_controls.fit_pca_four(x, config=config.encoder_config())
            np.savez_compressed(out / "point.npz", **point_arrays(point))
            rows = complete_heads(
                out, 0, point.scores, pca_point=point, trace=arrays(out / "density-trace.npz")
            )
        else:
            mean, scale = (x.mean(0), x.std(0))
            if np.any(scale <= 0):
                raise ValueError("Nonconstant raw reference required")
            scores = (x - mean) / scale
            np.savez_compressed(out / "point.npz", scores=scores, means=mean, scales=scale)
            rows = complete_heads(out, 0, scores, raw=True)
    write_json(rows, out / "evaluations.json")
    result = dict(
        arm=arm,
        status=status,
        error=error,
        configuration=c.to_dict(),
        draws_requested=draws,
        planned=(1 if arm == "raw_shared" else 3) * (1 + draws),
        attempted=sum(r["status"] != "unattempted" for r in rows),
        failed_points=sum(r["status"] != "available" for r in rows),
        neural_fits_attempted=len(events),
        seconds=time.monotonic() - started,
        numerical_guard_failed=bool(guard.get("failed")),
    )
    write_json(result, out / "execution.json")
    return result


def run_dataset(job):
    started = time.monotonic()
    task = job["task"]
    out = Path(job["phase_root"]) / task["key"]
    out.mkdir()
    write_json(task, out / "task.json")
    data = generate(task)
    np.savez_compressed(out / "generating-reference.npz", **data)
    write_dataset(data, out / "raw")
    write_dataset(data, out / "canonical", canonical=True)
    records = []
    for arm in ARMS:
        if Path(job["stop_file"]).exists():
            raise ValueError("Earlier numerical guard failed")
        row = run_arm(task, out / "raw", out / arm, arm)
        records.append(row)
        if row["numerical_guard_failed"]:
            record_stop(
                job["stop_file"],
                dict(reason="Actual-path numerical guard", key=task["key"], arm=arm),
            )
    write_json(records, out / "fits.json")
    return dict(
        key=task["key"],
        seconds=time.monotonic() - started,
        planned=sum(r["planned"] for r in records),
        attempted=sum(r["attempted"] for r in records),
        neural_fits=sum(r["neural_fits_attempted"] for r in records),
        failed_points=sum(r["failed_points"] for r in records),
        numerical_guard_failed=any(r["numerical_guard_failed"] for r in records),
    )


def replay_equal(source, out, entry):
    old = arrays(source / entry["path"] / entry["arm"] / "point.npz")
    new = arrays(out / ("separate_" + str(entry["seed"])) / "point.npz")
    for key, v in old.items():
        if key not in ("labels", "strength"):
            np.testing.assert_array_equal(new[key], v)
    density_fit = arrays(
        evaluation_path(out / ("separate_" + str(entry["seed"])), 0, "hdbscan") / "fit.npz"
    )
    for key in ("labels", "strength"):
        np.testing.assert_array_equal(density_fit[key], old[key])
    if "coordinate_path" in entry:
        for head in ("gmm_full", "gmm_shared"):
            reference = arrays(source / entry["coordinate_path"] / entry["arm"] / head / "fit.npz")
            actual = arrays(
                evaluation_path(out / ("separate_" + str(entry["seed"])), 0, head) / "fit.npz"
            )
            for key, value in reference.items():
                np.testing.assert_array_equal(actual[key], value)


def run_replay(job):
    started = time.monotonic()
    source = Path(job["archive_root"])
    entry = job["entry"]
    out = Path(job["phase_root"]) / entry["key"]
    out.mkdir()
    write_json(entry, out / "replay.json")
    task = {**entry["task"], "draws": 0}
    arm = "separate_" + str(entry["seed"])
    row = run_arm(task, source / entry["path"] / "raw", out / arm, arm)
    try:
        replay_equal(source, out, entry)
        row["exact_numeric_replay"] = True
    except Exception as exc:
        row.update(error=str(exc), numerical_guard_failed=True)
        record_stop(
            job["stop_file"], dict(reason="Exact legacy replay", key=entry["key"], error=str(exc))
        )
    write_json([row], out / "fits.json")
    return dict(
        key=entry["key"],
        seconds=time.monotonic() - started,
        planned=3,
        attempted=row["attempted"],
        neural_fits=row["neural_fits_attempted"],
        failed_points=row["failed_points"],
        numerical_guard_failed=row["numerical_guard_failed"],
    )
