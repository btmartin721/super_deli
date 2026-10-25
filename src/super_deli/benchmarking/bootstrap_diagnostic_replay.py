"""Exact point/draw replays; original seeds and evidence stay immutable."""

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.local_archives import compact_directory, materialize
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy


def compare_workflows(reference, replay):
    import numpy as np

    reference, replay = Path(reference), Path(replay)
    if (
        read_json(reference / "summary.json")["selection"]
        != read_json(replay / "summary.json")["selection"]
    ):
        raise ValueError("Exact replay selection differs")
    prefixes = ("full",) + tuple(f"fold_{i:03d}" for i in range(5))
    checked = 0
    for prefix in prefixes:
        for name in ("run/candidates.json",):
            if read_json(reference / prefix / name) != read_json(replay / prefix / name):
                raise ValueError("Exact replay candidates differ")
        for file in sorted((reference / prefix / "run").rglob("*.npz")):
            relative = file.relative_to(reference)
            with (
                np.load(file, allow_pickle=False) as a,
                np.load(replay / relative, allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files):
                    raise ValueError("Exact replay parameter keys differ")
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
        if prefix != "full":
            with (
                np.load(reference / prefix / "validation.npz", allow_pickle=False) as a,
                np.load(replay / prefix / "validation.npz", allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files):
                    raise ValueError("Exact replay density keys differ")
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
        checked += 1
    name = (
        "original_predictions.npz"
        if (reference / "original_predictions.npz").exists()
        else "selected_memberships.npz"
    )
    with (
        np.load(reference / name, allow_pickle=False) as a,
        np.load(replay / name, allow_pickle=False) as b,
    ):
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    return checked


def _replay_observation(task):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.predictive import run_predictive
    from super_deli.workflows.predictive_resampling import occurrence_screen

    source, output, raw, row, scratch = task
    source, output = Path(source), Path(output)
    c = PredictiveBootstrapStudy.from_dict(raw)
    target = output / ".working" / f"{row['cell']}_{row['repetition']:04d}"
    target.mkdir(parents=True, exist_ok=False)
    with materialize(
        source / row["archive"], row, scratch=scratch, min_free_bytes=c.min_free_bytes
    ) as base:
        with np.load(base / "values.npz", allow_pickle=False) as original_values:
            values = {key: original_values[key].copy() for key in original_values.files}
        layout = read_json(base / "generating_layout.json")
        dataset = write_raw_views(values, layout, target / "observed")
        original = base / "workflow"
        assignments = read_json(original / "point/fold_assignments.json")
        run_predictive(
            dataset,
            model_config=c.model,
            screen_config=replace(c.screening, random_seed=row["split_seed"]),
            base_dir=target / "observed",
            output_dir=target / "point",
        )
        checked = compare_workflows(original / "point", target / "point")
        records = read_json(original / "summary.json")["replicates"]
        for record in records[:2]:
            name = record["prefix"]
            occurrence_screen(
                target / name,
                values,
                layout,
                np.asarray(record["training_indices"]),
                c.model,
                assignments,
            )
            checked += compare_workflows(original / name, target / name)
        write_json(dict(source_archive=row, exact_searches=checked), target / "verification.json")
    relative = f"{row['cell']}/dataset_{row['repetition']:04d}.tar.gz"
    packed = compact_directory(target, output / relative)
    shutil.rmtree(target)
    return dict(
        cell=row["cell"],
        repetition=row["repetition"],
        exact_searches=checked,
        **{k: v for k, v in packed.items() if k not in {"archive", "manifest"}},
        archive=relative,
        manifest=relative + ".manifest.json",
    )


def replay_observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _replay_observation(task)


def replay_study(config, *, source, output_dir, scratch, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    source, output = Path(source), Path(output_dir)
    c = PredictiveBootstrapStudy.from_dict(read_json(source / "configuration.json"))
    if config.model != c.model or config.screening != c.screening:
        raise ValueError("Replay must retain the original model and screening settings")
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    rows = sorted(
        [
            r
            for r in rows
            if r["cell"] in {"four_n64_s3_0", "four_n64_s4_5"} and r["repetition"] < 5
        ],
        key=lambda r: (r["cell"], r["repetition"]),
    )
    if len(rows) != 10:
        raise ValueError("Expected ten archived replay observations")
    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(
            pool.map(
                replay_observation,
                [(str(source), str(output), c.to_dict(), r, str(scratch)) for r in rows],
            )
        )
    with (output / "observations.jsonl").open("x") as ledger:
        for row in records:
            ledger.write(json.dumps(row, sort_keys=True) + "\n")
    summary = dict(
        status="verified",
        observations=len(records),
        exact_searches=sum(r["exact_searches"] for r in records),
    )
    write_json(summary, output / "summary.json")
    return summary
