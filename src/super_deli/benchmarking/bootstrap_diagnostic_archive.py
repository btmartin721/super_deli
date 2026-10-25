"""Read-only diagnosis of the previous cohort; original evidence is never duplicated or changed."""

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.bootstrap_diagnostic_audit import audit_metrics, audit_point
from super_deli.benchmarking.bootstrap_diagnostic_metrics import diagnose_workflow
from super_deli.benchmarking.local_archives import compact_directory, materialize
from super_deli.benchmarking.predictive_audit import independent_simulation
from super_deli.benchmarking.predictive_bootstrap_audit import audit_draw
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy
from super_deli.models.predictive_study_config import PredictiveCell


def _archived_observation(task):
    import numpy as np

    source, destination, raw, row, scratch = task
    source, destination = Path(source), Path(destination)
    c = PredictiveBootstrapStudy.from_dict(raw)
    cell = PredictiveCell.from_dict(row["cell_config"])
    target = destination / ".working" / f"{row['cell']}_{row['repetition']:04d}"
    target.mkdir(parents=True, exist_ok=False)
    with materialize(
        source / row["archive"], row, scratch=scratch, min_free_bytes=c.min_free_bytes
    ) as base:
        values, layout, truth = independent_simulation(cell, cell.n_units, row["data_seed"])
        layout = json.loads(json.dumps(layout))
        if read_json(base / "generating_layout.json") != layout:
            raise ValueError("Archived layout differs from generating law")
        with np.load(base / "values.npz", allow_pickle=False) as p:
            for key, value in values.items():
                np.testing.assert_allclose(p[key], value, atol=4e-14, rtol=4e-14)
        with np.load(base / "truth.npz", allow_pickle=False) as p:
            np.testing.assert_array_equal(p["labels"], truth)
        workflow = base / "workflow"
        saved = read_json(workflow / "summary.json")
        assignments = read_json(workflow / "point/fold_assignments.json")
        point_labels = None
        if saved["point"]["selection"]["status"] == "complete":
            with np.load(workflow / "point/selected_memberships.npz", allow_pickle=False) as p:
                point_labels = p["labels"].copy()
        evaluated = truth if cell.law in {"complementary", "unequal_variance"} else None
        checked, available = 0, 0
        seeds = PredictiveBootstrapConfig(c.n_bootstrap, row["bootstrap_seed"]).draw_seeds()
        records = [(-1, list(range(cell.n_units)), saved["point"])]
        for i, record in enumerate(saved["replicates"]):
            indices = (
                np.random.default_rng(seeds[i]).integers(cell.n_units, size=cell.n_units).tolist()
            )
            records.append((i, indices, record))
        if len(records) != 1 + c.n_bootstrap:
            raise ValueError("Archived draw cohort incomplete; no replacements allowed")
        for i, indices, record in records:
            name = "point" if i == -1 else f"draw_{i:04d}"
            directory = workflow / "point" if i == -1 else workflow / name
            if i == -1:
                checked += audit_point(directory, values, layout, c.model)
            else:
                checked += audit_draw(
                    directory, values, layout, np.asarray(indices), c.model, assignments, record
                )
            result = diagnose_workflow(
                directory, values, layout, indices, c.model, evaluated, point_labels
            )
            audit_metrics(directory, values, layout, indices, c.model, evaluated, result)
            available += result["gate"]["status"] == "available"
            write_json(result, target / (name + ".json"))
        write_json(
            dict(
                source_archive=row,
                source_configuration_sha256=fingerprint_file(source / "configuration.json").sha256,
            ),
            target / "source.json",
        )
    relative = f"{row['cell']}/dataset_{row['repetition']:04d}.tar.gz"
    packed = compact_directory(target, destination / relative)
    shutil.rmtree(target)
    return {
        **row,
        **packed,
        "archive": relative,
        "manifest": relative + ".manifest.json",
        "diagnosed_workflows": len(records),
        "available_workflows": available,
        "audited_searches": checked,
    }


def archived_observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _archived_observation(task)


def archive_study(source, *, output_dir, scratch, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    source, output = Path(source).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    c = PredictiveBootstrapStudy.from_dict(read_json(source / "configuration.json"))
    rows = [json.loads(line) for line in (source / "observations.jsonl").read_text().splitlines()]
    expected = {(r["cell"], r["repetition"]): r for r in c.plan()}
    if len(rows) != len(expected) or {(r["cell"], r["repetition"]) for r in rows} != set(expected):
        raise ValueError("Archive cohort differs from its declared observations")
    write_json(
        dict(
            source=str(source),
            configuration_sha256=fingerprint_file(source / "configuration.json").sha256,
            observations_sha256=fingerprint_file(source / "observations.jsonl").sha256,
        ),
        output / "source.json",
    )
    results = []
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (output / "observations.jsonl").open("x") as ledger,
    ):
        tasks = [(str(source), str(output), c.to_dict(), r, str(scratch)) for r in rows]
        for row in pool.map(archived_observation, tasks, chunksize=1):
            results.append(row)
            ledger.write(json.dumps(row, sort_keys=True) + "\n")
            ledger.flush()
    summary = dict(
        status="complete",
        observations=len(results),
        draws=len(results) * c.n_bootstrap,
        audited_searches=sum(r["audited_searches"] for r in results),
        available_workflows=sum(r["available_workflows"] for r in results),
        requested_workflows=len(results) * (1 + c.n_bootstrap),
        refits=0,
    )
    write_json(summary, output / "summary.json")
    return summary
