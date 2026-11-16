"""Paired native bootstrap/subsampling workflows with frozen independent evaluation."""

import importlib.metadata
import platform
import shutil
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from super_deli.benchmarking.bootstrap_split_audit import audit
from super_deli.benchmarking.bootstrap_split_metrics import measure
from super_deli.benchmarking.bootstrap_split_study import parity
from super_deli.benchmarking.full_retention_observer import observe_pipeline
from super_deli.benchmarking.full_retention_run import run_native
from super_deli.benchmarking.individual_subsample_audit import audit_subsampling
from super_deli.benchmarking.individual_subsample_config import tasks
from super_deli.benchmarking.individual_subsample_observer import predictions
from super_deli.benchmarking.individual_subsample_simulation import (
    audit_law,
    generate,
    load_evaluation,
)
from super_deli.benchmarking.initialization_audit import audit_initialization
from super_deli.benchmarking.initialization_study import native, size
from super_deli.benchmarking.signal_loss_fit import observe_candidates
from super_deli.benchmarking.signal_loss_source import digest
from super_deli.io.config import read_json, write_json
from super_deli.io.individual_subsample_source import load_point_source
from super_deli.models.full_retention import full_retention_model
from super_deli.models.individual_subsample_config import IndividualSubsampleConfig
from super_deli.workflows.individual_subsampling import run_individual_subsampling


def parent(task, directory, generator, *, draws=25):
    directory = Path(directory)
    directory.mkdir()
    started = time.monotonic()
    innovations = [
        int(c.generate_state(1)[0])
        for c in np.random.SeedSequence([task["seed"], 20261116]).spawn(2)
    ]
    data = directory / "observed"
    dataset, model, truth = generate(
        data,
        parameter_seed=task["seed"],
        innovation=innovations[0],
        n=task["n"],
        law=task["law"],
        tag="o",
        generator=generator,
    )
    model = replace(model, n_bootstrap=draws)
    edata = directory / "evaluation"
    ed, em, et = generate(
        edata,
        parameter_seed=task["seed"],
        innovation=innovations[1],
        n=1024,
        law=task["law"],
        tag="e",
        generator=generator,
    )
    audit_law(data)
    audit_law(edata)
    evaluation = load_evaluation(ed, em, edata, directory / "evaluation-ingestion")
    assert not set(o.specimen_id for o in dataset.specimens) & set(evaluation[1])
    ordinary = directory / "ordinary"
    with predictions(ordinary, evaluation):
        baseline = run_native(
            dataset,
            model,
            data,
            ordinary,
            law="continuous_control" if task["law"] == "continuous_control" else "shape_only",
            truth=truth,
            auditor=audit_initialization,
            observer=partial(observe_candidates, initialization=True),
        )
    assert baseline["status"] == "complete"
    source = load_point_source(ordinary)
    original = (source.inputs, source.units)
    lookup = dict(zip(truth["unit_ids"], truth["labels"], strict=True))
    y = np.array([lookup[u] for u in source.units])
    # Bind input identities and exact fixed-REF genomic dosage after native ingestion.
    with np.load(data / "known-raw.npz", allow_pickle=False) as values:
        order = {u: i for i, u in enumerate(truth["unit_ids"])}
        for d in source.inputs:
            np.testing.assert_array_equal(d.values, values[d.kind][[order[u] for u in d.unit_ids]])
    ba = audit(
        ordinary,
        original,
        evaluation,
        candidate_directory="observed-candidates",
        pipeline_directory="observed-pipeline",
    )
    target = directory / "subsample"
    captures = directory / "subsample-captures"
    captures.mkdir()
    with predictions(target, evaluation):
        with observe_candidates(
            captures / "observed-candidates", initialization=True
        ) as candidates:
            with observe_pipeline(captures / "observed-pipeline", candidates):
                summary = run_individual_subsampling(
                    ordinary,
                    stability_config=IndividualSubsampleConfig(draws, 1601),
                    output_dir=target,
                )
    for child in captures.iterdir():
        child.rename(target / child.name)
    captures.rmdir()
    sa = audit_subsampling(target, evaluation)
    point = source.labels
    be = read_json(ordinary / "observed-pipeline/events.json")
    se = read_json(target / "observed-pipeline/events.json")
    assert [e["seed"] for e in be[1:]] == [e["seed"] for e in se]
    b = measure(ordinary, ba["rows"], y, point, [e["draw"] for e in be])
    s = measure(target, sa["numerical"]["rows"], y, point, [e["draw"] for e in se])
    result = dict(
        task=task,
        innovations=innovations,
        configuration=model.to_dict(),
        arms=dict(ordinary=b, subsample=s),
        source_point_k=source.summary["selected_k"],
        subsample_summary=summary,
        seconds=time.monotonic() - started,
        bytes=size(directory),
        searches=1 + 2 * draws,
        attempted_searches=ba["attempted_searches"] + len(se),
        withheld=ba["withheld"] + sa["withheld"],
    )
    write_json(result, directory / "summary.json")
    return result


def study(source, config, output, phase, generator):
    if config.fixture:
        raise ValueError("Invented source seals cannot authorize scientific validation")
    output = Path(output)
    output.mkdir(exist_ok=True)
    generator = Path(generator)
    package = Path(__file__).resolve().parents[1]
    frozen = dict(
        config=config.to_dict(),
        source={
            p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob("*.py"))
        },
        python=platform.python_version(),
        libraries={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "snpio")
        },
        generator=digest(generator),
        base_generator=digest(generator.parents[1] / "matched-genomic/smoke.py"),
        tasks={p: tasks(p) for p in ("preflight", "main")},
        sampling="80percent without replacement",
        draws=25,
        subsample_seed=1601,
        model_seed=301,
        paired_fit_seeds=True,
        main_searches=2552,
        preflight_searches=358,
    )
    if (output / "freeze.json").exists():
        assert read_json(output / "freeze.json") == frozen
    else:
        write_json(frozen, output / "freeze.json")
    if phase == "main" and not read_json(output / "resource-gate.json")["passed"]:
        raise ValueError("Passed resource gate required")
    root = output / phase
    root.mkdir()
    started = time.monotonic()
    records = []
    replays = []
    limit = config.max_preflight_seconds if phase == "preflight" else config.max_main_seconds

    def guard():
        if (
            size(output) > config.artifact_cap_bytes
            or shutil.disk_usage(output).free < config.min_free_bytes
            or time.monotonic() - started > limit
        ):
            raise ValueError("Resource guard failed; retain evidence and stop")

    guard()
    for task in tasks(phase):
        records.append(parent(task, root / f"fresh-{task['index']:03d}", generator))
        guard()
        print(phase, "fresh", task["index"], flush=True)
    for index in (0, 160) if phase == "preflight" else (0, 60, 100, 160):
        data = root / f"replay-{index:03d}-inputs"
        dataset, analysis = source.stage(index, data)
        truth = read_json(data / "truth-evaluation-only.json")
        model = full_retention_model(
            replace(analysis.arms[-1].model, n_init=100, n_bootstrap=25), min_cluster_size=9
        )
        row = native(dataset, model, data, root / f"replay-{index:03d}", truth["law"], truth)
        assert row["status"] == "complete"
        row["exact_arrays"] = parity(source, source.baseline(index), root / f"replay-{index:03d}")
        replays.append(row)
        guard()
    source.reverify()
    elapsed = time.monotonic() - started
    searches = sum(r["searches"] for r in records) + 26 * len(replays)
    assert searches == (358 if phase == "preflight" else 2552)
    result = dict(
        passed=True,
        phase=phase,
        parents=records,
        replays=replays,
        searches=searches,
        attempted_searches=sum(r["attempted_searches"] for r in records) + 26 * len(replays),
        withheld=sum(r["withheld"] for r in records),
        seconds=elapsed,
        bytes=size(root),
        source_consumed=source.consumed,
    )
    write_json(result, root / "summary.json")
    if phase == "preflight":
        seconds = (
            48 * max(r["seconds"] for r in records) + 4 * max(r["seconds"] for r in replays)
        ) * 1.25
        storage = int(
            (48 * max(r["bytes"] for r in records) + 4 * max(r["bytes"] for r in replays)) * 1.25
            + size(output)
            + 512 * 1024**2
        )
        gate = dict(
            passed=seconds < config.max_main_seconds
            and storage < config.artifact_cap_bytes
            and elapsed < config.max_preflight_seconds
            and shutil.disk_usage(output).free >= config.min_free_bytes,
            projected_seconds=seconds,
            projected_bytes=storage,
            preflight_seconds=elapsed,
            free_bytes=shutil.disk_usage(output).free,
            method="Worst complete parent/replay,25percent allowance and512MiB reserve",
        )
        write_json(gate, output / "resource-gate.json")
        if not gate["passed"]:
            raise ValueError("Preflight resource gate failed; preserve and stop")
    return result
