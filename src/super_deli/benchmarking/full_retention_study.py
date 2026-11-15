"""Frozen raw-input study, paired native draws and bounded optimization diagnostics."""

import shutil
import time
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.full_retention_config import MISSES, REPLAYS, tasks
from super_deli.benchmarking.full_retention_diagnostics import truth_initialized
from super_deli.benchmarking.full_retention_run import run_native
from super_deli.benchmarking.multimodal_validation import equal_point_arrays
from super_deli.benchmarking.signal_loss_source import digest
from super_deli.benchmarking.signal_loss_study import load_generator
from super_deli.io.config import read_json, write_json
from super_deli.models.full_retention import full_retention_model
from super_deli.models.multimodal_analysis_config import MultimodalAnalysisArm


def _bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _native(dataset, model, base, directory, law, truth):
    start = time.monotonic()
    row = run_native(dataset, model, base, directory, law=law, truth=truth)
    row.update(seconds=time.monotonic() - start, bytes=_bytes(directory), path=str(directory))
    return row


def study(source, config, output, phase, generator):
    if config.fixture:
        raise ValueError("Invented seals cannot authorize scientific validation")
    output = Path(output)
    output.mkdir(exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    runtime = {p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob("*.py"))}
    frozen = dict(
        runtime_source_sha256=runtime,
        config=config.to_dict(),
        main=tasks("main"),
        preflight=tasks("preflight"),
        generator_sha256=digest(generator),
        base_generator_sha256=digest(Path(generator).parents[1] / "matched-genomic/smoke.py"),
        model_seed=301,
        starts=10,
        increased_starts=100,
        max_k=4,
        view_pcs=2,
        main_searches=2332,
        preflight_searches=637,
        main_truth_fits=22,
        preflight_truth_fits=2,
    )
    if (output / "freeze.json").exists():
        if read_json(output / "freeze.json") != frozen:
            raise ValueError("Study configuration or generator changed")
    else:
        write_json(frozen, output / "freeze.json")
    if phase == "main" and not read_json(output / "resource-gate.json")["passed"]:
        raise ValueError("A passed independent preflight resource gate is required")
    start = time.monotonic()
    root = output / phase
    root.mkdir()
    generate = load_generator(generator)
    records, point_controls, boot_controls, diagnostic_times = [], [], [], []
    searches = parity = 0
    diagnostic_roster = []
    for task in tasks(phase):
        data = root / f"fresh-{task['index']:03d}-inputs"
        case = root / f"fresh-{task['index']:03d}"
        case.mkdir()
        dataset, analysis = generate(data, seed=task["seed"], law=task["law"], n=task["n"], draws=0)
        truth = read_json(data / "truth-evaluation-only.json")
        base = replace(analysis.arms[-1].model, n_bootstrap=25 if task["stability"] else 0)
        assert (
            base.n_pcs == 2
            and base.min_cluster_size == 5
            and base.n_init == 10
            and base.random_seed == 301
        )
        models = dict(
            legacy_two=base,
            matched_two=replace(base, min_cluster_size=9),
            retained_all=full_retention_model(base, min_cluster_size=9),
        )
        planned = replace(
            analysis,
            arms=(
                *analysis.arms[:-1],
                *[
                    MultimodalAnalysisArm(
                        name, "integrative", tuple(v.modality for v in model.views), model
                    )
                    for name, model in models.items()
                ],
            ),
        )
        write_json(planned.to_dict(), case / "comparison-plan.json")
        arms = {}
        for name, model in models.items():
            row = _native(dataset, model, data, case / name, task["law"], truth)
            arms[name] = row
            searches += row["planned_searches"]
            if task["stability"]:
                boot_controls.append(row)
        record = dict(origin="fresh", task=task, arms=arms, source_bytes=_bytes(data))
        records.append(record)
        if phase == "preflight" or task["repeat"] < 2:
            old = replace(models["retained_all"], fusion_policy="joint_pca", n_bootstrap=0)
            control = _native(dataset, old, data, case / "explicit_eight", task["law"], truth)
            searches += 1
            point_controls.append(control)
            if control["status"] == arms["retained_all"]["status"] == "complete":
                parity += equal_point_arrays(case / "explicit_eight", case / "retained_all")
        if task["law"] == "shape_only" and task["n"] == 64:
            diagnostic_roster.append((dataset, models["retained_all"], data, case, truth, "fresh"))
        print(phase, "fresh", task["index"], {n: r["status"] for n, r in arms.items()}, flush=True)
    for index in REPLAYS if phase == "main" else (0, 11):
        data, case = root / f"archive-{index:02d}-inputs", root / f"archive-{index:02d}"
        case.mkdir()
        dataset, analysis = source.stage(index, data)
        truth = read_json(data / "truth-evaluation-only.json")
        model = replace(analysis.arms[-1].model, n_bootstrap=0)
        row = _native(dataset, model, data, case / "legacy_two", truth["law"], truth)
        searches += 1
        prefix = source.baseline(index)
        source.verify_numeric(prefix)
        if row["status"] == "complete":
            parity += equal_point_arrays(source.root / prefix, case / "legacy_two")
        record = dict(
            origin="archived", index=index, n=64, law=truth["law"], arms=dict(legacy_two=row)
        )
        records.append(record)
        if index in (MISSES if phase == "main" else (11,)):
            retained = full_retention_model(model, min_cluster_size=9)
            row = _native(dataset, retained, data, case / "retained_all", truth["law"], truth)
            record["arms"]["retained_all"] = row
            searches += 1
            diagnostic_roster.append((dataset, retained, data, case, truth, "archived"))
    diagnostics = []
    for dataset, model, data, case, truth, origin in diagnostic_roster:
        high = replace(model, n_init=100, n_bootstrap=0)
        row = _native(dataset, high, data, case / "hundred_starts", truth["law"], truth)
        searches += 1
        diagnostic_times.append(row)
        point = read_json(case / "retained_all/validation.json")
        then = time.monotonic()
        known = truth_initialized(case / "retained_all", truth, case / "truth_initialized")
        known.update(seconds=time.monotonic() - then, bytes=_bytes(case / "truth_initialized"))
        diagnostics.append(
            dict(
                origin=origin,
                n=64,
                law="shape_only",
                path=str(case),
                baseline=point,
                hundred_starts=row,
                truth_initialized=known,
            )
        )
    source.reverify()
    from super_deli.benchmarking.full_retention_validation import audit_phase

    independent = audit_phase(root, records, diagnostics)
    elapsed = time.monotonic() - start
    expected_searches, expected_truth = (637, 2) if phase == "preflight" else (2332, 22)
    assert searches == expected_searches and len(diagnostics) == expected_truth
    result = dict(
        status="complete",
        phase=phase,
        records=records,
        diagnostics=diagnostics,
        candidate_searches=searches,
        planned_truth_fits=expected_truth,
        exact_parity_arrays=parity,
        elapsed_seconds=elapsed,
        bytes=_bytes(root),
        independent=independent,
        consumed=source.consumed,
    )
    write_json(result, root / "summary.json")
    if phase == "preflight":
        # Point-only controls measure fixed ingestion/export costs. Boot workflow/25
        # bounds each draw conservatively by also charging its fixed overhead.
        completed = [r for r in point_controls if r["status"] == "complete"]
        assert len(completed) == 8 and all(r["status"] == "complete" for r in boot_controls)
        fixed = max(r["bytes"] for r in completed) + 8192
        per_draw = max(r["bytes"] for r in boot_controls) / 25
        truth_bytes = max(d["truth_initialized"]["bytes"] for d in diagnostics)
        input_bytes = max(r["source_bytes"] for r in records if r["origin"] == "fresh")
        projected_bytes = int(
            (532 * fixed + 1800 * per_draw + 22 * truth_bytes + 174 * input_bytes) * 1.25
            + _bytes(output)
            + 512 * 1024**2
        )
        projected_time = (
            510 * max(r["seconds"] for r in completed)
            + 1800 * max(r["seconds"] for r in boot_controls) / 25
            + 22 * max(r["seconds"] for r in diagnostic_times)
            + 22 * max(d["truth_initialized"]["seconds"] for d in diagnostics)
            + independent["seconds"] * 4
        ) * 1.25
        gate = dict(
            passed=bool(
                independent["passed"]
                and elapsed < config.max_preflight_seconds
                and projected_time < config.max_main_seconds
                and projected_bytes < config.artifact_cap_bytes
                and shutil.disk_usage(output).free >= config.min_free_bytes
            ),
            preflight_serial_seconds=elapsed,
            projected_serial_main_seconds=projected_time,
            projected_bytes=projected_bytes,
            free_bytes=shutil.disk_usage(output).free,
            method=(
                "Measured worst-case fixed point/ingestion plus conservative per-draw costs; "
                "25% allowance"
            ),
        )
        write_json(gate, output / "resource-gate.json")
        if not gate["passed"]:
            raise ValueError("Full-retention preflight resource/audit gate failed")
    elif elapsed > config.max_main_seconds or _bytes(output) > config.artifact_cap_bytes:
        raise ValueError("Full-retention main runtime/storage guard exceeded")
    return result
