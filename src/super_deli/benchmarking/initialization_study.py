"""Full native pipeline comparisons with paired raw-specimen resampling."""

import shutil
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from super_deli.benchmarking.full_retention_run import run_native
from super_deli.benchmarking.initialization_audit import audit_initialization, audit_phase
from super_deli.benchmarking.initialization_config import tasks
from super_deli.benchmarking.signal_loss_fit import observe_candidates
from super_deli.benchmarking.signal_loss_source import digest
from super_deli.benchmarking.signal_loss_study import load_generator
from super_deli.io.config import read_json, write_json
from super_deli.models.full_retention import full_retention_model
from super_deli.models.initialization import truth_free_model


def size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def native(dataset, model, base, directory, law, truth):
    start = time.monotonic()
    row = run_native(
        dataset,
        model,
        base,
        directory,
        law=law,
        truth=truth,
        auditor=audit_initialization,
        observer=partial(observe_candidates, initialization=True),
    )
    row.update(path=str(directory), seconds=time.monotonic() - start, bytes=size(directory))
    write_json(row, Path(directory) / "timing-and-result.json")
    return row


def replay_parity(source, prefix, current):
    count = 0
    for name in source.inventory:
        if not name.startswith(str(prefix) + "/"):
            continue
        relative = Path(name).relative_to(prefix)
        if (
            relative.parts[0] in {"observed-candidates", "observed-pipeline"}
            or relative.name.startswith(("stability", "coassignment"))
            or relative.suffix not in {".npy", ".npz"}
        ):
            continue
        original = source.path(name)
        other = Path(current) / relative
        assert other.is_file()
        if relative.suffix == ".npy":
            np.testing.assert_array_equal(
                np.load(original, allow_pickle=False), np.load(other, allow_pickle=False)
            )
            count += 1
        else:
            with (
                np.load(original, allow_pickle=False) as a,
                np.load(other, allow_pickle=False) as b,
            ):
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
                    count += 1
    assert count > 0
    return count


def study(source, config, output, phase, generator):
    if config.fixture:
        raise ValueError("Invented seals cannot authorize scientific validation")
    output = Path(output)
    output.mkdir(exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    frozen = dict(
        runtime_source_sha256={
            p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob("*.py"))
        },
        config=config.to_dict(),
        main=tasks("main"),
        preflight=tasks("preflight"),
        generator_sha256=digest(generator),
        base_generator_sha256=digest(Path(generator).parents[1] / "matched-genomic/smoke.py"),
        model_seed=301,
        joint_pcs=8,
        view_pcs=2,
        max_k=4,
        minimum=9,
        arms=[
            dict(name="legacy10", starts=10),
            dict(name="legacy100", starts=100),
            dict(
                name="diverse100",
                batches=[["kmeans", 34], ["k-means++", 33], ["random_from_data", 33]],
            ),
        ],
        main_searches=2880,
        preflight_searches=785,
    )
    if (output / "freeze.json").exists():
        if read_json(output / "freeze.json") != frozen:
            raise ValueError("Scientific freeze changed")
    else:
        write_json(frozen, output / "freeze.json")
    if phase == "main" and not read_json(output / "resource-gate.json")["passed"]:
        raise ValueError("Passed independent preflight required")
    if shutil.disk_usage(output).free < config.min_free_bytes:
        raise ValueError("Free-space reserve failed")
    started = time.monotonic()
    root = output / phase
    root.mkdir()
    generate = load_generator(generator)
    records = []
    searches = parity = 0
    for task in tasks(phase):
        data = root / f"fresh-{task['index']:03d}-inputs"
        case = root / f"fresh-{task['index']:03d}"
        case.mkdir()
        dataset, analysis = generate(data, seed=task["seed"], law=task["law"], n=task["n"], draws=0)
        truth = read_json(data / "truth-evaluation-only.json")
        base = full_retention_model(
            replace(analysis.arms[-1].model, n_bootstrap=25 if task["stability"] else 0),
            min_cluster_size=9,
        )
        assert (
            base.n_pcs == 8
            and base.n_init == 10
            and base.covariance_type == "tied"
            and base.random_seed == 301
        )
        models = dict(
            legacy10=base, legacy100=replace(base, n_init=100), diverse100=truth_free_model(base)
        )
        write_json(
            {name: model.to_dict() for name, model in models.items()}, case / "comparison-plan.json"
        )
        arms = {}
        for name, model in models.items():
            arms[name] = native(dataset, model, data, case / name, task["law"], truth)
            searches += 1 + model.n_bootstrap
        records.append(dict(origin="fresh", task=task, arms=arms, source_bytes=size(data)))
        print(phase, "fresh", task["index"], {k: r["status"] for k, r in arms.items()}, flush=True)
        if (
            size(output) > config.artifact_cap_bytes
            or shutil.disk_usage(output).free < config.min_free_bytes
        ):
            raise ValueError("Storage/free-space guard failed; retain evidence and stop")
    indices = (*range(6), *range(80, 86)) if phase == "main" else (9, 80)
    for index in indices:
        data = root / f"archive-{index:03d}-inputs"
        case = root / f"archive-{index:03d}"
        case.mkdir()
        dataset, analysis = source.stage(index, data)
        truth = read_json(data / "truth-evaluation-only.json")
        base = full_retention_model(
            replace(analysis.arms[-1].model, n_bootstrap=0), min_cluster_size=9
        )
        models = dict(legacy10=base)
        if index < 20:
            models["legacy100"] = replace(base, n_init=100)
        models["diverse100"] = truth_free_model(base)
        arms = {}
        for name, model in models.items():
            row = native(dataset, model, data, case / name, truth["law"], truth)
            arms[name] = row
            searches += 1
            if name != "diverse100" and row["status"] == "complete":
                parity += replay_parity(source, source.baseline(index, name), case / name)
        records.append(
            dict(
                origin="archived",
                index=index,
                n=64 if index < 20 else 256,
                law=truth["law"],
                arms=arms,
            )
        )
    source.reverify()
    then = time.monotonic()
    independent = audit_phase(root, records)
    independent["seconds"] = time.monotonic() - then
    elapsed = time.monotonic() - started
    expected = 785 if phase == "preflight" else 2880
    assert searches == expected
    result = dict(
        status="complete",
        phase=phase,
        records=records,
        candidate_searches=searches,
        exact_parity_arrays=parity,
        elapsed_seconds=elapsed,
        bytes=size(root),
        independent=independent,
        consumed=source.consumed,
    )
    write_json(result, root / "summary.json")
    if phase == "preflight":
        points = [r for x in records if x["origin"] == "archived" for r in x["arms"].values()]
        boots = [r for x in records if x["origin"] == "fresh" for r in x["arms"].values()]
        assert all(r["status"] == "complete" for r in [*points, *boots])
        projected_bytes = int(
            (
                2850 * max(r["bytes"] / 25 for r in boots)
                + 30 * max(r["bytes"] for r in points)
                + 212 * max(x["source_bytes"] for x in records if x["origin"] == "fresh")
            )
            * 1.25
            + size(output)
            + 512 * 1024**2
        )
        projected_time = (
            2850 * max(r["seconds"] / 25 for r in boots)
            + 30 * max(r["seconds"] for r in points)
            + independent["seconds"] * 5
        ) * 1.25
        gate = dict(
            passed=bool(
                elapsed < config.max_preflight_seconds
                and projected_time < config.max_main_seconds
                and projected_bytes < config.artifact_cap_bytes
                and shutil.disk_usage(output).free >= config.min_free_bytes
            ),
            preflight_seconds=elapsed,
            projected_serial_main_seconds=projected_time,
            projected_bytes=projected_bytes,
            free_bytes=shutil.disk_usage(output).free,
            method=("Worst boot workflow/25 and archived points; 25% allowance plus512MiB reserve"),
        )
        write_json(gate, output / "resource-gate.json")
        if not gate["passed"]:
            raise ValueError("Preflight resource/audit gate failed")
    elif elapsed > config.max_main_seconds or size(output) > config.artifact_cap_bytes:
        raise ValueError("Main runtime/storage guard failed")
    return result
