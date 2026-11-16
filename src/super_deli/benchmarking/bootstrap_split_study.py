"""Bounded known-law reference experiment; native numerical algorithms stay unchanged."""

import importlib.metadata
import platform
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from super_deli.benchmarking.bootstrap_split_audit import audit
from super_deli.benchmarking.bootstrap_split_config import tasks
from super_deli.benchmarking.bootstrap_split_data import load
from super_deli.benchmarking.bootstrap_split_fit import run
from super_deli.benchmarking.bootstrap_split_metrics import measure, native_counts
from super_deli.benchmarking.bootstrap_split_simulation import cohort, parameters
from super_deli.benchmarking.initialization_study import native, size
from super_deli.benchmarking.signal_loss_source import digest
from super_deli.io.config import read_json, write_json
from super_deli.models.full_retention import full_retention_model
from super_deli.models.integrative_config import IntegrativeConfig


def parity(source, prefix, current):
    count = 0
    for name in sorted(source.inventory):
        if not name.startswith(str(prefix) + "/") or not name.endswith((".npz", ".npy")):
            continue
        left = source.arrays(name)
        other = Path(current) / Path(name).relative_to(prefix)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, np.load(other, allow_pickle=False))
            count += 1
        else:
            with np.load(other, allow_pickle=False) as right:
                assert set(left) == set(right.files)
                for key, value in left.items():
                    np.testing.assert_array_equal(value, right[key])
                    count += 1
    assert count > 0
    return count


def seeds(seed, n, draws):
    children = np.random.SeedSequence(seed).spawn(draws + 1)
    return [(np.arange(n), int(children[0].generate_state(1)[0]))] + [
        (
            np.random.default_rng(int(c.generate_state(2)[0])).integers(n, size=n),
            int(c.generate_state(2)[1]),
        )
        for c in children[1:]
    ]


def parent(task, directory, generator, *, head=False, draws=25, head_model=None):
    directory = Path(directory)
    directory.mkdir()
    start = time.monotonic()
    n = task["n"]
    innovations = np.random.SeedSequence([task["seed"], 20261115]).spawn(draws + 2)
    innovations = [int(c.generate_state(1)[0]) for c in innovations]
    assert len(set(innovations)) == len(innovations)
    if head:
        train = np.random.default_rng(innovations[0]).normal(size=(n, 8))
        evaluation = np.random.default_rng(innovations[1]).normal(size=(1024, 8))
        truth = np.zeros(n, dtype=int)
        model = head_model
        if model is None:
            raise ValueError("Explicit head configuration required")
        with (directory / "known-cohorts.npz").open("xb") as f:
            np.savez(f, train=train, evaluation=evaluation)
    else:
        parameter = parameters(task["seed"], task["law"])
        dataset, model, truth0 = cohort(
            directory / "observed", parameter, n, innovations[0], "o", generator
        )
        train = load(dataset, model, directory / "observed", directory / "observed-ingestion")
        ed, em, et = cohort(
            directory / "evaluation", parameter, 1024, innovations[1], "e", generator
        )
        assert not set(train[1]) & set(s.specimen_id for s in ed.specimens)
        evaluation = load(ed, em, directory / "evaluation", directory / "evaluation-ingestion")
        lookup = dict(zip(truth0["unit_ids"], truth0["labels"], strict=True))
        truth = np.array([lookup[u] for u in train[1]])
    assert (
        model.n_init == 100
        and model.n_pcs == 8
        and model.covariance_type == "tied"
        and model.random_seed == 301
        and model.min_cluster_size == 9
    )
    nativeseeds = seeds(301, n, draws)
    results = {}
    point = None
    paths = []
    write_json(
        dict(
            task=task,
            innovations=innovations,
            configuration=model.to_dict(),
            head_only=head,
            draws=draws,
        ),
        directory / "freeze.json",
    )
    if head:
        run(
            directory / "ordinary-point",
            train,
            train,
            evaluation,
            model,
            head=True,
            draw=nativeseeds[0][0],
            seed=nativeseeds[0][1],
        )
        a = audit(directory / "ordinary-point", train, evaluation, head=True)
        pointrows = a["rows"]
        paths.append(directory / "ordinary-point")
        for i, (draw, seed) in enumerate(nativeseeds[1:]):
            path = directory / f"ordinary-{i:02d}"
            run(path, train, train, evaluation, model, head=True, draw=draw, seed=seed)
            a = audit(path, train, evaluation, head=True)
            paths.append(path)
        ordinary = [read_json(p / "diagnostic-audit.json")["rows"][0] for p in paths]
    else:
        run(directory / "ordinary", train, train, evaluation, model, ordinary=True, draws=draws)
        a = audit(directory / "ordinary", train, evaluation)
        native_counts(directory / "ordinary", n, 301, draws)
        ordinary = a["rows"]
        pointrows = ordinary[:1]
        paths = [directory / "ordinary"]
    if pointrows[0]["status"] == "complete":
        with np.load(paths[0] / pointrows[0]["prediction_arrays"], allow_pickle=False) as f:
            point = f["original_labels"].copy()
    results["ordinary"] = []
    if head:
        for i, p in enumerate(paths):
            results["ordinary"].extend(
                measure(
                    p,
                    read_json(p / "diagnostic-audit.json")["rows"],
                    truth,
                    point,
                    [nativeseeds[i][0]],
                )
            )
    else:
        results["ordinary"] = measure(
            paths[0], ordinary, truth, point, [np.arange(n)] + [d for d, s in nativeseeds[1:]]
        )
    for arm in ("unique", "reference"):
        results[arm] = []
        for i, (draw, seed) in enumerate(nativeseeds[1:]):
            if arm == "unique":
                current = train
                indices = np.unique(draw)
            elif head:
                current = np.random.default_rng(innovations[i + 2]).normal(size=(n, 8))
                indices = np.arange(n)
            else:
                rd, rm, rt = cohort(
                    directory / f"reference-input-{i:02d}",
                    parameter,
                    n,
                    innovations[i + 2],
                    f"r{i:02d}",
                    generator,
                )
                assert not set(train[1]) & set(s.specimen_id for s in rd.specimens) and not set(
                    evaluation[1]
                ) & set(s.specimen_id for s in rd.specimens)
                current = load(
                    rd,
                    rm,
                    directory / f"reference-input-{i:02d}",
                    directory / f"reference-ingestion-{i:02d}",
                )
                indices = np.arange(n)
            path = directory / f"{arm}-{i:02d}"
            run(path, current, train, evaluation, model, head=head, draw=indices, seed=seed)
            a = audit(path, train, evaluation, head=head)
            results[arm].extend(
                measure(path, a["rows"], truth, point, [draw] if arm == "unique" else None)
            )
    result = dict(
        task=task,
        head_only=head,
        arms=results,
        seconds=time.monotonic() - start,
        bytes=size(directory),
        searches=1 + 3 * draws,
        attempted_searches=sum(
            read_json(p)["attempted_searches"] for p in directory.rglob("diagnostic-audit.json")
        ),
    )
    write_json(result, directory / "summary.json")
    return result


def study(source, config, output, phase, generator):
    if config.fixture:
        raise ValueError("Fixture cannot authorize scientific study")
    output = Path(output)
    output.mkdir(exist_ok=True)
    generator = Path(generator).resolve()
    package = Path(__file__).resolve().parents[1]
    frozen = dict(
        config=config.to_dict(),
        python=platform.python_version(),
        libraries={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "snpio")
        },
        source={
            p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob("*.py"))
        },
        generator=digest(generator),
        base_generator=digest(generator.parents[1] / "matched-genomic/smoke.py"),
        tasks={p: tasks(p) for p in ("preflight", "main")},
        protocol="ordinary100-known-raw-and-Gaussian-score-v1",
        draws=25,
        evaluation_n=1024,
        preflight_searches=660,
        main_searches=3856,
    )
    if (output / "freeze.json").exists():
        assert read_json(output / "freeze.json") == frozen
    else:
        write_json(frozen, output / "freeze.json")
    if phase == "main" and not read_json(output / "resource-gate.json")["passed"]:
        raise ValueError("Passed resource gate required")
    if shutil.disk_usage(output).free < config.min_free_bytes:
        raise ValueError("Free-space reserve failed")
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
            raise ValueError("Retain evidence: resource guard failed")

    for task in tasks(phase):
        records.append(parent(task, root / f"raw-{task['index']:02d}", generator))
        guard()
        print(phase, "raw", task["index"], flush=True)
    count = 4 if phase == "preflight" else 24
    base = 14100 if phase == "preflight" else 14200
    for i in range(count):
        task = dict(index=i, n=64 if i < count // 2 else 256, law="Gaussian_score", seed=base + i)
        records.append(
            parent(
                task,
                root / f"head-{i:02d}",
                generator,
                head=True,
                head_model=IntegrativeConfig.from_dict(
                    read_json(root / "raw-00/observed/model.json")
                ),
            )
        )
        guard()
        print(phase, "head", i, flush=True)
    for i in (0, 160) if phase == "preflight" else (0, 1, 60, 61, 100, 101, 160, 161):
        data = root / f"replay-{i:03d}-inputs"
        ds, analysis = source.stage(i, data)
        truth = read_json(data / "truth-evaluation-only.json")
        model = full_retention_model(
            replace(analysis.arms[-1].model, n_init=100, n_bootstrap=25), min_cluster_size=9
        )
        row = native(ds, model, data, root / f"replay-{i:03d}", truth["law"], truth)
        assert row["status"] == "complete"
        row["exact_arrays"] = parity(source, source.baseline(i), root / f"replay-{i:03d}")
        replays.append(row)
        guard()
    source.reverify()
    elapsed = time.monotonic() - started
    searches = sum(r["searches"] for r in records) + 26 * len(replays)
    assert searches == (660 if phase == "preflight" else 3856)
    result = dict(
        passed=True,
        phase=phase,
        parents=records,
        replays=replays,
        searches=searches,
        seconds=elapsed,
        bytes=size(root),
        source_consumed=source.consumed,
    )
    write_json(result, root / "summary.json")
    if phase == "preflight":
        raw = [r for r in records if not r["head_only"]]
        heads = [r for r in records if r["head_only"]]
        pt = (
            24 * max(r["seconds"] for r in raw)
            + 24 * max(r["seconds"] for r in heads)
            + 8 * max(r["seconds"] for r in replays)
        ) * 1.25
        pb = int(
            (
                24 * max(r["bytes"] for r in raw)
                + 24 * max(r["bytes"] for r in heads)
                + 8 * max(r["bytes"] for r in replays)
            )
            * 1.25
            + size(output)
            + 512 * 1024**2
        )
        gate = dict(
            passed=pt < config.max_main_seconds
            and pb < config.artifact_cap_bytes
            and elapsed < config.max_preflight_seconds
            and shutil.disk_usage(output).free >= config.min_free_bytes,
            projected_seconds=pt,
            projected_bytes=pb,
            preflight_seconds=elapsed,
            free_bytes=shutil.disk_usage(output).free,
            method="Worst parent by type, 25 percent allowance, 512 MiB reserve",
        )
        write_json(gate, output / "resource-gate.json")
        if not gate["passed"]:
            raise ValueError("Retain evidence: preflight gate failed")
    return result
