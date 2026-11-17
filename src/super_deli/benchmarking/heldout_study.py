"""Frozen full-pipeline validation with separately counted native K=1 score heads."""

import importlib.metadata
import platform
import shutil
import time
from pathlib import Path

from super_deli.benchmarking.bootstrap_split_study import parity
from super_deli.benchmarking.heldout_config import tasks
from super_deli.benchmarking.heldout_run import evaluate_case, parent, subsample
from super_deli.benchmarking.initialization_study import size
from super_deli.benchmarking.signal_loss_source import digest
from super_deli.io.config import read_json, write_json
from super_deli.io.individual_subsample_source import load_point_source


def evaluation_inputs(case, model):
    import numpy as np

    from super_deli.models.integrative import IntegrationInput, _prepare

    root = Path(case) / "evaluation-ingestion"
    inputs = []
    for i, v in enumerate(model.views):
        path = root / f"view_{i:03d}"
        r = read_json(path / "source_rows.json")
        assert digest(path / "source_values.npy") == r["array_sha256"]
        values = np.load(path / "source_values.npy", allow_pickle=False)
        inputs.append(
            IntegrationInput(
                v.modality,
                v.kind,
                r["source_level"],
                values,
                tuple(r["observation_ids"]),
                tuple(r["unit_ids"]),
                tuple(r["target_ids"]),
                tuple(r["feature_names"]),
                tuple(r["feature_types"]),
                tuple(r["landmark_ids"]),
            )
        )
    inputs, units, excluded = _prepare(inputs, model)
    assert not excluded
    return inputs, units


def replay(source, index, directory):
    directory = Path(directory)
    directory.mkdir()
    started = time.monotonic()
    case = source.case(index)
    source.stage_point(index, directory / "point")
    point = load_point_source(directory / "point")
    evaluation = evaluation_inputs(case, point.model)
    ss, sa = subsample(directory / "point", directory / "subsample", evaluation, 25)
    exact = parity(
        source, Path(f"validation/main/fresh-{index:03d}/subsample"), directory / "subsample"
    )
    # Only truth metadata is needed for post-fit evaluation; raw inputs stay in the sealed source.
    data = directory / "observed"
    data.mkdir()
    shutil.copyfile(
        source.path(f"validation/main/fresh-{index:03d}/observed/truth-evaluation-only.json"),
        data / "truth-evaluation-only.json",
    )
    truth = read_json(data / "truth-evaluation-only.json")
    task = dict(
        index=index, n=len(point.units), counts=truth["group_counts"], design="legacy_replay"
    )
    result = evaluate_case(
        directory, task, point, ss, sa, dict(withheld=0), native_searches=25, seconds=started
    )
    result["exact_arrays"] = exact
    write_json(dict(passed=True, exact_arrays=exact), directory / "replay-parity.json")
    return result


def study(source, config, output, phase, generator):
    if config.fixture:
        raise ValueError("Invented source seals cannot authorize the validation")
    output = Path(output)
    output.mkdir(exist_ok=True)
    generator = Path(generator)
    package = Path(__file__).resolve().parents[1]
    frozen = dict(
        configuration=config.to_dict(),
        source={
            p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob("*.py"))
        },
        python=platform.python_version(),
        libraries={
            k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn", "snpio")
        },
        generator=digest(generator),
        base_generator=digest(generator.parents[1] / "matched-genomic/smoke.py"),
        tasks={p: tasks(p) for p in ("preflight", "main")},
        model_seed=301,
        sampling_seed=1601,
        draws=25,
        main_searches=10400,
        preflight_searches=610,
        main_native=5300,
        main_baseline=5100,
        preflight_native=310,
        preflight_baseline=300,
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
    parents = []
    replays = []
    limit = config.max_preflight_seconds if phase == "preflight" else config.max_main_seconds

    def guard():
        if (
            size(output) > config.artifact_cap_bytes
            or shutil.disk_usage(output).free < config.min_free_bytes
            or time.monotonic() - started > limit
        ):
            raise ValueError("Resource guard failed; preserve evidence and stop")

    guard()
    for task in tasks(phase):
        parents.append(parent(task, root / f"fresh-{task['index']:03d}", generator))
        guard()
        print(phase, "fresh", task["index"], flush=True)
    for index in (0, 40) if phase == "preflight" else (0, 8, 16, 40):
        replays.append(replay(source, index, root / f"replay-{index:03d}"))
        guard()
    source.reverify()
    elapsed = time.monotonic() - started
    records = parents + replays
    native = sum(p["native_searches"] for p in records)
    baseline = sum(p["baseline_searches"] for p in records)
    planned = sum(p["planned_searches"] for p in records)
    assert planned == (610 if phase == "preflight" else 10400)
    result = dict(
        passed=True,
        phase=phase,
        parents=parents,
        replays=replays,
        native_searches=native,
        baseline_searches=baseline,
        planned_searches=planned,
        attempted_searches=sum(p["attempted_searches"] for p in records),
        numerical_withheld=sum(p["numerical_withheld"] for p in records),
        seconds=elapsed,
        bytes=size(root),
        source_files_consumed=len(source.consumed),
    )
    write_json(result, root / "summary.json")
    if phase == "preflight":
        projected = (
            200 * max(r["seconds"] for r in parents) + 4 * max(r["seconds"] for r in replays)
        ) * 1.25
        storage = int(
            (200 * max(r["bytes"] for r in parents) + 4 * max(r["bytes"] for r in replays)) * 1.25
            + size(output)
            + 512 * 1024**2
        )
        full_settings = all(p["attempted_searches"] == p["planned_searches"] for p in records)
        gate = dict(
            passed=full_settings
            and projected < config.max_main_seconds
            and storage < config.artifact_cap_bytes
            and elapsed < config.max_preflight_seconds
            and shutil.disk_usage(output).free >= config.min_free_bytes,
            full_settings=full_settings,
            projected_seconds=projected,
            projected_bytes=storage,
            preflight_seconds=elapsed,
            basis="Worst complete parent/replay;25percent allowance;512MiB reserve",
        )
        write_json(gate, output / "resource-gate.json")
        if not gate["passed"]:
            raise ValueError("Preflight gate failed; preserve and stop")
    return result
