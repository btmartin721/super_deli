"""Source-bound audit, separately gated preflight and frozen point-diagnostic study."""

import importlib.util
import shutil
import time
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.signal_loss_config import SignalLossConfig, fresh_tasks
from super_deli.benchmarking.signal_loss_run import run_case
from super_deli.benchmarking.signal_loss_source import digest
from super_deli.benchmarking.signal_loss_trace import public_trace, read_context
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json
from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig


def load_generator(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("signal_loss_invented_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.invented


def audit_source(source, output):
    output = Path(output)
    output.mkdir()
    rows = []
    for entry in source.entries():
        base = Path(entry["analysis"]) / "arms"
        context = read_context(
            source,
            base / "combined",
            base / "genomic",
            base / "shape",
            Path(entry["source"]) / "truth-evaluation-only.json",
        )
        rows.append(
            dict(index=entry["index"], law=entry["law"], n=entry["n"], **public_trace(context))
        )
    source.reverify()
    result = dict(
        status="complete",
        fit_free=True,
        observations=len(rows),
        records=rows,
        fixture=source.config.fixture,
        decision="not_evaluated",
        consumed=source.consumed,
    )
    write_json(result, output / "summary.json")
    return result


def archive_inputs(source, entry, directory):
    directory = Path(directory)
    directory.mkdir()
    manifest = source.json(Path(entry["analysis"]) / "manifest.json")
    dataset = config_from_dict(manifest["frozen_dataset"])
    for m in dataset.modalities:
        for value in (m.path, m.popmapfile):
            if value is None:
                continue
            original = source.path(Path(entry["analysis"]) / "inputs" / value)
            target = directory / value
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            if digest(target) != digest(original):
                raise ValueError("Input snapshot copy mismatch")
    truth = source.json(Path(entry["source"]) / "truth-evaluation-only.json")
    write_json(truth, directory / "truth-evaluation-only.json")
    return dataset, MultimodalAnalysisConfig.from_dict(manifest["analysis"])


def study(source, config, *, output_dir, phase, generator_path):
    config = SignalLossConfig.from_dict(config.to_dict())
    if config.fixture:
        raise ValueError("Invented source fixtures permit fit-free audit only")
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    document = dict(
        configuration=config.to_dict(),
        main=fresh_tasks("main"),
        preflight=fresh_tasks("preflight"),
        source_head=source.completion["head_sha"],
        generator_sha256=digest(generator_path),
        main_searches=1080,
        preflight_searches=72,
    )
    freeze = output / "freeze.json"
    if freeze.exists():
        if read_json(freeze) != document:
            raise ValueError("Frozen settings changed")
    else:
        write_json(document, freeze)
    if phase == "main" and (
        not read_json(output / "resource-gate.json")["passed"]
        or not read_json(output / "preflight-audit.json")["passed"]
    ):
        raise ValueError("A passed independent preflight gate is required")
    start = time.monotonic()
    root = output / phase
    root.mkdir()
    records = []
    generator = load_generator(generator_path)
    if phase == "main":
        for entry in source.entries():
            case = root / f"archived-{entry['index']:02d}"
            data = root / f"archived-{entry['index']:02d}-inputs"
            dataset, analysis = archive_inputs(source, entry, data)
            result = run_case(
                dataset,
                analysis,
                base_dir=data,
                directory=case,
                truth_path=data / "truth-evaluation-only.json",
                law=entry["law"],
                origin="archived",
                source_reader=source,
                source_entry=entry,
            )
            records.append(dict(path=case.relative_to(output).as_posix(), **result))
            print("archived", entry["index"], result["status"], flush=True)
    for task in fresh_tasks(phase):
        data = root / f"fresh-{task['index']:02d}-inputs"
        case = root / f"fresh-{task['index']:02d}"
        dataset, analysis = generator(
            data, seed=task["seed"], law=task["law"], n=task["n"], draws=0
        )
        result = run_case(
            dataset,
            analysis,
            base_dir=data,
            directory=case,
            truth_path=data / "truth-evaluation-only.json",
            law=task["law"],
            origin="fresh",
        )
        records.append(dict(path=case.relative_to(output).as_posix(), task=task, **result))
        print("fresh", task["index"], result["status"], flush=True)
    if phase == "preflight":
        # Four explicit native-only archived replays; no cached score contrasts.
        from super_deli.benchmarking.multimodal_validation import equal_point_arrays
        from super_deli.workflows.genomic_analysis_adapters import run_genomic_arm

        for index in (0, 2, 4, 5):
            entry = source.entries()[index]
            data = root / f"replay-{index:02d}-inputs"
            dataset, analysis = archive_inputs(source, entry, data)
            for arm in analysis.arms:
                if arm.name not in {"genomic", "shape", "combined"}:
                    continue
                arm = replace(arm, model=replace(arm.model, n_bootstrap=0))
                run = root / f"replay-{index:02d}-{arm.name}"
                from super_deli.benchmarking.signal_loss_fit import (
                    audit_candidates,
                    observe_candidates,
                )

                candidate_dir = root / f"replay-{index:02d}-{arm.name}-candidates"
                with observe_candidates(candidate_dir):
                    run_genomic_arm(arm, dataset, data, run)
                audit_candidates(candidate_dir)
                prefix = Path(entry["analysis"]) / "arms" / arm.name
                for name in source.inventory:
                    if (
                        name.startswith(prefix.as_posix() + "/")
                        and name.endswith((".npy", ".npz"))
                        and not Path(name).name.startswith(("stability", "coassignment"))
                    ):
                        source.path(name)
                equal_point_arrays(source.root / prefix, run)
    from super_deli.benchmarking.signal_loss_audit import audit_phase

    independent = audit_phase(output, dict(records=records))
    write_json(independent, output / (phase + "-audit.json"))
    source.reverify()
    elapsed = time.monotonic() - start
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    result = dict(
        status="complete",
        phase=phase,
        observations=len(records),
        records=records,
        planned_searches=72 if phase == "preflight" else 1080,
        retained_bytes=size,
        elapsed_seconds=elapsed,
        consumed=source.consumed,
    )
    write_json(result, root / "summary.json")
    if phase == "preflight":
        gate = dict(
            passed=elapsed * 15 * 1.25 < config.max_projected_seconds
            and size * 15 * 1.25 + 128 * 1024**2 < config.artifact_cap_bytes
            and shutil.disk_usage(output).free >= config.min_free_bytes,
            projected_serial_seconds=elapsed * 15 * 1.25,
            projected_bytes=int(size * 15 * 1.25 + 128 * 1024**2),
            free_bytes=shutil.disk_usage(output).free,
        )
        write_json(gate, output / "resource-gate.json")
        if not gate["passed"]:
            raise ValueError("Preflight resource gate failed")
    elif elapsed > config.max_projected_seconds or size > config.artifact_cap_bytes:
        raise ValueError("Main runtime/storage guard exceeded")
    return result
