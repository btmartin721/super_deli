"""Unchanged native workflows with read-only, worker-local training/candidate traces."""

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.matched_config import ARMS
from super_deli.benchmarking.matched_data import configuration, generate, write_dataset
from super_deli.benchmarking.unknown_k_run import retain_gmm_candidates
from super_deli.io.config import read_json, write_json


@contextmanager
def retain_initial_weights(out, fit_events):
    from super_deli.models import tabular_neural as numerical

    original = numerical._network
    events = []
    saved = {}

    def network(d):
        result = original(d)
        index = fit_events[-1]["fit"]
        events.append({"fit": index, "dimensions": d})
        saved.update({f"fit_{index}_{k}": v for k, v in numerical._weights(result).items()})
        return result

    numerical._network = network
    try:
        yield events
    finally:
        numerical._network = original
        out = Path(out)
        out.mkdir(exist_ok=True)
        np.savez_compressed(out / "initial-weights.npz", **saved)
        write_json(events, out / "training-events.json")


@contextmanager
def retain_fit_times(arm, out):
    if arm.startswith("paired_"):
        from super_deli.models import matched_neural as module

        name = "fit_matched_point"
    elif arm.endswith("density"):
        from super_deli.models import tabular_hdbscan as module

        name = "fit_density_once"
    else:
        from super_deli.models import tabular_neural as module

        name = "fit_neural_point"
    original = getattr(module, name)
    events = []

    def fit(*args, **kwargs):
        start = time.monotonic()
        event = {"fit": len(events), "n": len(args[0])}
        events.append(event)
        try:
            result = original(*args, **kwargs)
            event["status"] = "complete"
            return result
        except Exception as exc:
            event.update(status="failed", error=str(exc))
            raise
        finally:
            event["seconds"] = time.monotonic() - start

    setattr(module, name, fit)
    try:
        yield events
    finally:
        setattr(module, name, original)
        Path(out).mkdir(exist_ok=True)
        write_json(events, Path(out) / "fit-times.json")


def run_arm(task, raw, canonical, dataset, canonical_dataset, out, arm, *, replay=False):
    from super_deli.workflows.integrative import run_integrative
    from super_deli.workflows.matched_neural import run_matched_neural
    from super_deli.workflows.tabular_gmm import run_tabular_gmm
    from super_deli.workflows.tabular_hdbscan import run_tabular_hdbscan
    from super_deli.workflows.tabular_neural import run_tabular_neural

    start = time.monotonic()
    seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
    neural = arm.rsplit("_", 1)[-1] in {"0", "101", "202"}
    draws = task["draws"] if not replay and (neural or arm.endswith("density")) else 0
    config = configuration(arm, seed, draws)
    events = []
    try:
        if neural:
            with (
                retain_fit_times(arm, out) as fit_events,
                retain_initial_weights(out, fit_events) as events,
            ):
                if arm.startswith("paired_"):
                    summary = run_matched_neural(
                        dataset, model_config=config, base_dir=raw, output_dir=out
                    )
                else:
                    summary = run_tabular_neural(
                        canonical_dataset,
                        model_config=config,
                        base_dir=canonical,
                        output_dir=out,
                        modality_name=arm.split("_")[0],
                    )
        elif arm.endswith("gmm"):
            with retain_gmm_candidates(out):
                if arm == "integrative_gmm":
                    summary = run_integrative(
                        canonical_dataset, model_config=config, base_dir=canonical, output_dir=out
                    )
                else:
                    summary = run_tabular_gmm(
                        canonical_dataset,
                        model_config=config,
                        base_dir=canonical,
                        output_dir=out,
                        modality_name=arm.split("_")[0],
                    )
        else:
            with retain_fit_times(arm, out):
                summary = run_tabular_hdbscan(
                    canonical_dataset,
                    model_config=config,
                    base_dir=canonical,
                    output_dir=out,
                    modality_name=arm.split("_")[0],
                )
        record = {"status": "available", "selected_k": summary["selected_k"]}
    except Exception as exc:
        record = {"status": "failed", "selected_k": None, "error": str(exc)}
    subs = read_json(out / "subsamples.json") if (out / "subsamples.json").exists() else []
    result = {
        "arm": arm,
        "kind": "replay" if replay else "point",
        "seconds": time.monotonic() - start,
        "configuration": config.to_dict(),
        "draws_requested": draws,
        "draws_attempted": len(subs),
        "neural_fits_attempted": len(events),
        **record,
    }
    write_json(result, out / "execution.json")
    return result


def run_dataset(args):
    root, task = args
    out = Path(root) / task["key"]
    out.mkdir()
    write_json(task, out / "task.json")
    data = generate(task)
    np.savez_compressed(out / "generating-reference.npz", **data)
    ds = write_dataset(data, out / "raw")
    canonical = write_dataset(data, out / "canonical", canonical=True)
    records = []
    with threadpool_limits(limits=1):
        for arm in ARMS:
            records.append(
                run_arm(task, out / "raw", out / "canonical", ds, canonical, out / arm, arm)
            )
            if task["replay"]:
                records.append(
                    run_arm(
                        task,
                        out / "raw",
                        out / "canonical",
                        ds,
                        canonical,
                        out / (arm + "_replay"),
                        arm,
                        replay=True,
                    )
                )
    write_json(records, out / "fits.json")
    return {"key": task["key"], "n": task["n"], "records": records}
