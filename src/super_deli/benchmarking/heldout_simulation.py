"""Fixed-count shape alternatives with unchanged genomic/trait noise laws."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from super_deli.benchmarking.individual_subsample_simulation import generate as base_generate
from super_deli.io.config import read_json


def generate(directory, task, innovation, tag, generator, *, evaluation=False):
    directory = Path(directory)
    n = 1024 if evaluation else task["n"]
    counts = [x * (1024 // task["n"]) for x in task["counts"]] if evaluation else task["counts"]
    assert sum(counts) == n
    dataset, model, truth = base_generate(
        directory,
        parameter_seed=task["seed"],
        innovation=innovation,
        n=n,
        law="continuous_control",
        tag=tag,
        generator=generator,
    )
    if len(counts) > 1:
        p = read_json(directory / "known-parameters.json")
        y = np.repeat(np.arange(4), counts)
        np.random.default_rng(np.random.SeedSequence([innovation, 86])).shuffle(y)
        with np.load(directory / "known-raw.npz", allow_pickle=False) as f:
            arrays = {k: f[k].copy() for k in f.files}
        arrays["morphometric"] += 0.18 * (
            (2 * (y // 2) - 1)[:, None, None] * np.array(p["modes"])[0]
            + (2 * (y % 2) - 1)[:, None, None] * np.array(p["modes"])[1]
        )
        source = next(m for m in dataset.modalities if m.kind == "morphometric")
        spec = importlib.util.spec_from_file_location("heldout_layout_writer", generator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.write_shape(
            directory / source.path,
            arrays["morphometric"],
            [o.observation_id for o in source.observations],
            tuple(f"L{i + 1}" for i in range(6)),
        )
        with (directory / "known-raw.npz").open("wb") as f:
            np.savez(f, **arrays)
        truth.update(labels=y.tolist(), group_counts=counts, law="shape_unequal")
        p.update(group_counts=counts, law="shape_unequal")
        # Only this new draft is completed before ingestion/fitting.
        import json

        (directory / "known-parameters.json").write_text(
            json.dumps(p, indent=2, sort_keys=True) + "\n"
        )
        (directory / "truth-evaluation-only.json").write_text(
            json.dumps(truth, indent=2, sort_keys=True) + "\n"
        )
    return dataset, replace(model, n_bootstrap=0), truth
