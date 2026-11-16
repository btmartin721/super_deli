"""Declared complete raw-data laws with balanced/unequal generating counts."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from super_deli.benchmarking.bootstrap_split_simulation import cohort, parameters
from super_deli.io.config import read_json
from super_deli.models.integrative import _prepare
from super_deli.workflows.integrative import _load_view


def generate(directory, *, parameter_seed, innovation, n, law, tag, generator):
    if law not in {"continuous_control", "shape_balanced", "shape_unequal"}:
        raise ValueError("Known study law required")
    p = parameters(parameter_seed, "continuous_control")
    dataset, model, truth = cohort(directory, p, n, innovation, tag, generator)
    directory = Path(directory)
    if law == "continuous_control":
        y = np.zeros(n, dtype=int)
        counts = [n]
    else:
        counts = (
            [n // 4] * 4
            if law == "shape_balanced"
            else [int(x * n // 64) for x in (25, 16, 13, 10)]
        )
        y = np.repeat(np.arange(4), counts)
        np.random.default_rng(np.random.SeedSequence([innovation, 86])).shuffle(y)
    with np.load(directory / "known-raw.npz", allow_pickle=False) as loaded:
        values = {k: loaded[k].copy() for k in loaded.files}
    if law != "continuous_control":
        values["morphometric"] += 0.18 * (
            (2 * (y // 2) - 1)[:, None, None] * np.array(p["modes"])[0]
            + (2 * (y % 2) - 1)[:, None, None] * np.array(p["modes"])[1]
        )
        source = next(m for m in dataset.modalities if m.kind == "morphometric")
        spec = importlib.util.spec_from_file_location("individual_layout_writer", generator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.write_shape(
            directory / source.path,
            values["morphometric"],
            [o.observation_id for o in source.observations],
            tuple(f"L{i + 1}" for i in range(6)),
        )
    # Replace only this freshly generated draft, before any ingestion or model fit.
    with (directory / "known-raw.npz").open("wb") as f:
        np.savez(f, **values)
    truth.update(labels=y.tolist(), law=law, group_counts=counts)
    (directory / "truth-evaluation-only.json").write_text(
        __import__("json").dumps(truth, indent=2, sort_keys=True) + "\n"
    )
    p.update(law=law, group_counts=counts, innovation=innovation, parameter_seed=parameter_seed)
    (directory / "known-parameters.json").write_text(
        __import__("json").dumps(p, indent=2, sort_keys=True) + "\n"
    )
    return dataset, replace(model, n_bootstrap=25), truth


def audit_law(directory):
    directory = Path(directory)
    p = read_json(directory / "known-parameters.json")
    truth = read_json(directory / "truth-evaluation-only.json")
    n = len(truth["unit_ids"])
    rng = np.random.default_rng(p["innovation"])
    zero = np.zeros(n, dtype=int)
    rng.shuffle(zero)
    expected = dict(
        genomic=rng.binomial(2, p["allele_frequencies"], size=(n, 64)),
        morphological=rng.normal(size=(n, 4)),
        ecological=rng.normal(size=(n, 4)),
    )
    shape = np.array(p["base"])[None] + rng.normal(0, 0.025, size=(n, 6, 2))
    y = np.repeat(np.arange(len(p["group_counts"])), p["group_counts"])
    if p["law"] != "continuous_control":
        np.random.default_rng(np.random.SeedSequence([p["innovation"], 86])).shuffle(y)
        shape += 0.18 * (
            (2 * (y // 2) - 1)[:, None, None] * np.array(p["modes"])[0]
            + (2 * (y % 2) - 1)[:, None, None] * np.array(p["modes"])[1]
        )
    expected["morphometric"] = shape
    np.testing.assert_array_equal(y, truth["labels"])
    np.testing.assert_array_equal(
        p["allele_frequencies"], np.random.default_rng(p["parameter_seed"]).uniform(0.2, 0.8, 64)
    )
    with np.load(directory / "known-raw.npz", allow_pickle=False) as a:
        for key, value in expected.items():
            np.testing.assert_array_equal(value, a[key])
    return dict(passed=True, n=n)


def load_evaluation(dataset, model, base, output):
    output = Path(output)
    output.mkdir()
    sources = {m.name: m for m in dataset.modalities}
    inputs = []
    for i, v in enumerate(model.views):
        d = output / f"view_{i:03d}"
        d.mkdir()
        data, evidence, excluded = _load_view(
            dataset, sources[v.modality], v, replace(model, n_bootstrap=0), base, d
        )
        assert not excluded
        inputs.append(data)
    inputs, units, excluded = _prepare(inputs, replace(model, n_bootstrap=0))
    assert not excluded
    truth = read_json(Path(base) / "truth-evaluation-only.json")
    lookup = {u: i for i, u in enumerate(truth["unit_ids"])}
    with np.load(Path(base) / "known-raw.npz", allow_pickle=False) as a:
        for d in inputs:
            np.testing.assert_array_equal(d.values, a[d.kind][[lookup[u] for u in d.unit_ids]])
    return inputs, units
