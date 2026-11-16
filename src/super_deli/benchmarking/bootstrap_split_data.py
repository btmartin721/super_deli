"""Fit-free native ingestion and typed raw-array persistence for diagnostic cohorts."""

from pathlib import Path

import numpy as np

from super_deli.benchmarking.multimodal_validation import array_hash
from super_deli.io.config import write_json
from super_deli.models.integrative import _prepare
from super_deli.workflows.integrative import _load_view


def load(dataset, model, base, directory):
    audit_law(base)
    directory = Path(directory)
    directory.mkdir()
    by_name = {s.name: s for s in dataset.modalities}
    inputs = []
    provenance = []
    for i, v in enumerate(model.views):
        path = directory / f"view_{i:03d}"
        path.mkdir()
        data, evidence, excluded = _load_view(dataset, by_name[v.modality], v, model, base, path)
        assert not excluded
        inputs.append(data)
        provenance.append(evidence)
    inputs, units, excluded = _prepare(inputs, model)
    assert not excluded
    write_json(
        dict(
            unit_ids=units,
            provenance=provenance,
            array_sha256=[array_hash(d.values) for d in inputs],
        ),
        directory / "cohort.json",
    )
    # Independent generation-versus-reader check, including fixed A/C dosage orientation.
    from super_deli.io.config import read_json

    truth = read_json(Path(base) / "truth-evaluation-only.json")
    lookup = {u: i for i, u in enumerate(truth["unit_ids"])}
    with np.load(Path(base) / "known-raw.npz", allow_pickle=False) as a:
        for d in inputs:
            np.testing.assert_array_equal(d.values, a[d.kind][[lookup[u] for u in d.unit_ids]])
    return inputs, units


def audit_law(base):
    """Reconstruct known raw draws independently of readers and model fitting."""
    from super_deli.io.config import read_json

    base = Path(base)
    p = read_json(base / "known-parameters.json")
    truth = read_json(base / "truth-evaluation-only.json")
    n = len(truth["unit_ids"])
    rng = np.random.default_rng(truth["seed"])
    labels = np.arange(n) % 4 if p["law"] == "shape_only" else np.zeros(n, dtype=int)
    rng.shuffle(labels)
    np.testing.assert_array_equal(labels, truth["labels"])
    np.testing.assert_array_equal(
        p["allele_frequencies"], np.random.default_rng(p["seed"]).uniform(0.2, 0.8, 64)
    )
    expected = {
        "genomic": rng.binomial(2, p["allele_frequencies"], size=(n, 64)),
        "morphological": rng.normal(size=(n, 4)),
        "ecological": rng.normal(size=(n, 4)),
    }
    shape = np.asarray(p["base"])[None] + rng.normal(0, 0.025, size=(n, 6, 2))
    if p["law"] == "shape_only":
        shape += 0.18 * (
            (2 * (labels // 2) - 1)[:, None, None] * np.asarray(p["modes"])[0]
            + (2 * (labels % 2) - 1)[:, None, None] * np.asarray(p["modes"])[1]
        )
    expected["morphometric"] = shape
    with np.load(base / "known-raw.npz", allow_pickle=False) as a:
        assert set(a.files) == set(expected)
        for key, value in expected.items():
            np.testing.assert_array_equal(a[key], value)
    assert len(set(truth["unit_ids"])) == n and all(
        u.startswith(truth["tag"] + "_") for u in truth["unit_ids"]
    )
    return dict(passed=True, n=n, known_parameters_held_fixed=True)
