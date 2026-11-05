"""Read-only binding and reconstruction of the completed coordinate experiment."""

import hashlib
import tarfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from super_deli.benchmarking.coordinate_config import SOURCE_HEAD, SPACES, CoordinateStudy, tasks
from super_deli.benchmarking.neural_analysis import arrays, neural_forward, require
from super_deli.benchmarking.neural_moons_coordinates import safe_path
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def bind(source, output, *, fixture=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(
        not source.is_relative_to(output) and not output.is_relative_to(source),
        "Separate source/evidence roots required",
    )
    config = CoordinateStudy(fixture=fixture)
    files = {}

    def include(relative):
        files[relative] = fingerprint_file(safe_path(source, relative)).sha256

    if fixture:
        require(
            read_json(source / "fixture.json")["kind"] == "invented_numeric_coordinate_fixture",
            "Explicit invented engineering source required",
        )
        include("fixture.json")
    else:
        completed, frozen = read_json(source / "completion.json"), read_json(source / "frozen.json")
        require(
            completed["completed"] and completed["head_sha"] == SOURCE_HEAD,
            "Expected completed separate-encoder archive",
        )
        require(
            completed["scientific_fit_failures"] == 0 and completed["ci_jobs"] == 6,
            "Completed source validation required",
        )
        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl"):
            require(
                version(name) == frozen["software"][name], "Exact source numerical runtime: " + name
            )
        with tarfile.open(source / "frozen-sources.tar.gz") as bundle:
            original = {
                m.name: hashlib.sha256(bundle.extractfile(m).read()).hexdigest()
                for m in bundle.getmembers()
                if m.isfile()
            }
        require(original == frozen["source_sha256"], "Bound original source bundle")
        package = Path(__file__).parents[1]
        for name in (
            "models/mixture.py",
            "models/density_scores.py",
            "models/hdbscan_config.py",
            "models/tabular_neural.py",
            "models/matched_neural.py",
            "models/separate_neural.py",
            "benchmarking/moons_hierarchy.py",
            "benchmarking/matched_data.py",
            "benchmarking/view_weight_data.py",
        ):
            require(
                fingerprint_file(package / name).sha256 == original[name],
                "Unchanged kernel: " + name,
            )
        for name in (
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "main-inventory.json",
            "main-audit.json",
        ):
            include(name)
    inventory = None if fixture else read_json(source / "main-inventory.json")
    for task in tasks(config, "main"):
        prefix = "main/" + task["key"] + "/"
        for name in ("task.json", "generating-reference.npz", "fits.json"):
            include(prefix + name)
        actual = read_json(source / prefix / "task.json")
        require(
            all(actual[k] == v for k, v in task.items()), "Original observation identity/settings"
        )
        rows = {r["arm"]: r for r in read_json(source / prefix / "fits.json")}
        for space in SPACES:
            require(
                rows[space]["status"] == "available", "Every prespecified source space required"
            )
            for name in ("point.npz", "cohort.json", "model-config.json", "density-trace.npz"):
                include(prefix + space + "/" + name)
    if inventory is not None:
        for name, digest in files.items():
            if name.startswith("main/"):
                require(
                    inventory.get(name.removeprefix("main/")) == digest,
                    "Original inventory: " + name,
                )
    output.mkdir(exist_ok=True)
    record = dict(
        passed=True,
        fixture=fixture,
        replays=[],
        source_sha256=files,
        source_head=None if fixture else SOURCE_HEAD,
        observations=tasks(config, "main"),
        spaces=list(SPACES),
    )
    write_json(record, output / "replay-binding.json")
    check_binding(source, record)
    return dict(passed=True, observations=len(record["observations"]), bound_files=len(files))


def branch_projection(x, a):
    kept = a["kept_indices"]
    z = (x[:, kept] - a["means"][kept]) / a["scales"][kept]
    if "components" in a:
        raw = (z - a["pca_mean"]) @ a["components"].T
    else:
        raw = neural_forward(z, a, "encoder")
    return (raw - a["latent_means"]) / a["latent_scales"]


def reconstruct(raw, point, space):
    if space.startswith("joint"):
        x = np.column_stack((raw["morphology"], raw["ecology"]))
        kept = point["kept_indices"]
        z = (x[:, kept] - point["means"][kept]) / point["scales"][kept]
        z *= point["column_weights"]
        encoded = neural_forward(z, point, "encoder")
        return (encoded - point["latent_means"]) / point["latent_scales"]
    branches = []
    for role in ("morphology", "ecology"):
        a = {k.removeprefix(role + "__"): v for k, v in point.items() if k.startswith(role + "__")}
        branches.append(branch_projection(raw[role], a))
    return np.column_stack(branches)


def load_context(source, task):
    base = safe_path(source, "main/" + task["key"])
    raw = arrays(base / "generating-reference.npz")
    require(
        raw["morphology"].shape == (task["n"], 4) and raw["ecology"].shape == (task["n"], 8),
        "Declared complete raw views",
    )
    context = {"reference": raw, "spaces": {}, "source_indices": np.arange(task["n"])}
    expected_ids = [f"unit_{i:04d}" for i in range(task["n"])]
    for space in SPACES:
        folder = base / space
        cohort = read_json(folder / "cohort.json")
        require(cohort["unit_ids"] == expected_ids, "Exact matched-unit identity/order")
        for i, role in enumerate(("morphology", "ecology")):
            view = cohort["views"][i]
            require(
                view["modality"] == role
                and view["observation_ids"] == [f"{role}_{j:04d}" for j in range(task["n"])],
                "Observation correspondence",
            )
            np.testing.assert_array_equal(
                view["source_row_indices"],
                np.arange(task["n"]) if i == 0 else np.argsort(raw["ecology_order"]),
            )
        point = arrays(folder / "point.npz")
        z = point["scores"]
        d = 2 if space.startswith("joint2_") else 4
        require(z.shape == (task["n"], d) and np.isfinite(z).all(), "Original finite score space")
        np.testing.assert_allclose(reconstruct(raw, point, space), z, rtol=1e-10, atol=1e-11)
        trace = arrays(folder / "density-trace.npz")
        index = 1 if space.startswith("separate_") else 0
        context["spaces"][space] = dict(
            scores=z,
            labels=point["labels"],
            strength=point["strength"],
            linkage=trace[f"fit_{index}_linkage"],
            condensed=trace[f"fit_{index}_condensed"],
        )
    return context
