"""Prespecified nonlinear measurements; references are never analytical features."""

import csv
from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_config import ECO, MORPH, SETTINGS
from super_deli.benchmarking.neural_power_data import generate as source_generate
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json


def cohort_task(task, cohort):
    number = {"train": 0, "iid": 1, "shifted": 2, "unfamiliar": 3}[cohort]
    n = (
        task["n"]
        if cohort == "train"
        else (32 if task["fixture"] else 512)
        if cohort == "unfamiliar"
        else 32
        if task["fixture"]
        else 2048
    )
    base = [*task["seed_path"], number]
    return dict(
        case="gaussian_independent"
        if task["case"] == "gaussian"
        else "continuous_gradient"
        if task["case"] == "gradient"
        else "four_complementary",
        n=n,
        signal_multiplier=0.5 if task["case"] in ("linear_weak", "confounded_size") else 1.0,
        **{
            name: base + [i]
            for i, name in enumerate(
                ("data_seed", "noise_seed", "permutation_seed", "ecology_order_seed", "size_seed")
            )
        },
    )


def generate(task, cohort):
    c = cohort_task(task, cohort)
    original = source_generate(c)
    result = {
        k: v.copy() for k, v in original.items() if k not in ("morphology_truth", "ecology_truth")
    }
    if cohort == "unfamiliar":
        truth = result["truth"]
        for role, axis, signs in (("morphology", 0, truth // 2), ("ecology", 1, truth % 2)):
            latent = result["latent_" + role]
            latent[:, axis] += 6 - np.where(signs == 0, -3.0, 3.0) * c["signal_multiplier"]
            loading = np.array(SETTINGS["loading_" + role], dtype=float)
            loading /= np.linalg.norm(loading, axis=1)[:, None]
            sl = slice(0, 4) if role == "morphology" else slice(4, 12)
            result[role] = latent @ loading.T + result["noise"][:, sl]
        result["truth"] = np.full(c["n"], 4, dtype=int)
    result["base_morphology"] = result["morphology"].copy()
    result["base_ecology"] = result["ecology"].copy()
    size = np.random.default_rng(c["size_seed"]).normal(0, 0.5, c["n"])[result["order"]]
    if task["case"] == "confounded_size" and cohort != "shifted":
        size += 0.8 * (
            np.ones(c["n"])
            if cohort == "unfamiliar"
            else np.where(result["truth"] // 2 == 0, -1.0, 1.0)
        )
    result["size"] = size
    if task["case"] in ("allometric", "combined", "confounded_size"):
        result["morphology"] = np.exp(
            0.35 * result["morphology"] + size[:, None] * np.array([0.8, 1.0, 1.2, 1.4])
        )
    if task["case"] in ("saturating", "combined"):
        result["ecology"] = 20 + 5 * np.tanh(result["ecology"] / 2)
    if task["case"] == "confounded_size":
        result["ecology"] += 0.4 * size[:, None] * np.array([1.0, -1.0] * 4)
    return result


def write_cohort(data, directory, *, tag):
    directory = Path(directory)
    directory.mkdir()
    n = len(data["morphology"])
    units = [f"{tag}_unit_{i:04d}" for i in range(n)]
    modalities = []
    for role, kind, features in (
        ("morphology", "morphological", MORPH),
        ("ecology", "ecological", ECO),
    ):
        order = np.arange(n) if role == "morphology" else data["ecology_order"]
        with (directory / (role + ".tsv")).open("x", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["observation_id", *features])
            writer.writerows([f"{tag}_{role}_{i:04d}", *data[role][i]] for i in order)
        modalities.append(
            dict(
                name=role,
                kind=kind,
                level="specimen",
                format="tsv",
                path=role + ".tsv",
                features=[dict(name=f, dtype="continuous") for f in features],
                observations=[
                    dict(observation_id=f"{tag}_{role}_{i:04d}", unit_id=u)
                    for i, u in enumerate(units)
                ],
            )
        )
    dataset = config_from_dict(
        dict(
            schema_version=1,
            dataset_id=tag,
            specimens=[dict(specimen_id=u) for u in units],
            modalities=modalities,
        )
    )
    write_json(dataset.to_dict(), directory / "dataset.json")
    return dataset, units


def write_labels(units, y, path):
    with Path(path).open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["unit_id", "reference_label"])
        writer.writerows((u, f"reference_{int(label)}") for u, label in zip(units, y, strict=True))
