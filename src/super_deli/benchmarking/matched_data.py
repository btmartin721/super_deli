"""Source-separated synthetic laws and lossless canonical baseline representations."""

import csv
from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_config import ECO, MORPH, SETTINGS
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json


def generate(task):
    rng = np.random.default_rng(task["data_seed"])
    n = task["n"]
    case = task["case"]
    em, ee = rng.normal(size=(2, n, 2))
    truth = mt = et = None
    centers = np.array([[-3, -3], [-3, 3], [3, -3], [3, 3]])
    if case == "gaussian_independent":
        um, ue = em, ee
        truth = np.zeros(n, dtype=int)
    elif case in ("gaussian_correlated", "heavy_tail"):
        z = (
            rng.normal(size=(n, 2))
            if case == "gaussian_correlated"
            else rng.standard_t(3, size=(n, 2)) / np.sqrt(3)
        )
        um, ue = z + 0.5 * em, z + 0.5 * ee
        if case == "gaussian_correlated":
            truth = np.zeros(n, dtype=int)
    elif case == "discordant":
        mt = np.repeat(np.arange(4), n // 4)
        et = np.tile(np.repeat(np.arange(4), n // 16), 4)
        um, ue = centers[mt] + em, centers[et] + ee
    elif case == "moons":
        angles = rng.uniform(0, np.pi, (2, n // 2))
        a, b = angles
        z = np.concatenate(
            (
                np.column_stack((np.cos(a), np.sin(a))),
                np.column_stack((1 - np.cos(b), 0.5 - np.sin(b))),
            )
        )
        z += rng.normal(0, 0.08, z.shape)
        um, ue = z + 0.1 * em, z + 0.1 * ee
        truth = np.repeat([0, 1], n // 2)
    else:
        truth = np.repeat(np.arange(4), n // 4)
        if case == "four_complementary":
            c = centers[truth]
            um = np.column_stack((c[:, 0], np.zeros(n))) + em
            ue = np.column_stack((np.zeros(n), c[:, 1])) + ee
            mt = truth // 2
            et = truth % 2
        elif case in ("four_redundant", "morphology_only"):
            um = centers[truth] + em
            ue = centers[truth] + ee if case == "four_redundant" else ee
        else:
            raise ValueError("Unknown simulation law")
    order = np.random.default_rng(task["permutation_seed"]).permutation(n)
    noise = np.random.default_rng(task["noise_seed"]).normal(0, 0.15, (n, 12))
    lm = np.array(SETTINGS["loading_morphology"], dtype=float)
    lm /= np.linalg.norm(lm, axis=1)[:, None]
    le = np.array(SETTINGS["loading_ecology"], dtype=float)
    le /= np.linalg.norm(le, axis=1)[:, None]
    result = {
        "morphology": (um @ lm.T + noise[:, :4])[order],
        "ecology": (ue @ le.T + noise[:, 4:])[order],
        "latent_morphology": um[order],
        "latent_ecology": ue[order],
        "noise": noise[order],
        "order": order,
        "ecology_order": np.random.default_rng(task["ecology_order_seed"]).permutation(n),
    }
    for name, value in (("truth", truth), ("morphology_truth", mt), ("ecology_truth", et)):
        if value is not None:
            result[name] = value[order]
    return result


def write_dataset(data, directory, *, canonical=False):
    directory = Path(directory)
    directory.mkdir()
    n = len(data["morphology"])
    units = [f"unit_{i:04d}" for i in range(n)]
    modalities = []
    for name, kind, features in (
        ("morphology", "morphological", MORPH),
        ("ecology", "ecological", ECO),
    ):
        order = np.arange(n) if canonical or name == "morphology" else data["ecology_order"]
        with (directory / f"{name}.tsv").open("x", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["observation_id", *features])
            writer.writerows([f"{name}_{i:04d}", *data[name][i]] for i in order)
        modalities.append(
            {
                "name": name,
                "kind": kind,
                "level": "specimen",
                "format": "tsv",
                "path": f"{name}.tsv",
                "features": [{"name": f, "dtype": "continuous"} for f in features],
                "observations": [
                    {"observation_id": f"{name}_{i:04d}", "unit_id": u} for i, u in enumerate(units)
                ],
            }
        )
    dataset = config_from_dict(
        {
            "schema_version": 1,
            "dataset_id": "invented_matched_neural",
            "specimens": [{"specimen_id": u} for u in units],
            "modalities": modalities,
        }
    )
    write_json(dataset.to_dict(), directory / "dataset.json")
    return dataset


def configuration(arm, seed, draws=0):
    from super_deli.models.hdbscan_config import HDBSCANConfig
    from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
    from super_deli.models.matched_neural_config import MatchedNeuralConfig
    from super_deli.models.neural_config import NeuralConfig
    from super_deli.models.tabular_cluster_config import TabularClusterConfig

    if arm == "integrative_gmm":
        return IntegrativeConfig(
            assume_independent_units=True,
            n_pcs=2,
            min_cluster_size=5,
            n_bootstrap=0,
            random_seed=71323,
            views=(
                ViewConfig("morphology", "morphological", feature_names=MORPH),
                ViewConfig("ecology", "ecological", feature_names=ECO),
            ),
        )
    view, last = arm.rsplit("_", 1)
    if view == "paired":
        return MatchedNeuralConfig(
            "morphology",
            "ecology",
            MORPH,
            ECO,
            True,
            "individual",
            random_seed=int(last),
            subsampling_seed=seed,
            n_subsamples=draws,
        )
    features = MORPH if view == "morphology" else ECO
    if last == "gmm":
        return TabularClusterConfig(
            True,
            feature_names=features,
            n_pcs=2,
            min_cluster_size=5,
            n_bootstrap=0,
            random_seed=71323,
        )
    if last == "density":
        return HDBSCANConfig(True, feature_names=features, n_subsamples=draws, random_seed=seed)
    return NeuralConfig(
        features, True, random_seed=int(last), subsampling_seed=seed, n_subsamples=draws
    )
