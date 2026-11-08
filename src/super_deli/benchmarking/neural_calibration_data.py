"""Prespecified Gaussian/signal laws and one independently checked heavy-tail law."""

import numpy as np

from super_deli.benchmarking.matched_config import SETTINGS
from super_deli.benchmarking.separate_gmm_data import generate as old_generate
from super_deli.benchmarking.separate_gmm_data import reference as old_reference


def generate(task):
    if task["case"] != "heavy_tail":
        return old_generate(task)
    result = old_generate({**task, "case": "gaussian_independent"})
    rng = np.random.default_rng(task["data_seed"])
    residual = rng.normal(size=(2, task["n"], 2))
    latent = residual / np.sqrt(rng.chisquare(3, size=(2, task["n"], 1)) / 3)
    for i, role in enumerate(("morphology", "ecology")):
        x = latent[i][result["order"]]
        load = np.asarray(SETTINGS["loading_" + role], dtype=float)
        load /= np.linalg.norm(load, axis=1)[:, None]
        sl = slice(0, 4) if i == 0 else slice(4, 12)
        result["latent_" + role] = x
        result[role] = x @ load.T + result["noise"][:, sl]
    return result


def reference(task):
    if task["case"] != "heavy_tail":
        return old_reference(task)
    r = old_reference({**task, "case": "gaussian_independent"})
    rng = np.random.default_rng(task["data_seed"])
    normals = rng.standard_normal((2, task["n"], 2))
    radial = rng.chisquare(3, (2, task["n"], 1))
    for j, role in enumerate(("morphology", "ecology")):
        x = np.empty((task["n"], 2))
        for row, original in enumerate(r["order"]):
            x[row] = normals[j, original] * np.sqrt(3 / radial[j, original, 0])
        loadings = np.array(SETTINGS["loading_" + role], dtype=float)
        loadings /= np.sqrt((loadings * loadings).sum(1))[:, None]
        sl = slice(0, 4) if j == 0 else slice(4, 12)
        r["latent_" + role] = x
        r[role] = np.einsum("ij,kj->ik", x, loadings) + r["noise"][:, sl]
    return r
