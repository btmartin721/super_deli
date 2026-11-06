"""Existing laws plus one declared covariance-misspecification stress law."""

import numpy as np

from super_deli.benchmarking.matched_config import SETTINGS
from super_deli.benchmarking.view_weight_data import generate as old_generate
from super_deli.benchmarking.view_weight_data import reference as old_reference

SCALES = (0.5, 1.0, 1.5, 2.0)


def generate(task):
    if task["case"] != "unequal_complementary":
        return old_generate(task)
    result = old_generate({**task, "case": "four_complementary"})
    truth = result["truth"]
    center = np.array([[-3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [3.0, 3.0]])[truth]
    for role, axis, start, stop in (("morphology", 0, 0, 4), ("ecology", 1, 4, 12)):
        c = np.zeros_like(center)
        c[:, axis] = center[:, axis]
        latent = c + (result["latent_" + role] - c) * np.asarray(SCALES)[truth, None]
        loading = np.asarray(SETTINGS["loading_" + role], dtype=float)
        loading /= np.linalg.norm(loading, axis=1)[:, None]
        result["latent_" + role] = latent
        result[role] = latent @ loading.T + result["noise"][:, start:stop]
    return result


def reference(task):
    if task["case"] != "unequal_complementary":
        return old_reference(task)
    result = old_reference({**task, "case": "four_complementary"})
    group = result["truth"]
    offset = np.column_stack((np.where(group < 2, -3.0, 3.0), np.where(group % 2 == 0, -3.0, 3.0)))
    for role, axis, sl in (("morphology", 0, slice(0, 4)), ("ecology", 1, slice(4, 12))):
        center = np.zeros((task["n"], 2))
        center[:, axis] = offset[:, axis]
        value = np.empty_like(center)
        for i in range(task["n"]):
            value[i] = center[i] + SCALES[int(group[i])] * (result["latent_" + role][i] - center[i])
        load = np.array(SETTINGS["loading_" + role], dtype=float)
        load /= np.sqrt(np.einsum("ij,ij->i", load, load))[:, None]
        result["latent_" + role] = value
        result[role] = np.einsum("ij,kj->ik", value, load) + result["noise"][:, sl]
    return result
