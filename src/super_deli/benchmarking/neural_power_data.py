"""Benchmark-only signal shifts and a continuous gradient; old laws remain intact."""

import numpy as np

from super_deli.benchmarking.matched_config import SETTINGS
from super_deli.benchmarking.neural_calibration_data import generate as old_generate
from super_deli.benchmarking.neural_calibration_data import reference as old_reference


def generate(task):
    case, n = task["case"], task["n"]
    if case != "continuous_gradient":
        result = old_generate(task)
        strength = task.get("signal_multiplier", 1.0)
        if case != "four_complementary" or strength == 1.0:
            return result
        truth = result["truth"]
        for role, axis, sl in (("morphology", 0, slice(0, 4)), ("ecology", 1, slice(4, 12))):
            center = np.zeros((n, 2))
            center[:, axis] = (
                np.where(truth // 2 == 0, -3.0, 3.0)
                if axis == 0
                else np.where(truth % 2 == 0, -3.0, 3.0)
            )
            latent = result["latent_" + role] + (strength - 1) * center
            load = np.array(SETTINGS["loading_" + role], dtype=float)
            load /= np.linalg.norm(load, axis=1)[:, None]
            result["latent_" + role] = latent
            result[role] = latent @ load.T + result["noise"][:, sl]
        return result
    result = old_generate({**task, "case": "gaussian_independent"})
    rng = np.random.default_rng(task["data_seed"])
    em, ee = rng.normal(size=(2, n, 2))
    gradient = rng.uniform(-3, 3, n)
    em[:, 0] += gradient
    ee[:, 1] += gradient
    for role, latent, sl in (("morphology", em, slice(0, 4)), ("ecology", ee, slice(4, 12))):
        x = latent[result["order"]]
        load = np.array(SETTINGS["loading_" + role], dtype=float)
        load /= np.linalg.norm(load, axis=1)[:, None]
        result["latent_" + role] = x
        result[role] = x @ load.T + result["noise"][:, sl]
    result.pop("truth")
    result["gradient"] = gradient[result["order"]]
    return result


def reference(task):
    """Independently reconstruct shifts/gradient, feature maps and shuffled rows."""
    case, n = task["case"], task["n"]
    if case != "continuous_gradient":
        result = old_reference(task)
        strength = task.get("signal_multiplier", 1.0)
        if case != "four_complementary" or strength == 1.0:
            return result
        for role, axis, sl in (("morphology", 0, slice(0, 4)), ("ecology", 1, slice(4, 12))):
            x = result["latent_" + role].copy()
            for i, label in enumerate(result["truth"]):
                bit = int(label) // 2 if axis == 0 else int(label) % 2
                x[i, axis] += (strength - 1) * (-3 if bit == 0 else 3)
            load = np.asarray(SETTINGS["loading_" + role], dtype=float)
            load = load / np.sqrt(np.sum(load * load, axis=1))[:, None]
            result["latent_" + role] = x
            result[role] = np.einsum("ij,kj->ik", x, load) + result["noise"][:, sl]
        return result
    result = old_reference({**task, "case": "gaussian_independent"})
    rng = np.random.default_rng(task["data_seed"])
    normal = rng.standard_normal((2, n, 2))
    gradient = rng.uniform(-3.0, 3.0, n)
    for j, role in enumerate(("morphology", "ecology")):
        x = np.empty((n, 2))
        for i, original in enumerate(result["order"]):
            x[i] = normal[j, original]
            x[i, j] += gradient[original]
        loading = np.asarray(SETTINGS["loading_" + role], dtype=float)
        loading /= np.sqrt(np.einsum("ij,ij->i", loading, loading))[:, None]
        result["latent_" + role] = x
        sl = slice(0, 4) if j == 0 else slice(4, 12)
        result[role] = np.einsum("ij,kj->ik", x, loading) + result["noise"][:, sl]
    result.pop("truth")
    result["gradient"] = np.array([gradient[i] for i in result["order"]])
    return result
