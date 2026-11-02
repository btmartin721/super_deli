"""Prespecified raw laws; truth remains outside all fitting calls."""

import json

import numpy as np

from super_deli.benchmarking.denoising_study import SETTINGS as DENOISING
from super_deli.benchmarking.density_study import generate_raw


def generate(task):
    n, case = task["n"], task["case"]
    rng = np.random.default_rng(task["data_seed"])
    if case == "gaussian_8d":
        z = rng.normal(size=(n, 8))
        truth = np.zeros(n, dtype=int)
    elif case == "heavy_tail":
        z = rng.standard_t(3, size=(n, 2)) / np.sqrt(3)
        truth = None
    elif case == "eight_spherical":
        angles = np.arange(8) * 2 * np.pi / 8
        centers = 8 * np.column_stack([np.cos(angles), np.sin(angles)])
        z = np.concatenate([rng.normal(m, 0.6, size=(n // 8, 2)) for m in centers])
        truth = np.repeat(np.arange(8), n // 8)
    else:
        z, truth = generate_raw(case, n, task["data_seed"])
    order = np.random.default_rng(task["permutation_seed"]).permutation(n)
    z = z[order]
    truth = truth[order] if truth is not None else None
    if case == "gaussian_8d":
        clean = z.copy()
    elif case == "moons":
        a, b = z.T
        clean = np.column_stack([a, b, a * b, a * a, b * b, np.sin(a), np.cos(b), np.sin(a + b)])
    else:
        clean = z @ np.asarray(DENOISING["linear_map"]).T
    noise = np.random.default_rng(task["noise_seed"]).normal(0, 0.15, clean.shape)
    result = {
        "latent": z,
        "clean": clean,
        "noise": noise,
        "noisy": clean + noise,
        "noise_mask": np.ones(clean.shape, dtype=bool),
        "order": order,
    }
    if truth is not None:
        result["truth"] = truth
    return result


def model_config(arm):
    from super_deli.benchmarking.neural_study import FEATURES
    from super_deli.models.hdbscan_config import HDBSCANConfig
    from super_deli.models.spectral_config import SpectralConfig
    from super_deli.models.tabular_cluster_config import TabularClusterConfig

    if arm in {"spectral_5", "spectral_15"}:
        return SpectralConfig(True, FEATURES, int(arm.split("_")[1]))
    if arm == "gmm":
        return TabularClusterConfig(True, feature_names=FEATURES, n_bootstrap=0, random_seed=71323)
    if arm == "density":
        return HDBSCANConfig(True, feature_names=FEATURES, random_seed=71323)
    raise ValueError("Unknown arm")


def arm_declarations():
    from super_deli.benchmarking.unknown_k_config import ARMS

    return json.loads(json.dumps({a: model_config(a).to_dict() for a in ARMS}))
