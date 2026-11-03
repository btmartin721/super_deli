"""New seed streams and mirrored signal placement; no changes to archived simulators."""

import numpy as np

from super_deli.benchmarking.matched_config import SETTINGS
from super_deli.benchmarking.matched_data import generate as original_generate


def generate(task):
    if task["case"] != "ecology_only":
        return original_generate(task)
    n = task["n"]
    em, ee = np.random.default_rng(task["data_seed"]).normal(size=(2, n, 2))
    truth = np.repeat(np.arange(4), n // 4)
    centers = np.array([[-3, -3], [-3, 3], [3, -3], [3, 3]])
    um, ue = em, centers[truth] + ee
    noise = np.random.default_rng(task["noise_seed"]).normal(0, 0.15, (n, 12))
    order = np.random.default_rng(task["permutation_seed"]).permutation(n)
    lm, le = (
        np.array(SETTINGS[name], dtype=float) for name in ("loading_morphology", "loading_ecology")
    )
    lm /= np.linalg.norm(lm, axis=1)[:, None]
    le /= np.linalg.norm(le, axis=1)[:, None]
    return {
        "morphology": (um @ lm.T + noise[:, :4])[order],
        "ecology": (ue @ le.T + noise[:, 4:])[order],
        "latent_morphology": um[order],
        "latent_ecology": ue[order],
        "truth": truth[order],
        "noise": noise[order],
        "order": order,
        "ecology_order": np.random.default_rng(task["ecology_order_seed"]).permutation(n),
    }


def reference(task):
    from super_deli.benchmarking.matched_reference import reference_data

    if task["case"] != "ecology_only":
        return reference_data(task)
    # Reuse the independently derived morphology-only law, swapping its latent
    # locations before applying the original, unequal-dimensional loadings.
    source = reference_data({**task, "case": "morphology_only"})
    n = task["n"]
    rng = np.random.default_rng(task["data_seed"])
    em = rng.standard_normal((n, 2))
    ee = rng.standard_normal((n, 2))
    order, truth = source["order"], source["truth"]
    ue = ee[order] + np.array([[-3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [3.0, 3.0]])[truth]
    um = em[order]
    lm, le = (
        np.array(SETTINGS[name], dtype=float) for name in ("loading_morphology", "loading_ecology")
    )
    lm /= np.sqrt((lm * lm).sum(1))[:, None]
    le /= np.sqrt((le * le).sum(1))[:, None]
    source.update(
        morphology=np.einsum("ij,kj->ik", um, lm) + source["noise"][:, :4],
        ecology=np.einsum("ij,kj->ik", ue, le) + source["noise"][:, 4:],
        latent_morphology=um,
        latent_ecology=ue,
    )
    return source
