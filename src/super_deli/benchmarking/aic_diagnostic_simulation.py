"""Count-preserving Gaussian mechanism controls and frozen linear projections."""

from __future__ import annotations

import copy
from pathlib import Path

from super_deli.benchmarking.grouped_study import simulate_grouped


def decompose(case, n, seed):
    """Recover original latent means without changing the existing generator."""
    import numpy as np

    values, layout, _ = simulate_grouped(case, n, seed)
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 5, n)
    if case == "zero_conditional_gaussian":
        ecology = rng.standard_normal((n, 4))
        rng.standard_normal((n, 4))
        regression = np.array(
            [[0.6, 0.2, 0, 0], [0, 0.6, 0.2, 0], [0, 0, 0.6, 0.2], [0.2, 0, 0, 0.6]]
        )
        latent = ecology @ regression.T
    else:
        z = rng.standard_normal((n, 8))
        scale = np.linspace(0.5, 2, 8)
        covariance = (
            np.eye(8)
            if case == "independent_gaussian"
            else 0.6 ** np.abs(np.arange(8)[:, None] - np.arange(8)) * np.outer(scale, scale)
        )
        mean = np.zeros(8) if case == "independent_gaussian" else np.linspace(-2, 2, 8)
        latent = (mean + z @ np.linalg.cholesky(covariance).T)[:, :4]
    targets = np.repeat(np.arange(n), counts)
    residual = values["morphological"] - latent[targets]
    return values, layout, latent, residual


def homogeneous(values, layout, latent, residual, r):
    import numpy as np

    counts = np.asarray(layout["counts"])
    target = np.repeat(np.arange(len(counts)), counts)
    return {
        "morphological": latent[target] + residual * np.sqrt(counts[target, None] * r),
        "ecological": values["ecological"].copy(),
    }


def heldout_layout(layout):
    """Keep independent test identities disjoint from training identities."""
    result = copy.deepcopy(layout)
    result["unit_ids"] = ["heldout_" + u for u in result["unit_ids"]]
    for row in result["specimens"]:
        for key in row:
            row[key] = "heldout_" + row[key]
    for view in result["views"]:
        for key in ("unit_ids", "observation_ids", "target_ids"):
            view[key] = ["heldout_" + u for u in view[key]]
    return result


def project_saved(run, values, layout):
    """Apply exported training parameters only; fit no held-out transformation."""
    import numpy as np

    run = Path(run)
    blocks = []
    for j, kind in enumerate(("morphological", "ecological")):
        directory = run / f"view_{j:03d}" / "parameters"
        with np.load(directory / "pca.npz", allow_pickle=False) as p:
            raw = values[kind]
            source = (
                (raw[:, p["kept_indices"]] - p["feature_means"]) / p["feature_scales"] - p["mean"]
            ) @ p["components"].T
        if kind == "morphological":
            offsets = np.r_[0, np.cumsum(layout["counts"])]
            source = np.stack(
                [source[a:b].mean(0) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
            )
        with np.load(directory / "profiles.npz", allow_pickle=False) as p:
            blocks.append((source - p["mean"]) * np.sqrt(p["normalized_weight"]) / p["scale"])
    with np.load(run / "fusion.npz", allow_pickle=False) as p:
        return (np.concatenate(blocks, axis=1) - p["mean"]) @ p["components"].T
