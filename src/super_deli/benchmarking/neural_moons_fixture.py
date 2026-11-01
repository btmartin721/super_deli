"""Invented engineering fixtures with fixed, untrained neural weights."""

from pathlib import Path

from super_deli.io.config import write_json


def create_fixture(directory):
    import numpy as np
    from sklearn.cluster import HDBSCAN

    from super_deli.benchmarking.neural_moons_config import DENSITY_ARMS
    from super_deli.models.tabular_neural import forward_numpy

    root = Path(directory).resolve()
    root.mkdir(exist_ok=False)
    observations, replays = [], []
    for case, n in (("gaussian", 24), ("moons", 32)):
        rng = np.random.default_rng(4301 + n)
        if case == "moons":
            angle = np.linspace(0.03, np.pi - 0.03, n // 2)
            z = np.concatenate(
                [np.c_[np.cos(angle), np.sin(angle)], np.c_[1 - np.cos(angle), 0.5 - np.sin(angle)]]
            )
            z += rng.normal(0, 0.03, z.shape)
            truth = np.repeat([0, 1], n // 2)
        else:
            z, truth = rng.normal(size=(n, 2)), np.zeros(n, dtype=int)
        a, b = z.T
        x = np.c_[a, b, a * b, a * a, b * b, np.sin(a), np.cos(b), np.sin(a + b)]
        x += rng.normal(0, 0.03, x.shape)
        key = f"invented_{case}_n{n}"
        folder = root / key
        folder.mkdir()
        with (folder / "reference.npz").open("xb") as f:
            np.savez_compressed(f, x=x, z=z, truth=truth)
        entries = []
        for phase, ids in (
            ("point", np.arange(n)),
            ("draw", np.sort(rng.choice(n, int(0.8 * n), replace=False))),
        ):
            selected = x[ids]
            means, scales = selected.mean(0), selected.std(0)
            std = (selected - means) / scales
            points = {}
            for arm in DENSITY_ARMS:
                target = folder / phase / arm
                target.mkdir(parents=True)
                arrays = {"means": means, "scales": scales, "kept_indices": np.arange(8)}
                if arm == "density":
                    pca_mean = std.mean(0)
                    _, singular, vt = np.linalg.svd(std - pca_mean, full_matrices=False)
                    scores = (std - pca_mean) @ vt[:2].T
                    arrays.update(
                        pca_mean=pca_mean,
                        components=vt[:2],
                        explained_variance=singular[:2] ** 2 / (len(ids) - 1),
                    )
                else:
                    # Fixed random coefficients are not a trained autoencoder or benchmark outcome.
                    weights_rng = np.random.default_rng(4409 + int(arm.split("_")[-1]))
                    for side, widths in (("encoder", [8, 16, 8, 2]), ("decoder", [2, 8, 16, 8])):
                        for i, (left, right) in enumerate(
                            zip(widths[:-1], widths[1:], strict=True)
                        ):
                            arrays[f"{side}_{i}_weight"] = weights_rng.normal(0, 0.2, (right, left))
                            arrays[f"{side}_{i}_bias"] = weights_rng.normal(0, 0.02, right)
                    raw = forward_numpy(std, arrays, "encoder")
                    arrays.update(latent_means=raw.mean(0), latent_scales=raw.std(0))
                    scores = (raw - raw.mean(0)) / raw.std(0)
                    arrays["losses"] = np.array(
                        [np.mean((forward_numpy(raw, arrays, "decoder") - std) ** 2)]
                    )
                fitted = HDBSCAN(
                    min_cluster_size=5,
                    min_samples=5,
                    allow_single_cluster=True,
                    n_jobs=1,
                    copy=True,
                ).fit(scores)
                arrays.update(scores=scores, labels=fitted.labels_, strength=fitted.probabilities_)
                with (target / "point.npz").open("xb") as f:
                    np.savez_compressed(f, **arrays)
                points[arm] = str((target / "point.npz").relative_to(root))
            entries.append(
                {
                    "key": key,
                    "case": case,
                    "n": n,
                    "replicate": 0,
                    "indices": ids.tolist(),
                    "unit_ids": [f"invented_{i:04d}" for i in ids],
                    "reference": key + "/reference.npz",
                    "points": points,
                }
            )
        observations.append(entries[0])
        for arm in DENSITY_ARMS:
            replays.append({**entries[1], "key": key + "_draw_" + arm, "replay_arm": arm})
    write_json(
        {
            "kind": "invented_untrained_fixture",
            "observations": observations,
            "replays": replays,
            "interpretation": (
                "Engineering fixture only; no scientific training or performance inference"
            ),
        },
        root / "fixture.json",
    )
    return root
