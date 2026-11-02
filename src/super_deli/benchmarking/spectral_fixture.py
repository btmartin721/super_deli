"""Invented untrained coordinates for installed tests, never empirical evidence."""

from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.density_study import generate_raw
from super_deli.benchmarking.spectral_config import SPACES
from super_deli.io.config import write_json
from super_deli.models.density_scores import fit_density_scores
from super_deli.models.hdbscan_config import HDBSCANConfig


def create_fixture(root):
    root = Path(root)
    root.mkdir()
    entries = []
    with threadpool_limits(limits=1):
        for c, case in enumerate(("gaussian", "four_spherical", "moons")):
            for n in (32, 48):
                z, truth = generate_raw(case, n, [851, c, n])
                clean = np.column_stack(
                    [
                        z,
                        z[:, 0] + z[:, 1],
                        z[:, 0] - z[:, 1],
                        z[:, 0] ** 2,
                        z[:, 1] ** 2,
                        np.sin(z[:, 0]),
                        np.cos(z[:, 1]),
                    ]
                )
                noisy = clean + np.random.default_rng([852, c, n]).normal(0, 0.15, clean.shape)
                data = {"truth": truth, "source": (z - z.mean(0)) / z.std(0)}
                for condition, x in (("clean", clean), ("noisy", noisy)):
                    x = (x - x.mean(0)) / x.std(0)
                    data[condition + "_traits"] = x
                    for j, arm in enumerate(
                        (
                            "density",
                            "neural_0",
                            "neural_101",
                            "neural_202",
                            "denoising_0",
                            "denoising_101",
                            "denoising_202",
                        )
                    ):
                        # Analytic untrained projection fixture, not a fitted neural network.
                        matrix = np.random.default_rng([853, j]).normal(size=(8, 2))
                        scores = x @ matrix
                        data[condition + "_" + arm] = (scores - scores.mean(0)) / scores.std(0)
                for space in SPACES:
                    if not space.endswith("_traits"):
                        labels, strength = fit_density_scores(data[space], HDBSCANConfig(True))
                        data[space + "_labels"], data[space + "_strength"] = labels, strength
                key = f"{case}_n{n}_0000"
                np.savez_compressed(root / (key + ".npz"), **data)
                entries.append(
                    {
                        "key": key,
                        "case": case,
                        "n": n,
                        "replicate": 0,
                        "file": key + ".npz",
                        "unit_ids": [f"invented_{i:04d}" for i in range(n)],
                    }
                )
    write_json(
        {
            "kind": "invented_untrained_coordinate_fixture",
            "sizes": [32, 48],
            "observations": entries,
        },
        root / "fixture.json",
    )
    return root
