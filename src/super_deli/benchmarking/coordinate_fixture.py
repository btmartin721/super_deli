"""Invented untrained numerical archive for engineering; never research evidence."""

from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.coordinate_config import SPACES, CoordinateStudy, tasks
from super_deli.benchmarking.matched_config import ECO, MORPH
from super_deli.benchmarking.matched_data import generate, write_dataset
from super_deli.benchmarking.moons_hierarchy import retained_hierarchy
from super_deli.benchmarking.neural_analysis import neural_forward
from super_deli.benchmarking.separate_config import model_for
from super_deli.io.config import write_json
from super_deli.models.matched_preprocessing import prepare_views
from super_deli.models.separate_neural import prepare_branch


def weights(d, latent, seed):
    rng = np.random.default_rng([810013, seed, d, latent])
    result = {}
    for side, widths in (("encoder", (d, 16, 8, latent)), ("decoder", (latent, 8, 16, d))):
        for i, (left, right) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            result[f"{side}_{i}_weight"] = rng.normal(0, 0.4 / np.sqrt(left), (right, left))
            result[f"{side}_{i}_bias"] = rng.normal(0, 0.01, right)
    return result


def normalized(raw):
    return dict(
        scores=(raw - raw.mean(0)) / raw.std(0), latent_means=raw.mean(0), latent_scales=raw.std(0)
    )


def create_fixture(source):
    source = Path(source)
    source.mkdir(exist_ok=False)
    task = tasks(CoordinateStudy(fixture=True), "main")[0]
    data = generate(
        {
            **task,
            "data_seed": [810019],
            "noise_seed": [810029],
            "permutation_seed": [810041],
            "ecology_order_seed": [810049],
        }
    )
    base = source / "main" / task["key"]
    base.mkdir(parents=True)
    write_json(task, base / "task.json")
    np.savez_compressed(base / "generating-reference.npz", **data)
    write_dataset(data, base / "raw")
    x = np.column_stack((data["morphology"], data["ecology"]))
    cohort = dict(
        unit_ids=[f"unit_{i:04d}" for i in range(task["n"])],
        views=[
            dict(
                modality=role,
                kind=kind,
                feature_names=list(features),
                observation_ids=[f"{role}_{j:04d}" for j in range(task["n"])],
                source_row_indices=(
                    np.arange(task["n"]) if i == 0 else np.argsort(data["ecology_order"])
                ).tolist(),
            )
            for i, (role, kind, features) in enumerate(
                (("morphology", "morphological", MORPH), ("ecology", "ecological", ECO))
            )
        ],
    )
    rows = []
    with threadpool_limits(limits=1):
        for space in SPACES:
            out = base / space
            out.mkdir()
            seed = 0 if space == "pca4" else int(space.rsplit("_", 1)[1])
            if space.startswith("joint"):
                z, a = prepare_views(x, 4)
                a.update(weights(12, 2 if space.startswith("joint2_") else 4, seed))
                a.update(normalized(neural_forward(z, a, "encoder")))
            else:
                a = {}
                scores = []
                for role in ("morphology", "ecology"):
                    z, b = prepare_branch(data[role])
                    if space == "pca4":
                        mean = z.mean(0)
                        _, _, vt = np.linalg.svd(z - mean, full_matrices=False)
                        b.update(pca_mean=mean, components=vt[:2])
                        encoded = (z - mean) @ vt[:2].T
                    else:
                        b.update(weights(z.shape[1], 2, seed))
                        encoded = neural_forward(z, b, "encoder")
                    b.update(normalized(encoded))
                    scores.append(b["scores"])
                    a.update({role + "__" + key: value for key, value in b.items()})
                a["scores"] = np.column_stack(scores)
            estimator = HDBSCAN(
                min_cluster_size=5, min_samples=5, allow_single_cluster=True, n_jobs=1, copy=True
            ).fit(a["scores"])
            a.update(labels=estimator.labels_, strength=estimator.probabilities_)
            np.savez_compressed(out / "point.npz", **a)
            linkage, condensed = retained_hierarchy(estimator, task["n"], 5)
            index = 1 if space.startswith("separate_") else 0
            np.savez_compressed(
                out / "density-trace.npz",
                **{f"fit_{index}_linkage": linkage, f"fit_{index}_condensed": condensed},
            )
            write_json(cohort, out / "cohort.json")
            write_json(model_for(space, 0).to_dict(), out / "model-config.json")
            rows.append(dict(arm=space, status="available"))
    write_json(rows, base / "fits.json")
    write_json(
        dict(
            kind="invented_numeric_coordinate_fixture",
            neural_training=0,
            interpretation=(
                "Untrained numeric weights; engineering only, no scientific recovery claims"
            ),
        ),
        source / "fixture.json",
    )
    return source
