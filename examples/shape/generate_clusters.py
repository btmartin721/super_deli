"""Seeded latent shape groups; nuisance pose/size and truth stay outside features."""

import csv
import json
import math
import random
from pathlib import Path


def generate(directory: Path):
    rng = random.Random(20261020)
    directory.mkdir(exist_ok=True)
    base = [(-1, -0.7), (0, -1), (1, -0.7), (1, 0.7), (0, 1), (-1, 0.7)]
    rows, observations, truth = [], [], []
    for i in range(64):
        unit, label = f"unit_{i + 1:03d}", i // 32
        a = (-0.16 if label == 0 else 0.16) + rng.gauss(0, 0.012)
        b = rng.gauss(0, 0.02)
        angle, scale = rng.uniform(-math.pi, math.pi), rng.uniform(0.5, 3)
        tx, ty = rng.uniform(-20, 20), rng.uniform(-20, 20)
        for k, (x, y) in enumerate(base):
            y += a if k == 4 else 0
            x += b if k in (2, 3) else 0
            x, y = x + rng.gauss(0, 0.001), y + rng.gauss(0, 0.001)
            rows.append(
                (
                    unit,
                    f"L{k + 1}",
                    tx + scale * (x * math.cos(angle) - y * math.sin(angle)),
                    ty + scale * (x * math.sin(angle) + y * math.cos(angle)),
                )
            )
        observations.append({"observation_id": unit, "unit_id": unit})
        truth.append({"unit_id": unit, "latent_group": label, "a": a, "b": b})
    with (directory / "coordinates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("observation_id", "landmark_id", "x", "y"))
        writer.writerows(rows)
    dataset = {
        "schema_version": 1,
        "dataset_id": "synthetic_shape_groups",
        "specimens": [{"specimen_id": o["unit_id"]} for o in observations],
        "modalities": [
            {
                "name": "outline",
                "kind": "morphometric",
                "format": "landmarks_csv",
                "level": "specimen",
                "path": "coordinates.csv",
                "observations": observations,
                "morphometric_options": {
                    "dimensions": 2,
                    "coordinate_unit": "arbitrary",
                    "landmark_ids": [f"L{k + 1}" for k in range(6)],
                },
            }
        ],
    }
    for name, value in (
        ("dataset.json", dataset),
        (
            "truth.json",
            {
                "schema_version": 1,
                "seed": 20261020,
                "scope": "Two generating shape groups; not empirical species truth or calibration",
                "units": truth,
            },
        ),
        (
            "model.json",
            {
                "schema_version": 1,
                "assume_independent_units": True,
                "n_pcs": 2,
                "max_k": 3,
                "min_cluster_size": 5,
                "n_bootstrap": 20,
                "n_init": 5,
                "random_seed": 71,
            },
        ),
    ):
        (directory / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "clustering")
