"""Two complementary binary signals define four generating groups, independently of IDs."""

import csv
import json
import math
import random
from pathlib import Path


def generate(directory):
    directory.mkdir(exist_ok=True)
    rng = random.Random(20261020)
    morphology, ecology, landmarks, genomic, truth = [], [], [], [], []
    grouped_specimens, matched_specimens = [], []
    grouped_links = {key: [] for key in ("genomic", "morphological", "ecological", "morphometric")}
    matched_links = {key: [] for key in grouped_links}
    for j in range(64):
        group, site = j // 16, f"site_{j:03d}"
        x, y = group // 2, group % 2
        ecology.append(
            (
                f"e{j:03d}",
                12 + x * 8 + rng.gauss(0, 0.4),
                1200 - x * 400 + rng.gauss(0, 40),
                300 + x * 700 + rng.gauss(0, 60),
            )
        )
        grouped_links["ecological"].append({"observation_id": f"e{j:03d}", "unit_id": site})
        matched_links["ecological"].append(
            {"observation_id": f"e{j:03d}", "unit_id": f"u{2 * j:03d}"}
        )
        truth.append(
            {"locality_id": site, "matched_unit_id": f"u{2 * j:03d}", "latent_group": group}
        )
        for replicate in range(2):
            i = j * 2 + replicate
            unit = f"u{i:03d}"
            matched_specimens.append({"specimen_id": unit, "locality_id": site})
            for key, prefix in (("genomic", "g"), ("morphological", "m"), ("morphometric", "s")):
                observation = f"{prefix}{i:03d}"
                source_unit = f"{prefix}_individual_{i:03d}"
                grouped_specimens.append({"specimen_id": source_unit, "locality_id": site})
                grouped_links[key].append({"observation_id": observation, "unit_id": source_unit})
                matched_links[key].append({"observation_id": observation, "unit_id": unit})
            sequence = "".join(
                "ARG"[sum(rng.random() < (0.12 if x == 0 else 0.88) for _ in range(2))]
                for _ in range(96)
            )
            genomic.append(f"g{i:03d} {sequence}")
            morphology.append(
                (
                    f"m{i:03d}",
                    20 + y * 6 + rng.gauss(0, 0.4),
                    5 + y * 2 + rng.gauss(0, 0.3),
                    25 + y * 15 + rng.gauss(0, 1.0),
                    rng.choice(["female", "male"]),
                )
            )
            a, b = (-0.18 if y == 0 else 0.18) + rng.gauss(0, 0.012), rng.gauss(0, 0.018)
            angle, scale = rng.uniform(-math.pi, math.pi), rng.uniform(0.5, 3)
            tx, ty = rng.uniform(-10, 10), rng.uniform(-10, 10)
            for k, (px, py) in enumerate(
                [(-1, -0.7), (0, -1), (1, -0.7), (1, 0.7), (0, 1), (-1, 0.7)]
            ):
                py += a if k == 4 else 0
                px += b if k in (2, 3) else 0
                px, py = px + rng.gauss(0, 0.001), py + rng.gauss(0, 0.001)
                landmarks.append(
                    (
                        f"s{i:03d}",
                        f"L{k + 1}",
                        tx + scale * (px * math.cos(angle) - py * math.sin(angle)),
                        ty + scale * (px * math.sin(angle) + py * math.cos(angle)),
                    )
                )
    (directory / "genotypes.phy").write_text("128 96\n" + "\n".join(genomic) + "\n")
    for filename, header, rows in (
        ("traits.csv", ("observation_id", "length_mm", "width_mm", "mass_g", "sex"), morphology),
        (
            "ecology.csv",
            ("observation_id", "temperature_degC", "rainfall_mm", "altitude_m"),
            ecology,
        ),
        ("landmarks.csv", ("observation_id", "landmark_id", "x", "y"), landmarks),
    ):
        with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    for mode, links, specimens in (
        ("grouped", grouped_links, grouped_specimens),
        ("matched", matched_links, matched_specimens),
    ):
        modalities = [
            {
                "name": "markers",
                "kind": "genomic",
                "level": "specimen",
                "format": "phylip",
                "path": "genotypes.phy",
                "observations": links["genomic"],
            },
            {
                "name": "traits",
                "kind": "morphological",
                "level": "specimen",
                "format": "csv",
                "path": "traits.csv",
                "observations": links["morphological"],
                "features": [
                    {"name": "length_mm", "dtype": "continuous", "unit": "mm"},
                    {"name": "width_mm", "dtype": "continuous", "unit": "mm"},
                    {"name": "mass_g", "dtype": "continuous", "unit": "g"},
                    {
                        "name": "sex",
                        "dtype": "categorical",
                        "categories": ["female", "male"],
                        "role": "covariate",
                    },
                ],
            },
            {
                "name": "environment",
                "kind": "ecological",
                "level": "locality" if mode == "grouped" else "specimen",
                "format": "csv",
                "path": "ecology.csv",
                "observations": links["ecological"],
                "features": [
                    {"name": "temperature_degC", "dtype": "continuous", "unit": "degC"},
                    {"name": "rainfall_mm", "dtype": "continuous", "unit": "mm"},
                    {"name": "altitude_m", "dtype": "continuous", "unit": "m"},
                ],
            },
            {
                "name": "outline",
                "kind": "morphometric",
                "level": "specimen",
                "format": "landmarks_csv",
                "path": "landmarks.csv",
                "observations": links["morphometric"],
                "morphometric_options": {
                    "dimensions": 2,
                    "landmark_ids": [f"L{k + 1}" for k in range(6)],
                    "coordinate_unit": "arbitrary",
                },
            },
        ]
        dataset = {
            "schema_version": 1,
            "dataset_id": f"synthetic_integrative_{mode}",
            "specimens": specimens,
            "localities": [f"site_{j:03d}" for j in range(64)],
            "modalities": modalities,
        }
        views = [
            {
                "modality": m["name"],
                "kind": m["kind"],
                "n_pcs": 2,
                "weight": 1.0,
                "aggregation": "mean" if mode == "grouped" and m["level"] == "specimen" else "none",
            }
            for m in modalities
        ]
        views[0].update(ploidy=2, missing_policy="error", scaling="patterson")
        views[-1]["shape"] = {"schema_version": 1}
        model = {
            "schema_version": 1,
            "assume_independent_units": True,
            "target_level": "locality" if mode == "grouped" else "specimen",
            "missing_views": "complete_cases" if mode == "matched" else "error",
            "n_pcs": 2,
            "max_k": 4,
            "min_cluster_size": 5,
            "n_bootstrap": 10,
            "n_init": 5,
            "random_seed": 71,
            "views": views,
        }
        for name, value in ((f"{mode}_dataset.json", dataset), (f"{mode}_model.json", model)):
            (directory / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (directory / "truth.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 20261020,
                "scope": "Four complementary generating groups; no empirical species truth",
                "units": truth,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
