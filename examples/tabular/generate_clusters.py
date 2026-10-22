"""Generate separate numeric morphology and locality-ecology truth fixtures."""

import csv
import json
import random
from pathlib import Path


def generate(directory):
    directory.mkdir(exist_ok=True)
    rng = random.Random(20261020)
    morphology, ecology, observations, sites, truth = [], [], [], [], []
    for i in range(64):
        specimen, site, group = f"unit_{i + 1:03d}", f"site_{i + 1:03d}", i // 32
        morphology.append(
            (
                specimen,
                20 + 5 * group + rng.gauss(0, 0.35),
                5 + 1.5 * group + rng.gauss(0, 0.25),
                25 + 12 * group + rng.gauss(0, 1),
                10 + 8 * group + rng.randrange(4),
                "striped" if group else "plain",
                rng.choice(["female", "male"]),
            )
        )
        ecology.append(
            (
                site,
                15 + 8 * group + rng.gauss(0, 0.5),
                1200 - 300 * group + rng.gauss(0, 60),
                300 + 600 * group + rng.gauss(0, 70),
                3 + 15 * group + rng.randrange(3),
            )
        )
        observations.append({"observation_id": specimen, "unit_id": specimen})
        sites.append({"observation_id": site, "unit_id": site})
        truth.append({"specimen_id": specimen, "locality_id": site, "latent_group": group})
    for filename, delimiter, columns, rows in (
        (
            "morphology.csv",
            ",",
            (
                "observation_id",
                "length_mm",
                "head_width_mm",
                "mass_g",
                "scale_count",
                "pigmentation",
                "sex",
            ),
            morphology,
        ),
        (
            "ecology.tsv",
            "\t",
            ("observation_id", "temperature_degC", "rainfall_mm", "altitude_m", "resource_count"),
            ecology,
        ),
    ):
        with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)
    morphology_features = [
        {"name": "length_mm", "dtype": "continuous", "unit": "mm"},
        {"name": "head_width_mm", "dtype": "continuous", "unit": "mm"},
        {"name": "mass_g", "dtype": "continuous", "unit": "g"},
        {"name": "scale_count", "dtype": "count"},
        {"name": "pigmentation", "dtype": "categorical", "categories": ["plain", "striped"]},
        {
            "name": "sex",
            "dtype": "categorical",
            "categories": ["female", "male"],
            "role": "covariate",
        },
    ]
    ecology_features = [
        {"name": "temperature_degC", "dtype": "continuous", "unit": "degC"},
        {"name": "rainfall_mm", "dtype": "continuous", "unit": "mm"},
        {"name": "altitude_m", "dtype": "continuous", "unit": "m"},
        {"name": "resource_count", "dtype": "count"},
    ]
    dataset = {
        "schema_version": 1,
        "dataset_id": "synthetic_tabular_groups",
        "specimens": [
            {"specimen_id": o["unit_id"], "locality_id": s["unit_id"]}
            for o, s in zip(observations, sites, strict=True)
        ],
        "localities": [s["unit_id"] for s in sites],
        "modalities": [
            {
                "name": "traits",
                "kind": "morphological",
                "level": "specimen",
                "format": "csv",
                "path": "morphology.csv",
                "observations": observations,
                "features": morphology_features,
            },
            {
                "name": "environment",
                "kind": "ecological",
                "level": "locality",
                "format": "tsv",
                "path": "ecology.tsv",
                "observations": sites,
                "features": ecology_features,
            },
        ],
    }
    common = {
        "schema_version": 1,
        "assume_independent_units": True,
        "n_pcs": 2,
        "max_k": 3,
        "n_init": 5,
        "n_bootstrap": 20,
        "random_seed": 71,
        "missing_policy": "error",
        "scaling": "standard",
        "count_transform": "log1p",
    }
    for name, value in (
        ("dataset.json", dataset),
        (
            "morphology_model.json",
            {**common, "feature_names": [f["name"] for f in morphology_features[:4]]},
        ),
        ("ecology_model.json", common),
        (
            "truth.json",
            {
                "schema_version": 1,
                "seed": 20261020,
                "scope": "Two generating groups in each table; not empirical species truth",
                "units": truth,
            },
        ),
    ):
        (directory / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "clustering")
