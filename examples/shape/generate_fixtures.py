"""Deterministic rectangular shapes with analytic GPA/tangent-PCA truth."""

import csv
import math
from pathlib import Path

from super_deli.io.config import write_json


def generate(destination: Path) -> None:
    destination.mkdir(exist_ok=False)
    heights = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    angles = [0, 0.7, -1.2, 2.4, -0.3, 1.4]
    scales = [1, 2, 0.5, 3, 1.5, 2.5]
    means = [(i * 3, -i * 2) for i in range(6)]
    landmarks = ["lower_left", "lower_right", "upper_right", "upper_left"]
    for filename, hs, aa, ss, cc, offset in (
        ("train", heights, angles, scales, means, 0),
        ("projection", [0.5, 0.9], [-0.4, 0.8], [3, 2], [(7, -2), (-8, 3)], 6),
    ):
        observations = []
        with (destination / f"{filename}.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("observation_id", "landmark_id", "x", "y"))
            for i, (h, angle, scale, center) in enumerate(zip(hs, aa, ss, cc, strict=True), offset):
                name = f"shape_{i + 1}"
                observations.append({"observation_id": name, "unit_id": name})
                for landmark, (x, y) in zip(
                    landmarks, [(-1, -h), (1, -h), (1, h), (-1, h)], strict=True
                ):
                    writer.writerow(
                        (
                            name,
                            landmark,
                            scale * (x * math.cos(angle) - y * math.sin(angle)) + center[0],
                            scale * (x * math.sin(angle) + y * math.cos(angle)) + center[1],
                        )
                    )
        write_json(
            {
                "schema_version": 1,
                "dataset_id": f"synthetic_shape_{filename}",
                "specimens": [{"specimen_id": o["unit_id"]} for o in observations],
                "modalities": [
                    {
                        "name": "outline",
                        "kind": "morphometric",
                        "format": "landmarks_csv",
                        "level": "specimen",
                        "path": f"{filename}.csv",
                        "observations": observations,
                        "morphometric_options": {
                            "dimensions": 2,
                            "landmark_ids": landmarks,
                            "coordinate_unit": "arbitrary",
                        },
                    }
                ],
            },
            destination / f"{filename}_dataset.json",
        )
    alpha = [math.atan(h) for h in heights]
    mean_angle = math.atan2(sum(map(math.sin, alpha)), sum(map(math.cos, alpha)))
    scores = [math.sin(a - mean_angle) for a in alpha]
    write_json({"schema_version": 1}, destination / "model.json")
    write_json(
        {
            "description": "Analytic fixed rectangles; no species or empirical landmark truth",
            "heights": heights,
            "centroids": means,
            "centroid_sizes": [
                2 * s * math.sqrt(1 + h * h) for h, s in zip(heights, scales, strict=True)
            ],
            "consensus": [
                [x * math.cos(mean_angle) / 2, y * math.sin(mean_angle) / 2]
                for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            ],
            "pca_rank": 1,
            "scores_up_to_sign": scores,
            "explained_variance": sum(s * s for s in scores) / (len(scores) - 1),
        },
        destination / "expected.json",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New directory")
    generate(parser.parse_args().destination)
