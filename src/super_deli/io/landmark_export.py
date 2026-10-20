"""Shared, non-overwriting coordinate exports with an ingestible dataset declaration."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from super_deli.core.data import DatasetConfig, Modality, MorphometricReaderOptions
from super_deli.core.morphometrics import LandmarkCollection
from super_deli.io.config import write_config, write_json


def write_landmark_coordinates(
    collection: LandmarkCollection,
    *,
    dataset: DatasetConfig,
    source: Modality,
    directory: Path,
    stem: str = "landmarks",
    config_name: str = "landmark_dataset.json",
) -> None:
    write_json(collection.to_dict(), directory / f"{stem}.json")
    scale = collection.scale_unit is not None or any(
        r.scale_factor is not None for r in collection.records
    )
    with (directory / f"{stem}.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "observation_id",
                "landmark_id",
                *("x", "y", "z")[: collection.dimensions],
                "unit_id",
                "image",
                *(("scale_factor",) if scale else ()),
            )
        )
        for record in collection.records:
            for name, point in zip(collection.landmark_ids, record.coordinates, strict=True):
                writer.writerow(
                    (
                        record.observation_id,
                        name,
                        *point,
                        record.unit_id,
                        record.image,
                        *((record.scale_factor,) if scale else ()),
                    )
                )
    retained = {r.observation_id for r in collection.records}
    modality = Modality(
        name=source.name,
        kind="morphometric",
        level=source.level,
        format="landmarks_csv",
        path=f"{stem}.csv",
        observations=tuple(o for o in source.observations if o.observation_id in retained),
        morphometric_options=MorphometricReaderOptions(
            dimensions=collection.dimensions,
            landmark_ids=collection.landmark_ids,
            coordinate_unit=collection.coordinate_unit,
            scale_unit=collection.scale_unit,
        ),
    )
    write_config(replace(dataset, modalities=(modality,)), directory / config_name)
