"""Long-form CSV coordinates with explicit landmark names and observation links."""

from __future__ import annotations

from pathlib import Path

from super_deli.core.data import Modality
from super_deli.core.morphometrics import LandmarkRecord
from super_deli.io.delimited import delimited_rows, identifier
from super_deli.io.morphometric_base import (
    MorphometricIngestionError,
    coordinates,
    options_for,
    scale_factor,
    unit_for,
)


def read_landmark_csv(path: Path, modality: Modality) -> tuple[LandmarkRecord, ...]:
    options = options_for(modality)
    axes = ("x", "y", "z")[: options.dimensions]
    links = {o.observation_id: o.unit_id for o in modality.observations}
    expected = set(options.landmark_ids)
    groups, metadata = {}, {}
    try:
        for line, row in delimited_rows(
            path,
            delimiter=",",
            context=modality.name,
            required={"observation_id", "landmark_id", *axes},
            optional=frozenset({"unit_id", "image", "scale_factor"}),
        ):
            location = f"{modality.name} line {line}"
            observation = row["observation_id"]
            unit = unit_for(observation, links, location)
            if "unit_id" in row and identifier(row["unit_id"], location) != unit:
                raise MorphometricIngestionError(
                    f"{location}: unit ID conflicts with observation declaration"
                )
            landmark = identifier(row["landmark_id"], location)
            if landmark not in expected:
                raise MorphometricIngestionError(f"{location}: undeclared landmark ID {landmark!r}")
            points = groups.setdefault(observation, {})
            if landmark in points:
                raise MorphometricIngestionError(f"{location}: duplicate observation/landmark pair")
            points[landmark] = coordinates([row[axis] for axis in axes], options, location)
            scale = (
                scale_factor(row["scale_factor"], location) if row.get("scale_factor", "") else None
            )
            image = identifier(row["image"], location) if row.get("image", "") else None
            current = (scale, image)
            if observation in metadata and metadata[observation] != current:
                raise MorphometricIngestionError(
                    f"{location}: inconsistent scale/image metadata across landmark rows"
                )
            metadata[observation] = current
        records = []
        for observation, points in groups.items():
            if missing := expected - points.keys():
                raise MorphometricIngestionError(
                    f"{modality.name} {observation!r}: missing landmark rows {sorted(missing)}"
                )
            scale, image = metadata[observation]
            records.append(
                LandmarkRecord(
                    observation,
                    links[observation],
                    tuple(points[name] for name in options.landmark_ids),
                    tuple(points),
                    scale,
                    image,
                )
            )
    except ValueError as exc:
        if isinstance(exc, MorphometricIngestionError):
            raise
        raise MorphometricIngestionError(str(exc)) from exc
    return tuple(records)
