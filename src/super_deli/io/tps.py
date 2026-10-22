"""Read the fixed-landmark LM/LM3 subset of TPS without modifying coordinates."""

from __future__ import annotations

import re
from pathlib import Path

from super_deli.core.data import Modality
from super_deli.core.morphometrics import LandmarkRecord
from super_deli.io.delimited import identifier
from super_deli.io.morphometric_base import (
    MorphometricIngestionError,
    coordinates,
    options_for,
    scale_factor,
    unit_for,
)

_DIRECTIVE = re.compile(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*=(.*)\Z")


def read_tps(path: Path, modality: Modality) -> tuple[LandmarkRecord, ...]:
    options = options_for(modality)
    links = {o.observation_id: o.unit_id for o in modality.observations}
    target = len(options.landmark_ids)
    records = []
    points = None
    fields, comments = {}, []
    line_number = 0

    def finish(location: str) -> LandmarkRecord:
        if len(points) != target:
            raise MorphometricIngestionError(
                f"{location}: truncated landmark block; expected {target} points, "
                f"found {len(points)}"
            )
        if options.tps_id_field not in fields:
            raise MorphometricIngestionError(
                f"{location}: missing {options.tps_id_field}= observation identifier"
            )
        observation = fields[options.tps_id_field]
        return LandmarkRecord(
            observation,
            unit_for(observation, links, location),
            tuple(points),
            options.landmark_ids,
            scale_factor(fields["SCALE"].strip(), location) if "SCALE" in fields else None,
            fields.get("IMAGE"),
            fields.get("ID"),
            tuple(comments),
        )

    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line_number, raw in enumerate(handle, 1):
                line = raw.rstrip("\r\n")
                location = f"{modality.name} line {line_number}"
                if not line.strip():
                    continue
                if "\x00" in line:
                    raise MorphometricIngestionError(f"{location}: NUL in TPS source")
                directive = _DIRECTIVE.fullmatch(line)
                if directive:
                    key, text = directive[1].upper(), directive[2]
                    if key in {"LM", "LM3"}:
                        if points is not None:
                            records.append(finish(location))
                        dimensions = 2 if key == "LM" else 3
                        if dimensions != options.dimensions:
                            raise MorphometricIngestionError(
                                f"{location}: {key} dimensionality conflicts with declaration"
                            )
                        if not re.fullmatch(r"[0-9]+", text.strip()) or int(text.strip()) != target:
                            raise MorphometricIngestionError(
                                f"{location}: landmark count must match {target} declared IDs"
                            )
                        points, fields, comments = [], {}, []
                    elif points is None:
                        raise MorphometricIngestionError(
                            f"{location}: TPS records must start with LM= or LM3="
                        )
                    elif len(points) != target:
                        raise MorphometricIngestionError(
                            f"{location}: metadata before all declared coordinates"
                        )
                    elif key == "COMMENT":
                        comments.append(text)
                    elif key in {"ID", "IMAGE", "SCALE"}:
                        if key in fields:
                            raise MorphometricIngestionError(f"{location}: duplicate {key}= field")
                        if key == "SCALE":
                            scale_factor(text.strip(), location)
                        else:
                            identifier(text, location)
                        fields[key] = text
                    else:
                        raise MorphometricIngestionError(
                            f"{location}: unsupported TPS directive {key}; "
                            "only fixed landmarks are supported"
                        )
                elif points is None or len(points) >= target:
                    raise MorphometricIngestionError(
                        f"{location}: unexpected coordinate or unsupported TPS syntax"
                    )
                else:
                    points.append(coordinates(line.split(), options, location))
        if points is None:
            raise MorphometricIngestionError(f"{modality.name}: no LM=/LM3= landmark records")
        records.append(finish(f"{modality.name} end of file"))
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, MorphometricIngestionError):
            raise
        raise MorphometricIngestionError(
            f"{modality.name} line {max(1, line_number)}: {exc}"
        ) from exc
    return tuple(records)
