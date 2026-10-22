"""Parse image references and selectors without opening images or TPS files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from super_deli.core.data import ImageReaderOptions, Modality
from super_deli.io.delimited import delimited_rows, identifier

# Preserve the established standalone utility's columns; its implementation
# stays separate because an installed wheel does not include that root script.
LEGACY_COLUMNS = frozenset(
    {
        "image_path",
        "specimen_id",
        "group",
        "split",
        "tps_path",
        "landmark_count",
        "scale_value",
        "scale_unit",
        "check_status",
        "check_details",
    }
)
SUFFIXES = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".heic", ".heif"})


class ImageIngestionError(ValueError):
    """Image input or selected content cannot be decoded under the declared policy."""


@dataclass(frozen=True)
class ImageReference:
    observation_id: str
    unit_id: str
    path: Path
    page: int | None
    frame: int | None
    metadata: tuple[tuple[str, str], ...]


def _selector(value: str, name: str, location: str) -> int | None:
    if not value:
        return None
    if not re.fullmatch(r"[0-9]+", value):
        raise ImageIngestionError(f"{location}: {name} must be an integer index")
    index = int(value)
    if name == "page" and index < 1:
        raise ImageIngestionError(f"{location}: PDF page numbers start at 1")
    return index


def read_image_manifest(
    path: Path, modality: Modality, *, source_dir: Path
) -> tuple[ImageReference, ...]:
    options = modality.image_options or ImageReaderOptions()
    links = {o.observation_id: o.unit_id for o in modality.observations}
    seen, references = set(), []
    try:
        for line, row in delimited_rows(
            path,
            delimiter=",",
            context=modality.name,
            required={options.observation_id_column, "image_path"},
            optional=LEGACY_COLUMNS | {"page", "frame", "unit_id"},
        ):
            location = f"{modality.name} line {line}"
            observation = identifier(row[options.observation_id_column], location)
            if observation in seen:
                raise ImageIngestionError(f"{location}: duplicate observation ID {observation!r}")
            if observation not in links:
                raise ImageIngestionError(f"{location}: undeclared observation ID {observation!r}")
            unit = links[observation]
            if "unit_id" in row and identifier(row["unit_id"], location) != unit:
                raise ImageIngestionError(f"{location}: unit ID conflicts with declaration")
            if "specimen_id" in row:
                if modality.level != "specimen" or identifier(row["specimen_id"], location) != unit:
                    raise ImageIngestionError(
                        f"{location}: specimen ID conflicts with biological link"
                    )
            if row.get("check_status", "") not in {"", "ok"}:
                raise ImageIngestionError(
                    f"{location}: manifest check_status is not ok: {row.get('check_details', '')}"
                )
            image = Path(identifier(row["image_path"], location))
            image = (source_dir / image).resolve() if not image.is_absolute() else image.resolve()
            if image.suffix.lower() not in SUFFIXES:
                raise ImageIngestionError(
                    f"{location}: unsupported image extension {image.suffix!r}"
                )
            page = _selector(row.get("page", ""), "page", location)
            frame = _selector(row.get("frame", ""), "frame", location)
            if (image.suffix.lower() == ".pdf" and frame is not None) or (
                image.suffix.lower() != ".pdf" and page is not None
            ):
                raise ImageIngestionError(
                    f"{location}: page selects PDF content; frame selects raster content"
                )
            references.append(
                ImageReference(
                    observation,
                    unit,
                    image,
                    page,
                    frame,
                    tuple(
                        (key, value)
                        for key, value in row.items()
                        if key not in {options.observation_id_column, "image_path", "page", "frame"}
                    ),
                )
            )
            seen.add(observation)
        if missing := links.keys() - seen:
            raise ImageIngestionError(
                f"{modality.name}: missing declared image observations {sorted(missing)}"
            )
    except ValueError as exc:
        if isinstance(exc, ImageIngestionError):
            raise
        raise ImageIngestionError(str(exc)) from exc
    return tuple(references)
