"""Strict morphology/ecology CSV/TSV ingestion without scientific dependencies."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from super_deli.core.data import DatasetConfig, Feature, Modality, TabularReaderOptions
from super_deli.core.provenance import collect_provenance
from super_deli.core.schema import config_from_dict
from super_deli.core.tabular import TabularTable, TabularValue
from super_deli.core.validation import resolve_input, validate_dataset
from super_deli.io.config import write_json
from super_deli.io.staging import stage_input

_NUMERIC = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_COUNT = re.compile(r"[0-9]+\Z")
_KINDS = {"morphological", "ecological"}


class TabularIngestionError(ValueError):
    """A table violates its declared schema, identifiers, or biological links."""


@dataclass(frozen=True)
class LoadedTabularData:
    tables: tuple[TabularTable, ...]
    output_dir: Path
    summary: dict


def _select_modalities(config: DatasetConfig, names: Sequence[str] | None) -> tuple[Modality, ...]:
    if names is not None:
        if (
            isinstance(names, str)
            or not isinstance(names, Sequence)
            or not names
            or any(not isinstance(n, str) for n in names)
        ):
            raise TabularIngestionError("modality_names must be a nonempty sequence of names")
        if len(set(names)) != len(names):
            raise TabularIngestionError("Duplicate requested modality names")
        available = {m.name for m in config.modalities if m.kind in _KINDS}
        if unknown := set(names) - available:
            raise TabularIngestionError(f"Unknown or non-tabular modalities: {sorted(unknown)}")
    selected = tuple(
        m for m in config.modalities if m.kind in _KINDS and (names is None or m.name in names)
    )
    if not selected:
        raise TabularIngestionError("Declare at least one morphological or ecological table")
    return selected


def _identifier(value: str, location: str) -> str:
    if not value or value != value.strip() or "\x00" in value:
        raise TabularIngestionError(f"{location}: identifiers must be nonempty and unpadded")
    return value


def _value(
    text: str, feature: Feature, options: TabularReaderOptions, location: str
) -> TabularValue:
    if text in options.missing_values:
        return None
    if text != text.strip() or "\x00" in text:
        raise TabularIngestionError(f"{location}: values must be unpadded text without NUL")
    if feature.dtype in {"binary", "ordinal", "categorical"}:
        if text not in feature.categories:
            raise TabularIngestionError(f"{location}: unknown {feature.dtype} state {text!r}")
        return text
    if feature.dtype == "count":
        if not _COUNT.fullmatch(text):
            raise TabularIngestionError(f"{location}: count must be a nonnegative integer literal")
        try:
            return int(text)
        except ValueError as exc:
            raise TabularIngestionError(
                f"{location}: count exceeds the integer parser limit"
            ) from exc
    if not _NUMERIC.fullmatch(text):
        raise TabularIngestionError(f"{location}: continuous value must be a finite decimal number")
    value = float(text)
    if not math.isfinite(value):
        raise TabularIngestionError(f"{location}: continuous value is outside finite float range")
    mantissa = re.split("[eE]", text)[0]
    if value == 0 and any(c in "123456789" for c in mantissa):
        raise TabularIngestionError(f"{location}: nonzero continuous value underflows float range")
    return value


def _read_table(path: Path, modality: Modality) -> TabularTable:
    options = modality.tabular_options or TabularReaderOptions()
    links = {o.observation_id: o.unit_id for o in modality.observations}
    expected = {
        options.observation_id_column,
        *options.ignored_columns,
        *(f.name for f in modality.features),
    }
    if options.unit_id_column is not None:
        expected.add(options.unit_id_column)
    observations, units, values = [], [], []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(
            handle, delimiter="," if modality.format == "csv" else "\t", strict=True
        )
        try:
            header = next(reader, [])
            if not header:
                raise TabularIngestionError(f"{modality.name}: missing table header")
            for column in header:
                _identifier(column, f"{modality.name} header")
            if duplicate := [name for name, n in Counter(header).items() if n > 1]:
                raise TabularIngestionError(f"{modality.name}: duplicate columns {duplicate}")
            if set(header) != expected:
                raise TabularIngestionError(
                    f"{modality.name}: header mismatch: missing={sorted(expected - set(header))}; "
                    f"undeclared={sorted(set(header) - expected)}"
                )
            columns = {name: i for i, name in enumerate(header)}
            for row in reader:
                location = f"{modality.name} line {reader.line_num}"
                if len(row) != len(header):
                    raise TabularIngestionError(
                        f"{location}: row has {len(row)} fields, expected {len(header)}"
                    )
                observation = _identifier(row[columns[options.observation_id_column]], location)
                if observation in seen:
                    raise TabularIngestionError(
                        f"{location}: duplicate observation ID {observation!r}"
                    )
                if observation not in links:
                    raise TabularIngestionError(
                        f"{location}: undeclared observation ID {observation!r}"
                    )
                unit = links[observation]
                if options.unit_id_column is not None:
                    declared_unit = _identifier(row[columns[options.unit_id_column]], location)
                    if declared_unit != unit:
                        raise TabularIngestionError(
                            f"{location}: unit ID conflicts with observation declaration"
                        )
                values.append(
                    tuple(
                        _value(row[columns[f.name]], f, options, f"{location} feature {f.name!r}")
                        for f in modality.features
                    )
                )
                observations.append(observation)
                units.append(unit)
                seen.add(observation)
        except (csv.Error, UnicodeError) as exc:
            raise TabularIngestionError(
                f"{modality.name} line {reader.line_num}: invalid UTF-8 CSV/TSV: {exc}"
            ) from exc
    if missing := set(links) - seen:
        raise TabularIngestionError(
            f"{modality.name}: missing declared observations {sorted(missing)}"
        )
    return TabularTable(
        modality.name,
        modality.kind,
        modality.level,
        modality.features,
        tuple(observations),
        tuple(units),
        tuple(values),
    )


def _table_summary(table: TabularTable, modality: Modality, coverage: dict, artifact: str) -> dict:
    missing = [sum(row[i] is None for row in table.values) for i in range(len(table.features))]
    analytical = [i for i, f in enumerate(table.features) if f.role == "feature"]
    return {
        "name": table.modality_name,
        "kind": table.kind,
        "level": table.level,
        "artifact": artifact,
        "source_format": modality.format,
        "reader_options": asdict(modality.tabular_options or TabularReaderOptions()),
        "coverage": coverage,
        "feature_order": [f.name for f in table.features],
        "missing_count_by_column": dict(
            zip((f.name for f in table.features), missing, strict=True)
        ),
        "all_missing_columns": [
            f.name for f, n in zip(table.features, missing, strict=True) if n == len(table.values)
        ],
        "all_missing_analytical_observation_ids": [
            o
            for o, row in zip(table.observation_ids, table.values, strict=True)
            if all(row[i] is None for i in analytical)
        ],
    }


def ingest_tabular(
    config: DatasetConfig,
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    modality_names: Sequence[str] | None = None,
) -> LoadedTabularData:
    """Load selected morphology/ecology tables; default to all declared tables.

    Source rows must exactly match the observation declarations by ID. Row order,
    repeated units, missing values, roles, units and category order are preserved.
    No scientific imports, encoding, imputation, aggregation or fitting occurs.
    A new output directory holds staged inputs and typed JSON artifacts. summary.json
    marks completion; failures after creation leave failure.json and diagnostics.
    """
    config = config_from_dict(config.to_dict())
    report = validate_dataset(config, base_dir=base_dir, check_files=False)
    if not report.valid:
        raise TabularIngestionError(
            "; ".join(f"{i.location}: {i.message}" for i in report.issues if i.severity == "error")
        )
    selected = _select_modalities(config, modality_names)
    sources = [resolve_input(m.path, base_dir) for m in selected]
    for path in sources:
        if not path.is_file():
            raise TabularIngestionError(f"Not a regular input file: {path}")
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    try:
        tables, fingerprints, summaries = [], [], []
        coverage = {c.modality: asdict(c) for c in report.coverage}
        for i, (modality, source) in enumerate(zip(selected, sources, strict=True)):
            # User modality names are identifiers, never output path components.
            directory = destination / f"modality_{i:03d}"
            directory.mkdir()
            staged, fingerprint = stage_input(source, directory / "input")
            fingerprints.append(fingerprint)
            table = _read_table(staged, modality)
            artifact = f"modality_{i:03d}/table.json"
            write_json(table.to_dict(), destination / artifact)
            tables.append(table)
            summaries.append(_table_summary(table, modality, coverage[modality.name], artifact))
        provenance = collect_provenance(config, base_dir=base_dir)
        provenance.update(input_hash_status="computed_selected_modalities", inputs=fingerprints)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "dataset_id": config.dataset_id,
            "modalities": summaries,
            "provenance": provenance,
            "processing": {
                "imputation": "none",
                "category_encoding": "labels_preserved",
                "unit_conversion": "none",
                "covariate_adjustment": "none",
                "cross_modality_alignment": "none",
            },
        }
        write_json(summary, destination / "summary.json")
        return LoadedTabularData(tuple(tables), destination, summary)
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
