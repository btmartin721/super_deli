"""Validate declared biological links and local file references without fitting models."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from super_deli.core.data import DatasetConfig
from super_deli.core.schema import config_from_dict

FORMATS = {
    "genomic": frozenset({"vcf", "phylip", "structure", "genepop"}),
    "morphological": frozenset({"csv", "tsv"}),
    "ecological": frozenset({"csv", "tsv"}),
    "morphometric": frozenset({"tps", "landmarks_csv", "image_manifest"}),
}


@dataclass(frozen=True)
class Issue:
    code: str
    location: str
    message: str
    severity: Literal["error", "warning"] = "error"


@dataclass(frozen=True)
class Coverage:
    modality: str
    level: str
    observation_count: int
    unit_count: int
    repeated_observation_count: int
    missing_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class ValidationReport:
    dataset_id: str
    issues: tuple[Issue, ...]
    coverage: tuple[Coverage, ...]
    checked_files: bool

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict:
        return {"schema_version": 1, "valid": self.valid, **asdict(self)}


def resolve_input(path: str, base_dir: str | Path) -> Path:
    """Accept absolute paths; resolve relative references independently of the shell cwd."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate
    return candidate.resolve()


def input_paths(config: DatasetConfig, base_dir: str | Path) -> tuple[Path, ...]:
    """Return unique declared files in deterministic order."""
    paths = {
        resolve_input(path, base_dir)
        for modality in config.modalities
        for path in (modality.path, modality.popmapfile)
        if path is not None
    }
    return tuple(sorted(paths))


def validate_dataset(
    config: DatasetConfig, *, base_dir: str | Path, check_files: bool = True
) -> ValidationReport:
    """Check declarations, not file contents or statistical readiness.

    No imputation, alignment, aggregation, or cross-modality matching is performed.
    Replicates remain linked to their original unit. Missing modalities are allowed.
    """
    config = config_from_dict(config.to_dict())
    issues: list[Issue] = []
    coverage: list[Coverage] = []

    def error(code: str, location: str, message: str):
        issues.append(Issue(code, location, message))

    def unique(values, location: str):
        for value, count in sorted(Counter(values).items()):
            if count > 1:
                error("duplicate_id", location, f"{value!r} occurs {count} times")

    units = {
        "specimen": {specimen.specimen_id for specimen in config.specimens},
        "population": set(config.populations),
        "locality": set(config.localities),
    }
    unique((s.specimen_id for s in config.specimens), "specimens")
    unique(config.populations, "populations")
    unique(config.localities, "localities")
    unique((m.name for m in config.modalities), "modalities")
    if not any(units.values()):
        error("no_units", "dataset", "Declare at least one specimen, population, or locality")
    if not config.modalities:
        error("no_modalities", "modalities", "At least one data type is required")
    for i, specimen in enumerate(config.specimens):
        for level in ("population", "locality"):
            value = getattr(specimen, f"{level}_id")
            if value is not None and value not in units[level]:
                error("unknown_unit", f"specimens[{i}].{level}_id", f"Unknown {level}: {value!r}")

    for i, modality in enumerate(config.modalities):
        location = f"modalities[{i}]"
        if modality.kind not in FORMATS:
            error("unsupported_kind", f"{location}.kind", f"Unknown data type: {modality.kind!r}")
        elif modality.format not in FORMATS[modality.kind]:
            error("unsupported_format", f"{location}.format", f"Invalid format for {modality.kind}")
        if modality.level not in units:
            error("unsupported_level", f"{location}.level", "Use specimen, population, or locality")
        if modality.kind == "genomic" and modality.level != "specimen":
            error("invalid_level", location, "GenotypeData records must link to specimens")
        if modality.popmapfile is not None and modality.kind != "genomic":
            error("invalid_popmap", location, "popmapfile is only defined for genomic inputs")
        if not modality.observations:
            error("no_observations", location, "Declare at least one source observation")
        unique((o.observation_id for o in modality.observations), f"{location}.observations")
        present = {o.unit_id for o in modality.observations}
        if modality.level in units:
            for unit in sorted(present - units[modality.level]):
                error("unknown_unit", f"{location}.observations", f"Unknown unit: {unit!r}")
            coverage.append(
                Coverage(
                    modality.name,
                    modality.level,
                    len(modality.observations),
                    len(present),
                    len(modality.observations) - len(present),
                    tuple(sorted(units[modality.level] - present)),
                )
            )
        unique((f.name for f in modality.features), f"{location}.features")
        if modality.kind in {"morphological", "ecological"}:
            if not any(f.role == "feature" for f in modality.features):
                error("no_features", location, "Declare at least one analytical feature")
        elif modality.features:
            error(
                "invalid_features", location, "Tabular feature schemas apply to morphology/ecology"
            )
        for j, feature in enumerate(modality.features):
            fpath = f"{location}.features[{j}]"
            if feature.role not in {"feature", "covariate"}:
                error("unsupported_role", fpath, "Use feature or covariate")
            if feature.dtype not in {"continuous", "count", "binary", "ordinal", "categorical"}:
                error("unsupported_dtype", fpath, f"Unknown feature type: {feature.dtype!r}")
            unique(feature.categories, f"{fpath}.categories")
            if feature.dtype in {"binary", "ordinal", "categorical"}:
                minimum = 1 if feature.dtype == "categorical" else 2
                if len(feature.categories) < minimum:
                    error("missing_categories", fpath, "Declare the allowed categorical states")
                if feature.dtype == "binary" and len(feature.categories) != 2:
                    error("invalid_categories", fpath, "Binary features require exactly two states")
            elif feature.categories:
                error("invalid_categories", fpath, "Numeric features cannot declare categories")

    if check_files:
        for i, modality in enumerate(config.modalities):
            for field in ("path", "popmapfile"):
                value = getattr(modality, field)
                if value is None:
                    continue
                try:
                    path = resolve_input(value, base_dir)
                    if not path.is_file():
                        raise OSError("not a regular file")
                    with path.open("rb"):
                        pass
                except (OSError, ValueError, RuntimeError) as exc:
                    error("unreadable_file", f"modalities[{i}].{field}", f"{value!r}: {exc}")
    issues.append(
        Issue(
            "contents_not_checked",
            "dataset",
            "Source rows, encodings, landmarks, and model suitability have not been checked",
            "warning",
        )
    )
    return ValidationReport(config.dataset_id, tuple(issues), tuple(coverage), check_files)
