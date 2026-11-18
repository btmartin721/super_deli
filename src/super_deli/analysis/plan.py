"""Resolve declared designs against existing workflows without opening measurement data."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from super_deli.core.data import ENVIRONMENTAL_SCHEMA_VERSION, DatasetConfig, Modality
from super_deli.core.provenance import configuration_hash
from super_deli.core.schema import ConfigError, config_from_dict
from super_deli.core.validation import resolve_input, validate_dataset
from super_deli.io.config import read_json

SINGLETONS = {
    "genomic": "baseline-genomic",
    "morphometric": "cluster-shape",
    "ecological": "cluster-tabular",
    "morphological": "cluster-tabular",
}
METHOD_OUTPUTS = {
    "baseline-genomic": "Genomic specimen partition and conditional memberships.",
    "cluster-shape": "Shape partition and conditional memberships at the measured level.",
    "cluster-tabular": "Tabular partition and conditional memberships at the measured level.",
    "cluster-integrative": "One joint partition of the declared target units.",
    "analyze-matched-genomic": "Singleton and combined partitions on the same specimens.",
    "analyze-matched-multimodal": "Singleton and combined partitions on the same specimens.",
    "analyze-matched-tabular": (
        "One joint partition per prespecified PCA or separate-encoder GMM arm, "
        "with same-cohort agreement among arms."
    ),
}


@dataclass(frozen=True)
class AnalysisPlan:
    dataset: dict
    status: Literal["ready", "conditional", "unsupported"]
    selection_capability: Literal["implemented", "partial", "planned"]
    target_level: str
    coverage: dict
    views: tuple[dict, ...]
    eligible_native_methods: tuple[str, ...]
    native_method_outputs: dict[str, str]
    legacy_environment_route: str | None
    same_cohort_singletons: bool
    standalone_routes: tuple[dict, ...]
    reasons: tuple[str, ...]
    outstanding_checks: tuple[str, ...]
    shared_source_views: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict:
        return json.loads(json.dumps({"schema_version": 1, **asdict(self)}, allow_nan=False))


def _validate(config: DatasetConfig, source: Path, *, check_files: bool) -> None:
    result = validate_dataset(
        config, base_dir=source.parent, check_files=check_files, allow_environmental=True
    )
    if not result.valid:
        raise ConfigError(
            "; ".join(f"{i.location}: {i.message}" for i in result.issues if i.severity == "error")
        )


def _views(config, selected, target_level, aggregation, source):
    specimens = {s.specimen_id: s for s in config.specimens}
    for view in selected:
        same_level = view.level == target_level
        can_group = view.level == "specimen" and target_level != "specimen"
        expected = "mean" if can_group else "none" if same_level else None
        policy = aggregation.get(view.name)
        if policy is not None and expected is not None and policy != expected:
            raise ConfigError(f"{view.name}: {view.level} to {target_level} requires {expected}")
        links, unlinked = defaultdict(list), []
        for record in view.observations:
            if same_level:
                target = record.unit_id
            elif can_group:
                target = getattr(specimens[record.unit_id], f"{target_level}_id")
            else:
                target = None
            if target is None:
                unlinked.append(record.observation_id)
            else:
                links[target].append(record.observation_id)
        yield {
            "name": view.name,
            "kind": view.kind,
            "source_level": view.level,
            "format": view.format,
            "path": str(resolve_input(view.path, source.parent)),
            "popmapfile": (
                str(resolve_input(view.popmapfile, source.parent)) if view.popmapfile else None
            ),
            "aggregation": policy if can_group else expected,
            "required_aggregation": expected,
            "observation_count": len(view.observations),
            "linked_target_count": len(links),
            "observations": [
                {"observation_id": o.observation_id, "unit_id": o.unit_id}
                for o in view.observations
            ],
            "target_observation_ids": dict(links),
            "unlinked_observation_ids": unlinked,
            "features": [
                {"name": f.name, "type": f.dtype, "role": f.role, "unit": f.unit}
                for f in view.features
            ],
            "environmental_options": view.environmental_options,
        }


def _shape_prerequisite(view: Modality) -> str | None:
    if view.kind == "morphometric":
        geometry = view.morphometric_options
        if view.format == "image_manifest" or geometry is None:
            return "supply reviewed fixed landmarks and declared geometry."
        if len(geometry.landmark_ids) < 3:
            return "declare at least three fixed homologous landmarks for Procrustes alignment."
    return None


def _feature_types(view: Modality) -> set[str]:
    return {f.dtype for f in view.features if f.role == "feature"}


def _standalone(view: Modality) -> str | None:
    if _shape_prerequisite(view):
        return None
    if view.kind in {"ecological", "morphological"} and not (
        _feature_types(view) & {"continuous", "count"}
    ):
        return None
    if len({o.unit_id for o in view.observations}) != len(view.observations):
        return None
    return SINGLETONS.get(view.kind)


def plan_analysis(
    dataset: str | Path,
    *,
    modalities: Sequence[str],
    target_level: str,
    missing_views: str = "error",
    aggregation: Mapping[str, str] | None = None,
) -> AnalysisPlan:
    """Read a dataset JSON declaration and describe feasible native workflows.

    Schema 2 accepts a CSV/TSV environmental kind and the usual typed features
    with units. Declare environmental_options with source and spatial_support
    (point, site, or grid_cell). Optional records identify observations by
    observation_id and may record date, time_window, source_cell_id,
    sampling_event_id, and a pair of coordinates. Coordinates require a
    coordinate_reference_system on the view.
    """
    source = Path(dataset).resolve()
    config = config_from_dict(read_json(source), allow_environmental=True)
    _validate(config, source, check_files=False)
    if target_level not in ("specimen", "population", "locality"):
        raise ConfigError("target_level must be specimen, population or locality")
    if missing_views not in ("error", "complete_cases"):
        raise ConfigError("missing_views must be error or complete_cases")
    if (
        isinstance(modalities, str)
        or not isinstance(modalities, Sequence)
        or not modalities
        or any(not isinstance(name, str) for name in modalities)
        or len(set(modalities)) != len(modalities)
    ):
        raise ConfigError("Select a nonempty sequence of unique modality names")
    names = set(modalities)
    if unknown := names - {m.name for m in config.modalities}:
        raise ConfigError(f"Unknown modalities: {sorted(unknown)}")
    if aggregation is None:
        aggregation = {}
    if not isinstance(aggregation, Mapping) or any(
        name not in names or policy not in ("none", "mean") for name, policy in aggregation.items()
    ):
        raise ConfigError("Aggregation must map selected modality names to none or mean")
    selected = tuple(m for m in config.modalities if m.name in names)
    _validate(replace(config, modalities=selected), source, check_files=True)

    views = tuple(_views(config, selected, target_level, aggregation, source))
    declared = (
        tuple(s.specimen_id for s in config.specimens)
        if target_level == "specimen"
        else config.populations
        if target_level == "population"
        else config.localities
    )
    intersection = set.intersection(*(set(v["target_observation_ids"]) for v in views))
    excluded = [
        {
            "unit_id": unit,
            "missing_views": [v["name"] for v in views if unit not in v["target_observation_ids"]],
        }
        for unit in declared
        if unit not in intersection
    ]
    counts = Counter(m.kind for m in selected)
    environment = "environmental" in counts
    separate_blocks = environment and "ecological" in counts
    capability = "planned" if separate_blocks else "partial" if environment else "implemented"
    integrated = len(selected) > 1
    tabular_types = {m.name: _feature_types(m) for m in selected if m.features}
    matched_specimens = target_level == "specimen" and missing_views == "error"
    singleton_coordinator = (
        integrated
        and matched_specimens
        and not environment
        and bool({"morphometric", "genomic"} & counts.keys())
        and all("continuous" in types for types in tabular_types.values())
    )
    # The native matched tabular contract accepts 3–32 selected continuous
    # features per view and its shared alignment requires ten complete specimens.
    matched_tabular = (
        matched_specimens
        and counts == {"morphological": 1, "ecological": 1}
        and len(declared) >= 10
        and all(
            sum(f.dtype == "continuous" and f.role == "feature" for f in view.features) >= 3
            for view in selected
        )
    )
    schema_conversion = config.schema_version == ENVIRONMENTAL_SCHEMA_VERSION
    unsupported, conditional = [], []
    checks = ["Measurement completeness and source-row identities require a content check."]
    if "genomic" in counts:
        checks.append("Verify supported diploid biallelic genotypes through SNPio.")
    if "morphometric" in counts:
        checks.append("Verify reviewed fixed homologous landmark coordinates.")
    if "ecological" in counts and integrated and target_level == "specimen":
        checks.append("Confirm ecology was measured on each specimen.")
        if singleton_coordinator or matched_tabular:
            checks.append("Declare ecology_measurement_scope='individual' in the native settings.")
    if matched_tabular:
        checks.extend(
            (
                "Prespecify joint arms with the same 3–32 continuous analytical features per view.",
                "Declare independent specimens in the native arm settings.",
                "Verify selected measurements are finite with enough variation for each arm.",
            )
        )
    if schema_conversion and not environment:
        conditional.append("Supply a schema-1 dataset declaration of the selected views.")
    if any(count > 1 for count in counts.values()):
        unsupported.append("Native analysis accepts at most one selected view per kind.")

    for modality, view in zip(selected, views, strict=True):
        name = modality.name
        if view["required_aggregation"] is None:
            conditional.append(
                f"{name}: no native link from {modality.level} to {target_level}. "
                "Choose a common target. Group measurements cannot be broadcast to specimens."
            )
        elif view["required_aggregation"] == "mean" and view["aggregation"] is None:
            conditional.append(f"{name}: declare mean aggregation to {target_level}.")
        if view["required_aggregation"] and view["unlinked_observation_ids"]:
            conditional.append(f"{name}: provide missing explicit {target_level} links.")
        if len({record.unit_id for record in modality.observations}) < len(modality.observations):
            conditional.append(
                f"{name}: resolve repeated observations to one record per measured unit."
            )
        if prerequisite := _shape_prerequisite(modality):
            conditional.append(f"{name}: {prerequisite}")
        if modality.features:
            types = tabular_types[name]
            if not types:
                unsupported.append(f"{name}: covariate-only input requires an adjustment method.")
            elif not types & {"continuous", "count"}:
                conditional.append(f"{name}: declare eligible continuous or count features.")
    if not integrated and missing_views == "complete_cases":
        unsupported.append("Native complete-case integration requires at least two views.")
    if not integrated and views[0]["required_aggregation"] == "mean":
        unsupported.append("Native standalone methods partition units at their measured level.")
    if not intersection:
        conditional.append("No declared units have links to every selected view.")
    elif (
        integrated
        and missing_views == "error"
        and (
            (target_level == "specimen" and excluded)
            or any(len(row["missing_views"]) < len(views) for row in excluded)
        )
    ):
        conditional.append(
            "Some declared target units lack selected views under missing_views=error."
        )
    if integrated and len(intersection) == 1:
        conditional.append("Integration requires at least two shared target units.")

    shared = defaultdict(list)
    for view in views:
        shared[view["path"]].append(view["name"])
    shared_sources = tuple(tuple(group) for group in shared.values() if len(group) > 1)
    if len(views) > 1:
        checks.append("Check shared measurement provenance before interpreting corroboration.")
    if shared_sources:
        checks.append("Views sharing a source file require provenance review for independence.")

    legacy_environment = None
    if separate_blocks:
        unsupported.insert(0, "Native methods lack distinct ecology and environment blocks.")
    elif environment:
        reason = "First-class environment has no native execution route."
        if not unsupported and not conditional:
            legacy_environment = "cluster-integrative" if integrated else "cluster-tabular"
            reason += (
                f" {legacy_environment} can use a compatible numeric table explicitly "
                "declared in the schema-1 ecology slot."
            )
        unsupported.insert(0, reason)
    if singleton_coordinator:
        method = (
            "analyze-matched-multimodal" if "morphometric" in counts else "analyze-matched-genomic"
        )
    elif integrated:
        method = "cluster-integrative"
    else:
        method = SINGLETONS.get(selected[0].kind)
    if method == "cluster-integrative" and (not unsupported or legacy_environment):
        checks.append("The lower-level integrator does not produce same-cohort singleton fits.")
    if (
        integrated
        and target_level == "specimen"
        and missing_views == "error"
        and any("continuous" not in types and "count" in types for types in tabular_types.values())
    ):
        checks.append("Matched singleton comparisons require continuous tabular features.")
    status = "unsupported" if unsupported else "conditional" if conditional else "ready"
    methods = (method,) if status == "ready" and method is not None else ()
    if methods and matched_tabular:
        methods += ("analyze-matched-tabular",)
    return AnalysisPlan(
        dataset={
            "id": config.dataset_id,
            "path": str(source),
            "schema_version": config.schema_version,
            "configuration_sha256": configuration_hash(config.to_dict()),
        },
        status=status,
        selection_capability=capability,
        target_level=target_level,
        coverage={
            "missing_views": missing_views,
            "declared_count": len(declared),
            "linked_count": len(intersection),
            "excluded_count": len(excluded),
            "linked_unit_ids": [unit for unit in declared if unit in intersection],
            "exclusions": excluded,
            "measurement_completeness": "unchecked",
        },
        views=views,
        eligible_native_methods=methods,
        native_method_outputs={name: METHOD_OUTPUTS[name] for name in methods},
        legacy_environment_route=legacy_environment,
        same_cohort_singletons=status == "ready" and singleton_coordinator,
        standalone_routes=tuple(
            {"view": m.name, "target_level": m.level, "method": method_name}
            for m in selected
            if not schema_conversion and (method_name := _standalone(m))
        ),
        reasons=tuple(unsupported + conditional),
        outstanding_checks=tuple(checks),
        shared_source_views=shared_sources,
    )
