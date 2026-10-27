"""Fit-free complete-cohort checks and immutable source snapshots for grouped adapters."""

import shutil
from dataclasses import asdict, replace
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.validation import resolve_input
from super_deli.io.config import write_json
from super_deli.io.tabular import ingest_tabular


def freeze_grouped_inputs(dataset, model, base_dir, directory):
    import numpy as np

    from super_deli.models.tabular_gmm import _matrix
    from super_deli.workflows.integrative import _target_ids

    out = Path(directory)
    out.mkdir()
    by_name = {m.name: m for m in dataset.modalities}
    modalities = []
    fingerprints = []
    for i, view in enumerate(model.views):
        if view.modality not in by_name or by_name[view.modality].kind != view.kind:
            raise ValueError("Selected grouped views must match declared modalities")
        source_modality = by_name[view.modality]
        expected = "specimen" if view.kind == "morphological" else model.target_level
        if source_modality.level != expected:
            raise ValueError("Explicit specimen/target levels required")
        source = resolve_input(source_modality.path, base_dir)
        before = fingerprint_file(source)
        target = out / f"view_{i:03d}.{source_modality.format}"
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if (
            fingerprint_file(source).sha256 != before.sha256
            or fingerprint_file(target).sha256 != before.sha256
        ):
            raise OSError("Source changed while freezing the grouped cohort")
        modalities.append(replace(source_modality, path=target.name))
        fingerprints.append(
            {"modality": view.modality, "snapshot": target.name, "source": asdict(before)}
        )
    frozen = replace(dataset, modalities=tuple(modalities))
    loaded = ingest_tabular(frozen, base_dir=out, output_dir=out / "ingestion")
    tables = {t.modality_name: t for t in loaded.tables}
    units = None
    views = []
    declared = set(dataset.localities if model.target_level == "locality" else dataset.populations)
    for view in model.views:
        table = tables[view.modality]
        values, names, types = _matrix(table, view.tabular_config(model.max_working_bytes))
        if set(types) != {"continuous"} or not np.isfinite(values).all():
            raise ValueError("Select complete finite continuous grouped features")
        targets = _target_ids(dataset, table.level, table.unit_ids, view, model.target_level)
        if set(targets) != declared:
            raise ValueError("Every declared target requires both complete views")
        if view.kind == "morphological":
            if set(table.unit_ids) != {s.specimen_id for s in dataset.specimens}:
                raise ValueError("Every declared morphology specimen requires exactly one row")
        elif len(targets) != len(set(targets)):
            raise ValueError("Exactly one ecology observation per target required")
        if units is None:
            units = list(dict.fromkeys(targets))
        views.append(
            {
                "modality": view.modality,
                "kind": view.kind,
                "values_sha256": configuration_hash(values.tolist()),
                "feature_names": list(names),
                "observation_ids": list(table.observation_ids),
                "unit_ids": list(table.unit_ids),
                "target_ids": list(targets),
            }
        )
    groups = [
        {
            "target_id": u,
            "morphology_count": views[0]["target_ids"].count(u),
            "ecology_count": views[1]["target_ids"].count(u),
            "morphology_specimens": [
                s
                for s, t in zip(views[0]["unit_ids"], views[0]["target_ids"], strict=True)
                if t == u
            ],
        }
        for u in units
    ]
    cohort = {
        "unit_ids": units,
        "groups": groups,
        "views": views,
        "target_level": model.target_level,
    }
    write_json(frozen.to_dict(), out / "dataset.json")
    write_json(cohort, out / "cohort.json")
    return frozen, cohort, fingerprints
