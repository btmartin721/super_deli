"""Freeze two sources and validate one complete specimen cohort before fitting."""

from dataclasses import replace
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.core.validation import resolve_input
from super_deli.io.config import write_json
from super_deli.io.staging import stage_input
from super_deli.io.tabular import ingest_tabular


def verify_sources(directory, sources):
    for source in sources:
        if (
            fingerprint_file(Path(directory) / source["snapshot"]).sha256
            != source["source"]["sha256"]
        ):
            raise ValueError("Frozen matched input hash mismatch")


def freeze_matched_inputs(dataset, analysis, base_dir, directory):
    from super_deli.models.matched_preprocessing import align_tables

    out = Path(directory)
    out.mkdir()
    common = analysis.arms[0].alignment_config()
    selected = {m.name: m for m in dataset.modalities}
    modalities, sources = [], []
    for i, name in enumerate((common.morphology, common.ecology)):
        if name not in selected:
            raise ValueError("Both selected morphology/ecology modalities must be declared")
        m = selected[name]
        if m.level != "specimen" or m.kind != ("morphological" if i == 0 else "ecological"):
            raise ValueError("Both views must be specimen-level measurements of the declared kind")
        source = resolve_input(m.path, base_dir)
        before = fingerprint_file(source)
        snapshot, original = stage_input(source, out / f"view_{i:03d}")
        if original["sha256"] != before.sha256:
            raise ValueError("Source changed while freezing matched inputs")
        relative = snapshot.relative_to(out).as_posix()
        modalities.append(replace(m, path=relative))
        sources.append(dict(modality=name, snapshot=relative, source=original))
    frozen = replace(dataset, modalities=tuple(modalities))
    loaded = ingest_tabular(frozen, base_dir=out, output_dir=out / "ingestion")
    # Validate all arms' resource bounds using the same existing fit-free join.
    for arm in analysis.arms:
        _, cohort = align_tables(loaded.tables, arm.alignment_config())
    if set(cohort["unit_ids"]) != {s.specimen_id for s in dataset.specimens}:
        raise ValueError("Every declared specimen requires both complete selected views")
    cohort.update(ecology_measurement_scope="individual", independence="assumed_not_verified")
    write_json(frozen.to_dict(), out / "dataset.json")
    write_json(cohort, out / "cohort.json")
    verify_sources(out, sources)
    return frozen, cohort, sources
