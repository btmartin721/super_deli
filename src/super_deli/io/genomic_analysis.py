"""Freeze selected genomic/phenotype sources and reject incomplete or unmatched cohorts."""

from dataclasses import replace
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.core.validation import resolve_input
from super_deli.io.config import write_json
from super_deli.io.matched_analysis import verify_sources
from super_deli.io.staging import stage_input


def freeze_genomic_analysis(dataset, config, base_dir, directory):
    import numpy as np

    from super_deli.io.genomic import ingest_genomic
    from super_deli.io.genomic_encoding import encode_diploid
    from super_deli.io.tabular import ingest_tabular

    out = Path(directory)
    out.mkdir()
    singletons = [a for a in config.arms if a.family != "integrative"]
    all_view = next(a for a in config.arms if len(a.modalities) == len(singletons))
    views = {v.modality: v for v in all_view.model.views}
    available = {m.name: m for m in dataset.modalities}
    wanted = sorted(s.specimen_id for s in dataset.specimens)
    if not wanted:
        raise ValueError("Explicit specimen declarations are required")
    modalities, sources, records = [], [], []
    for i, arm in enumerate(singletons):
        name = arm.modalities[0]
        if name not in available:
            raise ValueError("Selected modality is absent from dataset")
        m = available[name]
        if m.level != "specimen" or m.kind != views[name].kind:
            raise ValueError("Every selected view requires its declared specimen-level kind")
        links = {o.unit_id: o.observation_id for o in m.observations}
        if len(links) != len(m.observations) or set(links) != set(wanted):
            raise ValueError(
                "Each selected view requires exactly one observation per declared specimen"
            )
        updates = {}
        for role, value in (("path", m.path), ("popmapfile", m.popmapfile)):
            if value is None:
                continue
            source = resolve_input(value, base_dir)
            before = fingerprint_file(source)
            snapshot, original = stage_input(source, out / f"view_{i:03d}_{role}")
            if original["sha256"] != before.sha256:
                raise ValueError("Input changed while freezing")
            relative = snapshot.relative_to(out).as_posix()
            updates[role] = relative
            sources.append(dict(modality=name, role=role, snapshot=relative, source=original))
        modalities.append(replace(m, **updates))
        records.append(
            dict(
                modality=name,
                kind=m.kind,
                observation_ids=[links[u] for u in wanted],
                feature_names=list(getattr(arm.model, "feature_names", ())),
            )
        )
    frozen = replace(dataset, modalities=tuple(modalities))
    genomic = next(a for a in singletons if a.family == "genomic")
    loaded = ingest_genomic(
        frozen,
        base_dir=out,
        output_dir=out / "genomic-ingestion",
        modality_name=genomic.modalities[0],
    )
    values, _ = encode_diploid(
        loaded, output_prefix=out / "encoding", random_seed=genomic.model.random_seed
    )
    if set(loaded.specimen_ids) != set(wanted) or len(loaded.specimen_ids) != len(wanted):
        raise ValueError("Genomic source does not match the declared specimen cohort")
    if not np.isfinite(values).all():
        raise ValueError("This comparison requires complete supported diploid biallelic genotypes")
    if genomic.block_ids is not None and len(genomic.block_ids) != values.shape[1]:
        raise ValueError("Block map must cover original source loci before filtering")
    tables = ingest_tabular(
        frozen,
        base_dir=out,
        output_dir=out / "tabular-ingestion",
        modality_names=[a.modalities[0] for a in singletons if a.family == "tabular"],
    )
    for table in tables.tables:
        names = views[table.modality_name].feature_names
        features = {f.name: (j, f) for j, f in enumerate(table.features)}
        if any(
            n not in features
            or features[n][1].dtype != "continuous"
            or features[n][1].role != "feature"
            for n in names
        ):
            raise ValueError("Select declared continuous feature columns only")
        selected = [features[n][0] for n in names]
        if any(row[j] is None for row in table.values for j in selected):
            raise ValueError("This comparison requires complete selected phenotype features")
        if set(table.unit_ids) != set(wanted) or len(table.unit_ids) != len(wanted):
            raise ValueError("Phenotype source does not match the declared specimen cohort")
    cohort = dict(
        unit_ids=wanted,
        views=records,
        independence="assumed_not_verified",
        ecology_measurement_scope=config.ecology_measurement_scope,
        source_locus_count=values.shape[1],
    )
    write_json(frozen.to_dict(), out / "dataset.json")
    write_json(cohort, out / "cohort.json")
    verify_sources(out, sources)
    return frozen, cohort, sources
