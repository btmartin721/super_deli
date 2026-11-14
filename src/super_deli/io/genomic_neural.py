"""One immutable SNPio read and explicit original-locus/allele coding evidence."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from super_deli.core.provenance import fingerprint_file
from super_deli.core.validation import resolve_input
from super_deli.io.config import write_json
from super_deli.io.genomic import _select_modality, ingest_genomic
from super_deli.io.genomic_encoding import encode_diploid
from super_deli.io.staging import stage_input
from super_deli.models.genomic_neural import validate_dosages


def prepare_genotypes(dataset, config, base_dir, directory, modality_name=None):
    out = Path(directory)
    out.mkdir()
    m = _select_modality(dataset, modality_name)
    units = tuple(o.unit_id for o in m.observations)
    if len(set(units)) != len(units):
        raise ValueError("Independent specimens require exactly one observation per unit")
    sources = []
    paths = {}
    for name, value in (("source", m.path), ("popmap", m.popmapfile)):
        if value is None:
            paths[name] = None
            continue
        source = resolve_input(value, base_dir)
        before = fingerprint_file(source)
        frozen, original = stage_input(source, out / name)
        if before.sha256 != original["sha256"]:
            raise ValueError("Input changed during genomic snapshot")
        paths[name] = frozen.relative_to(out).as_posix()
        sources.append(dict(kind=name, snapshot=paths[name], source=original))
    selected = replace(m, path=paths["source"], popmapfile=paths["popmap"])
    frozen = replace(dataset, modalities=(selected,))
    write_json(frozen.to_dict(), out / "dataset.json")
    write_json(sources, out / "sources.json")
    loaded = ingest_genomic(
        frozen, base_dir=out, output_dir=out / "ingestion", modality_name=m.name
    )
    values, encoding = encode_diploid(
        loaded, output_prefix=out / "encoding", random_seed=config.encoding_seed
    )
    np.save(out / "dosages.npy", values, allow_pickle=False)
    write_json(encoding, out / "encoding.json")
    if encoding.get("exclusions") or not np.isfinite(values).all():
        raise ValueError(
            "This opt-in workflow requires complete supported diploid biallelic calls; "
            "see encoding.json"
        )
    values = validate_dosages(values, config)
    source_hash = sources[0]["source"]["sha256"]
    keys = tuple(f"{source_hash}:{j}" for j in range(values.shape[1]))
    # Retain source allele strings and encoded states together, rather than
    # suggesting that a file-relative locus key establishes cross-file homology.
    alleles = np.asarray(loaded.genotype_data.snp_data, dtype=str)
    np.save(out / "source-alleles.npy", alleles, allow_pickle=False)
    cohort = dict(
        sample_ids=list(loaded.sample_ids),
        unit_ids=list(loaded.specimen_ids),
        locus_keys=list(keys),
        original_locus_indices=list(range(values.shape[1])),
        coding=(
            "SNPio encoding retained with raw allele states; exact source/order/coding "
            "required for projection"
        ),
        independence="assumed_not_verified",
        populations_used_as_predictors=False,
    )
    if len(set(cohort["unit_ids"])) != len(values):
        raise ValueError("SNPio cohort contains repeated biological units")
    write_json(cohort, out / "cohort.json")
    verify_genomic_sources(out, sources)
    return values, cohort, sources


def verify_genomic_sources(directory, sources):
    for source in sources:
        if (
            fingerprint_file(Path(directory) / source["snapshot"]).sha256
            != source["source"]["sha256"]
        ):
            raise ValueError("Frozen genomic input changed")
