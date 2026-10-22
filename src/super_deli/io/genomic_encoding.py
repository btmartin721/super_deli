"""Conservative diploid SNPio encoding, retaining original locus coordinates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from super_deli.io.genomic import LoadedGenomicData, _reader_logging

_ALLELES = {
    "A": "AA",
    "C": "CC",
    "G": "GG",
    "T": "TT",
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "K": "GT",
    "M": "AC",
}
_MISSING = {"N", "-", ".", "?"}


def _vcf_checks(loaded: LoadedGenomicData) -> tuple[dict[int, str], np.ndarray]:
    """Check original GT ploidy and SNP alleles before IUPAC loses information."""
    import pysam

    partial = np.zeros(loaded.missing_mask.shape, dtype=bool)
    exclusions = {}
    # SNPio may create compressed data and index sidecars in this directory.
    # Its public filename identifies the actual active VCF, not a directory entry.
    staged = Path(loaded.genotype_data.filename)
    count = 0
    with pysam.VariantFile(str(staged)) as variants:
        if tuple(variants.header.samples) != loaded.sample_ids:
            raise ValueError("VCF sample order changed during ingestion")
        for j, record in enumerate(variants):
            count += 1
            if j >= partial.shape[1]:
                raise ValueError("VCF records and SNPio loci do not correspond one-to-one")
            alleles = record.alleles or ()
            if len(alleles) > 2 or any(a not in "ACGT" or len(a) != 1 for a in alleles):
                exclusions[j] = "non_biallelic_snp_metadata"
            for i, sample in enumerate(loaded.sample_ids):
                gt = record.samples[sample].get("GT")
                if gt is not None and any(a is not None for a in gt):
                    if len(gt) != 2:
                        raise ValueError(f"Non-diploid called VCF GT at locus {j}, sample {sample}")
                    partial[i, j] = any(a is None for a in gt)
    if count != partial.shape[1]:
        raise ValueError("VCF records and SNPio loci do not correspond one-to-one")
    return exclusions, partial


def encode_diploid(
    loaded: LoadedGenomicData, *, output_prefix: Path, random_seed: int
) -> tuple[np.ndarray, dict]:
    """Use SNPio's encoder on a copy; exclude unsupported whole loci explicitly.

    Locus indices are never compacted, so block maps address the original source.
    Non-VCF inputs require a truthful diploidy declaration: IUPAC cannot reveal
    whether a homozygous call originally represented one or two allele copies.
    """
    from snpio import GenotypeEncoder

    raw = np.char.upper(np.asarray(loaded.genotype_data.snp_data).astype(str))
    exclusions: dict[int, str] = {}
    partial = np.zeros(raw.shape, dtype=bool)
    if loaded.summary["format"] == "vcf":
        exclusions, partial = _vcf_checks(loaded)
    for j in range(raw.shape[1]):
        tokens = set(raw[:, j]) - _MISSING
        if tokens - _ALLELES.keys():
            exclusions[j] = "unsupported_diploid_iupac"
        elif len(set("".join(_ALLELES[t] for t in tokens))) > 2:
            exclusions[j] = "more_than_two_observed_alleles"
    with _reader_logging():
        copied = loaded.genotype_data.copy()
        copied.prefix = str(output_prefix)
        calls = raw.copy()
        if exclusions:
            calls[:, list(exclusions)] = "N"
        copied.snp_data = calls
        # Invalid VCF metadata must not reach the encoder even for masked loci.
        if loaded.summary["format"] == "vcf" and exclusions:
            ref, alt = list(copied.ref), list(copied.alt)
            for j in exclusions:
                ref[j], alt[j] = "A", ["C"]
            copied.ref, copied.alt = ref, alt
        encoder = GenotypeEncoder(copied)
        encoder.rng = np.random.default_rng(random_seed)
        dosages = np.asarray(encoder.genotypes_012, dtype=float)
    if dosages.shape != raw.shape:
        raise ValueError("SNPio encoding changed locus or sample dimensions")
    expected_missing = np.isin(calls, list(_MISSING)) | partial
    if np.any((dosages == -9) & ~expected_missing):
        raise ValueError("SNPio could not encode an eligible observed diploid call")
    dosages[expected_missing | (dosages == -9)] = np.nan
    if not np.isin(dosages[~np.isnan(dosages)], [0, 1, 2]).all():
        raise ValueError("SNPio returned unsupported dosage values")
    return dosages, {
        "method": "SNPio GenotypeEncoder.genotypes_012 on an isolated copy",
        "encoding_seed": random_seed,
        "locus_index_basis": "zero-based original SNPio source order",
        "exclusions": [
            {"locus_index": j, "reason": reason} for j, reason in sorted(exclusions.items())
        ],
        "partially_missing_vcf_calls": int(partial.sum()),
        "missing_code": "NaN",
        "ploidy_validation": "called_VCF_GT"
        if loaded.summary["format"] == "vcf"
        else "user_declaration",
    }
