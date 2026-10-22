"""Raw genotype MLEs and simulation under an explicitly unlinked HWE null."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import xlogy

from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json


@dataclass(frozen=True)
class GenomicNull:
    values: np.ndarray
    frequencies: np.ndarray
    called_counts: np.ndarray
    layout: dict

    def model_arrays(self):
        return {
            "values": self.values,
            "frequencies": self.frequencies,
            "called_counts": self.called_counts,
            "missing_mask": np.isnan(self.values),
        }


def fit_null(dataset, loaded, dosages, encoding, config):
    """Fit one allele frequency per original locus, before filtering or imputation.

    Entirely missing or unsupported loci are rejected. Observed monomorphic
    frequencies remain at their exact MLE boundaries; no pseudo-count is used.
    """
    if len(dataset.modalities) != 1 or dataset.modalities[0].name != config.modality:
        raise ValueError("Select a genomic-only dataset with the configured modality name")
    m = dataset.modalities[0]
    if m.kind != "genomic" or m.level != "specimen":
        raise ValueError("Use individual diploid genomic data; grouped calibration is unsupported")
    if encoding["exclusions"]:
        raise ValueError("The genomic null requires exclusively supported biallelic SNP loci")
    values = np.array(dosages, dtype=float, copy=True)
    n, p = values.shape
    if 96 * n * p + 32 * n**2 > config.max_working_bytes:
        raise ValueError("Genomic null dimensions exceed max_working_bytes")
    if n <= config.model.n_pcs or n < 2 * config.model.min_cluster_size:
        raise ValueError("Too few independent specimens for the requested calibration")
    if len(set(loaded.specimen_ids)) != n:
        raise ValueError("Repeated specimens cannot be independent calibration units")
    if any(any(c.isspace() for c in name) for name in loaded.sample_ids):
        raise ValueError("Canonical PHYLIP draws require whitespace-free sample IDs")
    if not np.isin(values[~np.isnan(values)], [0, 1, 2]).all():
        raise ValueError("Only called diploid 0/1/2 dosages and NaN are supported")
    counts = np.sum(~np.isnan(values), axis=0)
    if np.any(counts == 0):
        raise ValueError("An entirely missing locus has no identifiable allele frequency")
    if config.model.missing_policy == "error" and np.isnan(values).any():
        raise ValueError("Missing calls require the declared mean-imputation policy")
    frequencies = np.nansum(values, axis=0) / (2 * counts)
    called = ~np.isnan(values)
    g = np.nan_to_num(values)
    log_likelihood = np.where(
        called, (g == 1) * np.log(2) + xlogy(g, frequencies) + xlogy(2 - g, 1 - frequencies), 0
    ).sum()
    layout = {
        "schema_version": 1,
        "modality": m.name,
        "sample_ids": list(loaded.sample_ids),
        "unit_ids": list(loaded.specimen_ids),
        "locus_indices": list(range(p)),
        "locus_identity": "zero-based SNPio source order, retained in canonical draws",
        "encoding": encoding,
        "fit": {
            "estimator": "pooled_hwe_binomial_mle_v1",
            "frequency_definition": "sum called dosage / (2 * called specimens), by original locus",
            "negative_log_likelihood": float(-log_likelihood),
            "boundary_loci": np.flatnonzero((frequencies == 0) | (frequencies == 1)).tolist(),
            "missing_calls": int(np.isnan(values).sum()),
            "ascertainment": "fixed panel; sample-dependent variant/MAF ascertainment is unmodeled",
            "conditioning": "sample IDs, source locus order and non-informative missing mask",
        },
    }
    return GenomicNull(values, frequencies, counts, layout)


def simulate_null(null, seed):
    values = (
        np.random.default_rng(seed)
        .binomial(2, null.frequencies, size=null.values.shape)
        .astype(float)
    )
    values[np.isnan(null.values)] = np.nan
    return values


def write_raw_genotypes(values, layout, directory):
    """Write a new canonical PHYLIP source and preserve source-to-specimen links."""
    values = np.asarray(values, dtype=float)
    samples, units = layout["sample_ids"], layout["unit_ids"]
    if values.shape != (len(samples), len(layout["locus_indices"])) or len(units) != len(samples):
        raise ValueError("Genotypes and source-identity layout disagree")
    if not np.isin(values[~np.isnan(values)], [0, 1, 2]).all():
        raise ValueError("Cannot write unsupported diploid dosages")
    if len(set(samples)) != len(samples) or any(
        not s or any(c.isspace() for c in s) for s in samples
    ):
        raise ValueError("Canonical PHYLIP sample IDs must be unique and whitespace-free")
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    with (directory / "genotypes.phy").open("x", encoding="utf-8") as handle:
        handle.write(f"{len(samples)} {values.shape[1]}\n")
        for sample, row in zip(samples, values, strict=True):
            calls = "".join("N" if np.isnan(g) else "ARG"[int(g)] for g in row)
            handle.write(f"{sample} {calls}\n")
    declaration = config_from_dict(
        {
            "schema_version": 1,
            "dataset_id": "genomic_null_raw",
            "specimens": [{"specimen_id": u} for u in units],
            "modalities": [
                {
                    "name": layout["modality"],
                    "kind": "genomic",
                    "level": "specimen",
                    "format": "phylip",
                    "path": "genotypes.phy",
                    "observations": [
                        {"observation_id": s, "unit_id": u}
                        for s, u in zip(samples, units, strict=True)
                    ],
                }
            ],
        }
    )
    write_json(declaration.to_dict(), directory / "dataset.json")
    return declaration
