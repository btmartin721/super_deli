"""A product of fitted categorical whole-block genotype distributions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from super_deli.io.config import write_json
from super_deli.models.genomic_null import validate_raw_genotypes, write_raw_genotypes


@dataclass(frozen=True)
class LinkedNull:
    values: np.ndarray
    layout: dict
    arrays: dict

    def model_arrays(self):
        return {"values": self.values, **self.arrays}


def fit_null(dataset, loaded, dosages, encoding, config):
    values, _ = validate_raw_genotypes(dataset, loaded, dosages, encoding, config)
    n, p = values.shape
    if len(config.block_ids) != p:
        raise ValueError("Block IDs must cover every original source locus before exclusions")
    groups = {}
    for j, name in enumerate(config.block_ids):
        groups.setdefault(name, []).append(j)
    blocks, arrays, nll = [], {}, 0.0
    for b, (name, indices) in enumerate(groups.items()):
        profiles, inverse, counts = np.unique(
            values[:, indices].astype(np.int8), axis=0, return_inverse=True, return_counts=True
        )
        probabilities = counts / n
        nll -= float(counts @ np.log(probabilities))
        prefix = f"block_{b:04d}"
        arrays.update(
            {
                prefix + "_profiles": profiles,
                prefix + "_counts": counts,
                prefix + "_probabilities": probabilities,
                prefix + "_profile_ids": inverse,
            }
        )
        blocks.append(
            {
                "block_id": name,
                "locus_indices": indices,
                "n_profiles": len(counts),
                "singleton_profiles": int(np.sum(counts == 1)),
                "singleton_specimen_fraction": float(np.sum(counts == 1) / n),
                "maximum_profile_probability": float(probabilities.max()),
            }
        )
    layout = {
        "schema_version": 1,
        "modality": dataset.modalities[0].name,
        "sample_ids": list(loaded.sample_ids),
        "unit_ids": list(loaded.specimen_ids),
        "locus_indices": list(range(p)),
        "locus_identity": "zero-based original SNPio source order",
        "encoding": encoding,
        "block_ids": list(config.block_ids),
        "blocks": blocks,
        "fit": {
            "estimator": "empirical_block_genotype_mle_v1",
            "negative_log_likelihood": nll,
            "n_blocks": len(blocks),
            "n_observed_profiles": sum(b["n_profiles"] for b in blocks),
            "missing_calls": 0,
            "likelihood": "ordered iid rows; product of block categorical probabilities, count/N",
            "support": "observed profiles only, without smoothing or unseen-profile generation",
            "hypothesis": (
                "independent declared blocks with arbitrary within-block genotype laws; "
                "no HWE constraint"
            ),
        },
    }
    return LinkedNull(values, layout, arrays)


def simulate_null(null, seed):
    n, p = null.values.shape
    donors = np.random.default_rng(seed).integers(n, size=(n, len(null.layout["blocks"])))
    values = np.empty((n, p), dtype=float)
    for b, block in enumerate(null.layout["blocks"]):
        indices = block["locus_indices"]
        values[:, indices] = null.values[donors[:, b]][:, indices]
    return {"genotypes": values, "donor_indices": donors}


def write_raw_linked(values, layout, directory):
    directory = Path(directory)
    draw = isinstance(values, dict)
    declaration = write_raw_genotypes(values["genotypes"] if draw else values, layout, directory)
    if draw:
        with (directory / "donor_indices.npy").open("xb") as handle:
            np.save(handle, values["donor_indices"], allow_pickle=False)
        write_json(
            {
                "index_basis": "zero-based source sample order",
                "block_order": [b["block_id"] for b in layout["blocks"]],
            },
            directory / "donor_layout.json",
        )
    return declaration
