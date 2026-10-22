"""Regenerate a small synthetic linked-SNP example; never run during analysis."""

from pathlib import Path

import numpy as np

from super_deli.io.config import write_json


def generate(destination: Path) -> None:
    """Write to a new directory, keeping generating labels separate from features."""
    destination.mkdir(exist_ok=False)
    rng = np.random.default_rng(1729)
    n, blocks, repeats = 48, 32, 3
    labels = np.repeat([0, 1], n // 2)
    # Conditional on group, block genotypes are independent; within each block
    # three perfectly linked copies make naive locus resampling inappropriate.
    frequencies = np.where(labels[:, None] == 0, 0.12, 0.88)
    dosages = np.repeat(rng.binomial(2, np.broadcast_to(frequencies, (n, blocks))), repeats, axis=1)
    missing = rng.random(dosages.shape) < 0.015
    calls = np.asarray(["A", "M", "C"])[dosages]
    calls[missing] = "N"
    samples = [f"sample_{i:02d}" for i in range(n)]
    specimens = [f"specimen_{i:02d}" for i in range(n)]
    text = f"{n} {blocks * repeats}\n" + "".join(
        f"{sample} {''.join(row)}\n" for sample, row in zip(samples, calls, strict=True)
    )
    (destination / "simulated.phy").write_text(text, encoding="utf-8")
    write_json(
        {
            "schema_version": 1,
            "dataset_id": "synthetic_linked_two_populations",
            "random_seed": 1729,
            "specimens": [{"specimen_id": s} for s in specimens],
            "modalities": [
                {
                    "name": "snps",
                    "kind": "genomic",
                    "level": "specimen",
                    "format": "phylip",
                    "path": "simulated.phy",
                    "popmapfile": None,
                    "observations": [
                        {"observation_id": s, "unit_id": u}
                        for s, u in zip(samples, specimens, strict=True)
                    ],
                }
            ],
        },
        destination / "dataset.json",
    )
    write_json(
        {
            "schema_version": 1,
            "ploidy": 2,
            "missing_policy": "mean",
            "assume_independent_specimens": True,
            "n_pcs": 2,
            "max_k": 3,
            "min_cluster_size": 4,
            "scaling": "patterson",
            "n_bootstrap": 20,
            "assume_unlinked": False,
            "n_init": 5,
            "random_seed": 1729,
        },
        destination / "model.json",
    )
    write_json(
        {
            "schema_version": 1,
            "block_ids": [f"block_{b:02d}" for b in range(blocks) for _ in range(repeats)],
        },
        destination / "blocks.json",
    )
    write_json(
        {
            "schema_version": 1,
            "seed": 1729,
            "kind": "synthetic_generating_populations",
            "species_truth": "not established",
            "independent_blocks": blocks,
            "linked_copies_per_block": repeats,
            "population_allele_frequencies": [0.12, 0.88],
            "missing_probability": 0.015,
            "numpy_version": np.__version__,
            "labels": dict(zip(specimens, labels.tolist(), strict=True)),
        },
        destination / "truth.json",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New output directory")
    generate(parser.parse_args().destination)
