"""Run one prespecified complete-genotype comparison case into a new directory."""

import argparse
import json
from pathlib import Path

import numpy as np

from super_deli.benchmarking.genomic_neural_study import (
    audit_steps,
    audited_points,
    simulate,
    write_audit,
)
from super_deli.io.config import write_json
from super_deli.models.genomic_neural_config import GenomicComparisonConfig, GenomicNeuralConfig
from super_deli.models.genomic_null import write_raw_genotypes
from super_deli.workflows.genomic_neural import run_genomic_comparison


def run(case, directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    write_json(case, directory / "case.json")
    values, truth, reference = simulate(case["law"], case["n"], case["seed"])
    np.savez_compressed(directory / "generating-law.npz", values=values, truth=truth, **reference)
    ids = [f"s{i:04d}" for i in range(len(values))]
    layout = dict(
        sample_ids=ids, unit_ids=ids, locus_indices=list(range(values.shape[1])), modality="genomic"
    )
    dataset = write_raw_genotypes(values, layout, directory / "raw")
    config = GenomicComparisonConfig(
        GenomicNeuralConfig(
            2,
            "error",
            True,
            n_subsamples=case["n_subsamples"],
            subsampling_seed=case["subsampling_seed"],
        ),
        (0, 1, 2),
    )
    with audited_points(directory / "step-audits"):
        summary = run_genomic_comparison(
            dataset,
            analysis_config=config,
            base_dir=directory / "raw",
            output_dir=directory / "analysis",
        )
    encoded = np.load(directory / "analysis/inputs/dosages.npy", allow_pickle=False)
    for j in range(values.shape[1]):
        assert np.array_equal(values[:, j], encoded[:, j]) or np.array_equal(
            2 - values[:, j], encoded[:, j]
        )
    audit = write_audit(directory / "analysis", truth)
    steps = audit_steps(directory / "step-audits")
    assert steps["trainings"] == 3 * (1 + case["n_subsamples"])
    result = dict(status=summary["status"], audit=audit, steps=steps)
    write_json(result, directory / "completion.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.case.read_text()), args.output_dir)
