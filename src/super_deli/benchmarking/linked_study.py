"""Independent empirical block-null studies, with byte-paired legacy observations."""

from dataclasses import replace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.empirical_study import run_fitted_study
from super_deli.benchmarking.genomic_study import simulate_genomic
from super_deli.benchmarking.pipeline_study import summarize_records
from super_deli.models.linked_config import CASES, PAIRED_CASES, LinkedStudyConfig


def simulate_linked(case, n, seed):
    if case in PAIRED_CASES:
        return simulate_genomic(case, n, seed)
    if case not in CASES or type(n) is not int or n < 20 or n % 4:
        raise ValueError("Use a supported linked case and >=20 specimens divisible by four")
    import numpy as np

    rng = np.random.default_rng(seed)
    if case == "moderate_ld":
        frequencies = rng.uniform(0.15, 0.5, 128)
        haplotypes = np.empty((n, 2, 128), dtype=int)
        for start in range(0, 128, 8):
            haplotypes[:, :, start] = rng.binomial(1, frequencies[start], size=(n, 2))
            for j in range(start + 1, start + 8):
                copy = rng.random((n, 2)) < 0.75
                new = rng.binomial(1, frequencies[j], size=(n, 2))
                haplotypes[:, :, j] = np.where(copy, haplotypes[:, :, j - 1], new)
        values = haplotypes.sum(axis=1).astype(float)
    else:
        frequencies = rng.uniform(0.15, 0.5, 8)
        values = np.repeat(rng.binomial(2, frequencies, size=(n, 8)), 16, axis=1).astype(float)
    layout = {
        "schema_version": 1,
        "modality": "genomic",
        "sample_ids": [f"sample_{i:04d}" for i in range(n)],
        "unit_ids": [f"unit_{i:04d}" for i in range(n)],
        "locus_indices": list(range(128)),
    }
    return values, layout, np.zeros(n, dtype=int)


def case_policy(study, case, repetition):
    if case not in CASES or not 0 <= repetition < study.n_datasets:
        raise ValueError("Unknown linked validation case/repetition")
    return replace(
        study.calibration,
        random_seed=derived_seed(study.random_seed, case, repetition, "null_stream"),
    )


def summarize_linked(records, study, case):
    interpretation = (
        "Model-law rejection rate for independent declared genomic blocks"
        if case in {"common_hwe", "linked_stress", "moderate_ld"}
        else "Power for the constructed four-group alternative"
        if case == "four_groups"
        else "Misspecification sensitivity; NOT species-boundary or discrete-cluster type-I error"
    )
    return summarize_records(records, study, case, interpretation=interpretation)


def run_linked_study(study, *, case, output_dir):
    from super_deli.models.linked_null import write_raw_linked
    from super_deli.workflows.linked_calibration import run_linked_calibration

    study = LinkedStudyConfig.from_dict(study.to_dict())
    if case not in CASES:
        raise ValueError("Unknown linked validation case")
    return run_fitted_study(
        study,
        case=case,
        output_dir=output_dir,
        simulator=simulate_linked,
        writer=write_raw_linked,
        runner=run_linked_calibration,
        policy_for=case_policy,
        summarizer=summarize_linked,
        decision_key="reject_independent_blocks",
    )
