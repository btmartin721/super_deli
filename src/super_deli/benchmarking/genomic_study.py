"""Truth-separated independent tests of a pooled genomic fitted null."""

from dataclasses import replace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.empirical_study import run_fitted_study
from super_deli.benchmarking.pipeline_study import summarize_records
from super_deli.models.genomic_config import CASES, GenomicStudyConfig


def simulate_genomic(case, n, seed):
    """Fixed 128-locus panels; stress cases deliberately violate the fitted law."""
    import numpy as np

    if case not in CASES or type(n) is not int or n < 20 or n % 4:
        raise ValueError("Use a supported genomic case and >=20 specimens divisible by four")
    rng = np.random.default_rng(seed)
    p = 128
    frequencies = rng.uniform(0.15, 0.5, p)
    labels = np.zeros(n, dtype=int)
    if case == "rare_missing_hwe":
        frequencies = rng.uniform(0.01, 0.1, p)
    if case == "four_groups":
        labels = rng.permutation(np.repeat(np.arange(4), n // 4))
        centers = np.column_stack((2 * (labels // 2) - 1, 2 * (labels % 2) - 1))
        logits = np.log(frequencies / (1 - frequencies))[None, :] + 2.5 * np.repeat(
            centers, p // 2, axis=1
        )
        probabilities = 1 / (1 + np.exp(-logits))
    elif case == "cline_stress":
        position = rng.uniform(-1, 1, (n, 1))
        logits = np.log(frequencies / (1 - frequencies))[None, :] + 3.0 * position * rng.choice(
            [-1, 1], p
        )
        probabilities = 1 / (1 + np.exp(-logits))
    else:
        probabilities = frequencies
    if case == "linked_stress":
        # Sixteen independent blocks, eight perfectly linked loci per block.
        # Every marginal locus is HWE; no discrete specimen groups are generated.
        blocks = rng.binomial(2, frequencies[::8], size=(n, p // 8))
        values = np.repeat(blocks, 8, axis=1).astype(float)
    else:
        values = rng.binomial(2, probabilities, size=(n, p)).astype(float)
    if case == "rare_missing_hwe":
        values[rng.random(values.shape) < 0.1] = np.nan
    units = [f"unit_{i:04d}" for i in range(n)]
    layout = {
        "schema_version": 1,
        "modality": "genomic",
        "sample_ids": [f"sample_{i:04d}" for i in range(n)],
        "unit_ids": units,
        "locus_indices": list(range(p)),
    }
    return values, layout, labels


def case_policy(study, case, repetition):
    if case not in CASES or not 0 <= repetition < study.n_datasets:
        raise ValueError("Unknown genomic validation case/repetition")
    return replace(
        study.calibration,
        random_seed=derived_seed(study.random_seed, case, repetition, "null_stream"),
    )


def summarize_genomic(records, study, case):
    interpretation = (
        "Type-I error under the specified unlinked pooled HWE law"
        if case.endswith("hwe")
        else "Power for the constructed four-group alternative"
        if case == "four_groups"
        else "Misspecification sensitivity; NOT species-boundary or discrete-cluster type-I error"
    )
    return summarize_records(records, study, case, interpretation=interpretation)


def run_genomic_study(study, *, case, output_dir):
    from super_deli.workflows.genomic_calibration import _dependencies, run_genomic_calibration

    study = GenomicStudyConfig.from_dict(study.to_dict())
    if case not in CASES or study.calibration.modality != "genomic":
        raise ValueError("Use a supported case and the genomic-named study modality")
    _, _, _, writer, _, _ = _dependencies()
    return run_fitted_study(
        study,
        case=case,
        output_dir=output_dir,
        simulator=simulate_genomic,
        writer=writer,
        runner=run_genomic_calibration,
        policy_for=case_policy,
        summarizer=summarize_genomic,
        decision_key="reject_pooled_hwe",
    )
