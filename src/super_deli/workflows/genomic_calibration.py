"""Approximate fitted HWE calibration with raw SNPio pipeline refits."""

from functools import partial
from importlib import import_module

from super_deli.core.provenance import fingerprint_file
from super_deli.models.genomic_config import GenomicCalibrationConfig
from super_deli.workflows.calibration_artifacts import fit_pipeline
from super_deli.workflows.empirical_calibration import run_fitted_calibration

NAMESPACE = "genomic_pooled_hwe_v1"
TARGET = "pooled_unlinked_hwe_genotypes"
TAIL_OPTIONS = {
    "p_value_kind": "Approximate plug-in pooled-HWE parametric bootstrap with plus-one rank",
    "decision_key": "reject_pooled_hwe",
    "interval_scope": (
        "Binomial fitted-null tail interval conditional on observed T, fitted frequencies and "
        "missing mask; excludes frequency-estimation, ascertainment and model uncertainty; "
        "not species confidence; failed-outcome bounds are not confidence intervals"
    ),
}
SCOPE = (
    "Approximate global departure test against pooled, unlinked diploid HWE with fitted allele "
    "frequencies. Raw genotypes are regenerated under a fixed panel and fixed non-informative "
    "missing mask. Every draw reruns SNPio ingestion/encoding, locus filtering, imputation, "
    "Patterson scaling, PCA and the complete diagonal-GMM K search. Requires unrelated "
    "independent specimens; does not preserve LD, relatedness, continuous population structure "
    "or sample-dependent ascertainment. No exact-size or species-boundary guarantee, "
    "cluster-specific significance, joint genomic-phenotypic null or stability bootstrap."
)


def _dependencies():
    try:
        import numpy as np

        import_module("sklearn")
        from super_deli.io.genomic import _dependencies as snpio_dependencies
        from super_deli.models.genomic_null import fit_null, simulate_null, write_raw_genotypes
        from super_deli.models.monte_carlo import summarize_upper_tail
        from super_deli.workflows.genomic_baseline import run_genomic_baseline

        snpio_dependencies()
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Install 'super-deli[genomics,baseline]' for genomic calibration"
        ) from exc
    return (
        np,
        fit_null,
        simulate_null,
        write_raw_genotypes,
        summarize_upper_tail,
        run_genomic_baseline,
    )


def _load_null(dataset, config, source, destination, fitter):
    from super_deli.io.genomic import ingest_genomic
    from super_deli.io.genomic_encoding import encode_diploid

    if len(dataset.modalities) != 1:
        raise ValueError("Genomic fitted-null calibration requires a genomic-only dataset")
    loaded = ingest_genomic(
        dataset,
        base_dir=source,
        output_dir=destination / "null_ingestion",
        modality_name=config.modality,
    )
    values, encoding = encode_diploid(
        loaded, output_prefix=destination / "null_encoding", random_seed=config.model.random_seed
    )
    return fitter(dataset, loaded, values, encoding, config)


def point_record(run, directory, model):
    import numpy as np

    return {
        "point_seed": int(
            np.random.SeedSequence(model.random_seed).spawn(1)[0].generate_state(1)[0]
        ),
        "n_retained": len(run["specimen_ids"]),
        "mixture_sha256": fingerprint_file(directory / "mixture.npz").sha256,
        "dosages_sha256": fingerprint_file(directory / "dosages.npy").sha256,
        "preprocessing_sha256": fingerprint_file(directory / "preprocessing.npz").sha256,
    }


def run_genomic_calibration(dataset, *, config, base_dir, output_dir):
    config = GenomicCalibrationConfig.from_dict(config.to_dict())
    return run_fitted_calibration(
        dataset,
        config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        dependencies=_dependencies(),
        scope=SCOPE,
        tail_options=TAIL_OPTIONS,
        seed_namespace=NAMESPACE,
        target=TARGET,
        null_hypothesis=(
            "Independent unrelated diploid specimens; unlinked Binomial(2, p_j) "
            "loci with pooled unknown p_j"
        ),
        load_null=_load_null,
        point_fitter=partial(
            fit_pipeline, record_for=point_record, error_stage="genomic_point_pipeline"
        ),
        evidence_method="Approximate full-pipeline fitted pooled-HWE BIC-improvement bootstrap",
        software_names=("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl", "snpio"),
    )
