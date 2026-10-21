"""Linkage-aware empirical fitted-null execution through shared raw-pipeline machinery."""

from functools import partial

from super_deli.models.linked_config import LinkedCalibrationConfig
from super_deli.workflows.calibration_artifacts import fit_pipeline
from super_deli.workflows.empirical_calibration import run_fitted_calibration
from super_deli.workflows.genomic_calibration import _load_null, point_record

NAMESPACE = "empirical_block_genotypes_v1"
TARGET = "independent_empirical_genotype_blocks"
TAIL_OPTIONS = {
    "decision_key": "reject_independent_blocks",
    "p_value_kind": "Approximate fitted empirical block-profile bootstrap with plus-one rank",
    "interval_scope": (
        "Binomial fitted-null tail interval conditional on observed T, block map "
        "and empirical profile probabilities; excludes support-"
        "estimation/block/model uncertainty; not species confidence; failed-"
        "outcome bounds are not confidence intervals"
    ),
}
SCOPE = (
    "Approximate global departure test of independent declared genomic blocks, fitted as "
    "categorical whole-genotype profiles before PCA. Complete diploid biallelic data and "
    "unrelated independent specimens; fixed panel and prespecified block map. Whole profiles "
    "are independently drawn with replacement per specimen/block. Arbitrary within-block "
    "associations are retained in the fitted distribution, without HWE or phase inference. "
    "No between-block LD, spatial/kinship model, ascertainment correction, unseen-profile "
    "smoothing, missing-data linkage model, exact-size guarantee or species-boundary inference. "
    "Every draw reruns SNPio, filtering/scaling, PCA and the full GMM search. "
    "Stability is separate."
)


def _dependencies(block_ids):
    from super_deli.models.linked_null import fit_null, simulate_null, write_raw_linked
    from super_deli.workflows.genomic_calibration import _dependencies as genomic_dependencies

    np, _, _, _, summarize, runner = genomic_dependencies()
    return (
        np,
        fit_null,
        simulate_null,
        write_raw_linked,
        summarize,
        partial(runner, block_ids=block_ids),
    )


def run_linked_calibration(dataset, *, config, base_dir, output_dir):
    config = LinkedCalibrationConfig.from_dict(config.to_dict())
    return run_fitted_calibration(
        dataset,
        config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        dependencies=_dependencies(config.block_ids),
        scope=SCOPE,
        tail_options=TAIL_OPTIONS,
        seed_namespace=NAMESPACE,
        target=TARGET,
        null_hypothesis=(
            "Independent unrelated specimen rows; independent declared blocks with "
            "unknown arbitrary genotype-profile probabilities"
        ),
        load_null=_load_null,
        point_fitter=partial(
            fit_pipeline, record_for=point_record, error_stage="linked_genomic_point_pipeline"
        ),
        evidence_method=(
            "Approximate full-pipeline empirical block-genotype BIC-improvement bootstrap"
        ),
        software_names=("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl", "snpio"),
    )
