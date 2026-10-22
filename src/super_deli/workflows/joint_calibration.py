"""Approximate raw-data joint calibration through the shared integrative pipeline."""

from super_deli.models.joint_config import JointCalibrationConfig
from super_deli.workflows.empirical_calibration import run_fitted_calibration

NAMESPACE = "block_genotypes_conditional_gaussian_v1"
TARGET = "joint_block_genotypes_conditional_gaussian_phenotypes"
TAIL_OPTIONS = {
    "decision_key": "reject_joint_genomic_phenotypic",
    "p_value_kind": (
        "Approximate fitted joint block/Gaussian-regression bootstrap with plus-one tail"
    ),
    "interval_scope": (
        "Conditional fitted-null tail Monte Carlo interval; excludes profile, "
        "regression, block-map and model uncertainty; not species confidence"
    ),
}
SCOPE = (
    "Matched unrelated independent specimens; complete diploid biallelic genotypes and continuous "
    "individual morphology/ecology. Independent prespecified genotype blocks with empirical "
    "profile probabilities; phenotypes have a linear conditional mean in prespecified raw block "
    "means and a constant full Gaussian residual covariance. Fit before PCA. Not causal, not "
    "a general genomic-phenotypic dependence model; no missingness, grouped sampling, shapes, "
    "kinship/spatial or ascertainment correction. Approximate model-departure p-values; no general "
    "size/species/selected-K guarantee. Stability is separate; no replacement of failed draws."
)


def canonical_dosages(loaded, values, encoding):
    """Freeze lexicographic allele polarity before constructing raw predictors.

    SNPio may choose polarity from sample frequencies. The conditional mean is
    defined in a deterministic base orientation, not in data-dependent PCA axes.
    """
    import numpy as np

    from super_deli.io.genomic_encoding import _ALLELES

    raw = np.char.upper(np.asarray(loaded.genotype_data.snp_data).astype(str))
    values = values.copy()
    flips, alleles = [], []
    for j in range(raw.shape[1]):
        tokens = raw[:, j]
        observed = sorted(set("".join(_ALLELES.get(t, "") for t in tokens)))
        alleles.append(observed)
        if not 1 <= len(observed) <= 2:
            continue  # Existing raw validation rejects unsupported or missing loci.
        target = np.array(
            [_ALLELES[t].count(observed[-1]) if t in _ALLELES else np.nan for t in tokens],
            dtype=float,
        )
        if len(observed) == 1:
            target[~np.isnan(target)] = 0
        if np.array_equal(values[:, j], target, equal_nan=True):
            continue
        if np.array_equal(2 - values[:, j], target, equal_nan=True):
            values[:, j] = target
            flips.append(j)
        else:
            raise ValueError("SNPio dosage and deterministic allele polarity disagree")
    return values, {
        **encoding,
        "joint_predictor_coding": {
            "rule": "copies of lexicographically larger observed base; monomorphic calls zero",
            "source_alleles": alleles,
            "snpio_reversed_loci": flips,
        },
    }


def _load_null(dataset, config, source, destination, fitter):
    from super_deli.io.genomic import ingest_genomic
    from super_deli.io.genomic_encoding import encode_diploid
    from super_deli.io.tabular import ingest_tabular

    loaded = ingest_genomic(
        dataset,
        base_dir=source,
        output_dir=destination / "null_genomic_ingestion",
        modality_name=config.model.views[0].modality,
    )
    values, encoding = encode_diploid(
        loaded, output_prefix=destination / "null_encoding", random_seed=config.model.random_seed
    )
    values, encoding = canonical_dosages(loaded, values, encoding)
    tables = ingest_tabular(
        dataset,
        base_dir=source,
        output_dir=destination / "null_tabular_ingestion",
        modality_names=[v.modality for v in config.model.views[1:]],
    ).tables
    return fitter(dataset, loaded, values, encoding, tables, config)


def run_joint_calibration(dataset, *, config, base_dir, output_dir):
    import numpy as np

    from super_deli.io.genomic import _dependencies
    from super_deli.models.joint_null import fit_null, simulate_null, write_raw_joint
    from super_deli.models.monte_carlo import summarize_upper_tail
    from super_deli.workflows.integrative import run_integrative

    _dependencies()
    config = JointCalibrationConfig.from_dict(config.to_dict())
    return run_fitted_calibration(
        dataset,
        config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        dependencies=(
            np,
            fit_null,
            simulate_null,
            write_raw_joint,
            summarize_upper_tail,
            run_integrative,
        ),
        scope=SCOPE,
        tail_options=TAIL_OPTIONS,
        seed_namespace=NAMESPACE,
        target=TARGET,
        null_hypothesis=(
            "Independent genotype blocks and Gaussian phenotypes conditional on "
            "declared block means"
        ),
        load_null=_load_null,
        evidence_method="Approximate full integrative pipeline fitted joint-null BIC bootstrap",
        software_names=("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl", "snpio"),
    )
