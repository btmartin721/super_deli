"""Raw joint block-profile / multivariate Gaussian-regression likelihood and draws."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.models.empirical_null import collect_raw_views, write_raw_views
from super_deli.models.linked_null import fit_null as fit_blocks
from super_deli.models.linked_null import simulate_null as draw_blocks
from super_deli.models.linked_null import write_raw_linked


@dataclass(frozen=True)
class JointNull:
    values: np.ndarray
    layout: dict
    arrays: dict
    genomic: object

    def model_arrays(self):
        return {"values": self.values, **self.arrays}


def design(genotypes, block_ids, predictors):
    columns = [np.ones(len(genotypes))]
    for block in predictors:
        indices = [j for j, b in enumerate(block_ids) if b == block]
        columns.append(genotypes[:, indices].mean(axis=1))
    return np.column_stack(columns)


def covariance_factor(covariance):
    scales = np.sqrt(np.diag(covariance))
    return scales[:, None] * np.linalg.cholesky(covariance / scales[:, None] / scales[None, :])


def fit_null(dataset, loaded, dosages, encoding, tables, config):
    genomic_dataset = config_from_dict(
        {
            **dataset.to_dict(),
            "modalities": [
                next(
                    m
                    for m in dataset.to_dict()["modalities"]
                    if m["name"] == config.model.views[0].modality
                )
            ],
        }
    )
    genomic = fit_blocks(genomic_dataset, loaded, dosages, encoding, config.genomic_policy())
    continuous, layout = collect_raw_views(dataset, tables, config, views=config.model.views[1:])
    units = list(loaded.specimen_ids)
    if set(units) != {s.specimen_id for s in dataset.specimens}:
        raise ValueError("Genomics must cover every declared specimen exactly once")
    order = {u: i for i, u in enumerate(layout["unit_ids"])}
    y = continuous[[order[u] for u in units]]
    x = design(genomic.values, config.block_ids, config.predictor_blocks)
    n, p = y.shape
    if (
        n - x.shape[1] < p
        or 8 * (24 * n * (p + x.shape[1]) + 8 * p * p) > config.model.max_working_bytes
    ):
        raise ValueError("Insufficient residual degrees of freedom or workspace for joint fit")
    # Normalize only for rank assessment; coefficients retain raw dosage/phenotypic units.
    scales = np.linalg.norm(x, axis=0)
    if np.any(scales == 0):
        raise ValueError("Genomic predictor design is rank deficient")
    singular = np.linalg.svd(x / scales, compute_uv=False)
    if singular[-1] <= config.rank_tol * singular[0]:
        raise ValueError("Genomic predictor design is rank deficient")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    covariance = residuals.T @ residuals / n
    sd = np.sqrt(np.diag(covariance))
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("Conditional residual covariance is singular")
    eigen = np.linalg.eigvalsh(covariance / sd[:, None] / sd[None, :])
    if eigen[0] <= config.rank_tol * max(eigen[-1], 1.0):
        raise ValueError("Conditional residual covariance is singular; no regularization")
    sign, logdet = np.linalg.slogdet(covariance)
    if sign != 1 or not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite conditional Gaussian fit")
    gaussian_nll = 0.5 * n * (p * np.log(2 * np.pi) + logdet + p)
    return JointNull(
        np.column_stack((genomic.values, y)),
        {
            "schema_version": 1,
            "unit_ids": units,
            "genomic": genomic.layout,
            "views": layout["views"],
            "block_ids": list(config.block_ids),
            "predictor_blocks": list(config.predictor_blocks),
            "fit": {
                "estimator": "block_genotypes_conditional_gaussian_mle_v1",
                "genomic": genomic.layout["fit"],
                "gaussian_negative_log_likelihood": float(gaussian_nll),
                "negative_log_likelihood": float(
                    gaussian_nll + genomic.layout["fit"]["negative_log_likelihood"]
                ),
                "design_rank": x.shape[1],
                "residual_df": n - x.shape[1],
                "n_phenotypic_features": p,
                "residual_covariance_divisor": n,
                "predictors": "intercept and prespecified raw block-mean dosages",
                "direction": "conditional association, not causation",
            },
        },
        {
            **genomic.arrays,
            "design": x,
            "coefficients": coefficients,
            "residual_covariance": covariance,
        },
        genomic,
    )


def simulate_null(null, seed):
    genomic_seed, phenotype_seed = np.random.SeedSequence(seed).spawn(2)
    block = draw_blocks(null.genomic, genomic_seed)
    x = design(block["genotypes"], null.layout["block_ids"], null.layout["predictor_blocks"])
    noise = np.random.default_rng(phenotype_seed).standard_normal(
        (len(x), null.arrays["coefficients"].shape[1])
    )
    return {
        **block,
        "phenotypes": x @ null.arrays["coefficients"]
        + noise @ covariance_factor(null.arrays["residual_covariance"]).T,
    }


def write_raw_joint(values, layout, directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    genomic = write_raw_linked(
        values if "donor_indices" in values else values["genotypes"],
        layout["genomic"],
        directory / "genomic",
    )
    phenotypic = write_raw_views(
        values["phenotypes"],
        {"unit_ids": layout["unit_ids"], "views": layout["views"]},
        directory / "phenotypic",
    )
    declarations = []
    for prefix, dataset in (("genomic", genomic), ("phenotypic", phenotypic)):
        for view in dataset.to_dict()["modalities"]:
            declarations.append({**view, "path": prefix + "/" + view["path"]})
    dataset = config_from_dict(
        {
            **genomic.to_dict(),
            "dataset_id": "joint_genomic_phenotypic_raw",
            "modalities": declarations,
        }
    )
    write_json(dataset.to_dict(), directory / "dataset.json")
    return dataset
