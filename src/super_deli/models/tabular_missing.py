"""Auditable per-training-fit adapter around existing PCA and clustering engines."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from super_deli.io.config import write_json


class TrainingContext:
    """One native arm owns a context; each PCA call is recorded, including failures."""

    def __init__(self, policy):
        if policy not in ("complete_case", "mean"):
            raise ValueError("Unknown missing-data policy")
        self.policy = policy
        self.records = []
        self.arrays = {}

    def declaration(self):
        return dict(
            schema_version=2,
            policy=self.policy,
            imputation="training_observed_mean" if self.policy == "mean" else "none",
            all_missing_training_feature="fail",
            conditional_on_original_mask=True,
            missing_value_uncertainty_integrated=False,
            p_value=None,
        )

    def density_config(self, config):
        return replace(
            config.preprocessing_config(),
            missing_policy="mean" if self.policy == "mean" else "error",
        )

    def check_budget(self, x, fits, budget):
        # Raw/mask, PCA and mixture arrays plus a conservative metadata allowance.
        estimate = fits * 8 * (12 * x.size + 32 * len(x) + 128 * x.shape[1])
        if estimate > budget:
            raise ValueError("Retained per-fit preprocessing exceeds max_working_bytes")

    def fit_pca(self, x, names, types, config):
        from super_deli.models.tabular_gmm import fit_tabular_pca

        i = len(self.records)
        prefix = f"fit_{i}_"
        self.arrays[prefix + "raw"] = np.array(x, copy=True)
        self.arrays[prefix + "missing"] = np.isnan(x)
        record = dict(
            fit=i,
            training_rows=len(x),
            features=list(names),
            missing_cells=int(np.isnan(x).sum()),
            preprocessing_status="failed",
        )
        self.records.append(record)
        try:
            if np.isnan(x).all(axis=0).any():
                raise ValueError("A selected training feature has no observed values")
            if np.isnan(x).all(axis=1).any():
                raise ValueError("A training unit has no observed selected values")
            if config.missing_policy != ("mean" if self.policy == "mean" else "error"):
                raise ValueError("Effective preprocessing policy mismatch")
            model, scores = fit_tabular_pca(x, names, types, config)
            for k, v in dict(
                kept_indices=model.kept_indices,
                means=model.means,
                scales=model.scales,
                pca_mean=model.pca.mean_,
                components=model.pca.components_,
                scores=scores,
            ).items():
                self.arrays[prefix + k] = np.array(v, copy=True)
            record.update(
                preprocessing_status="complete",
                imputed_cells=model.imputed_cells,
                excluded_features=list(model.exclusions),
            )
            return model, scores
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def capture_mixture(self, fitted):
        mixture = fitted.mixture
        estimator = mixture.mixture
        prefix = f"fit_{len(self.records) - 1}_"
        for key, value in dict(
            weights=estimator.weights_[list(mixture.component_order)],
            mixture_means=estimator.means_[list(mixture.component_order)],
            covariances=estimator.covariances_[list(mixture.component_order)],
            labels=mixture.labels,
            membership=mixture.membership,
        ).items():
            self.arrays[prefix + key] = np.array(value, copy=True)
        self.records[-1].update(
            candidates=mixture.candidates,
            selected_k=mixture.selected_k,
            component_order=list(mixture.component_order),
        )

    def capture_density(self, point):
        prefix = f"fit_{len(self.records) - 1}_"
        self.arrays[prefix + "labels"] = point.labels.copy()
        self.arrays[prefix + "strength"] = point.strength.copy()
        self.records[-1]["selected_k"] = point.selected_k

    def save(self, directory):
        root = Path(directory)
        if (root / "training-context.json").exists():
            return
        write_json(
            dict(declaration=self.declaration(), fits=self.records), root / "training-context.json"
        )
        with (root / "training-context.npz").open("xb") as handle:
            np.savez_compressed(handle, **self.arrays)
