"""Prespecified diagnostic contrasts; production configuration is untouched."""

from dataclasses import dataclass, replace

from super_deli.models.recovery_config import COHORTS as COHORTS
from super_deli.models.recovery_config import RecoveryConfig

ARMS = ("baseline", "pcs_2", "pcs_3", "starts_100", "shared_4", "source_4")
PARTITIONS = ("four", "axis_a", "axis_b")


@dataclass(frozen=True)
class FullGroupedDiagnosticConfig(RecoveryConfig):
    random_seed: int = 20261112


def variants(model):
    if (
        model.covariance_type != "full"
        or model.n_bootstrap
        or model.n_pcs != 4
        or model.max_k != 4
        or model.min_cluster_size != 5
        or model.n_init != 10
        or model.target_level not in {"population", "locality"}
        or tuple(v.kind for v in model.views) != ("morphological", "ecological")
        or tuple(v.aggregation for v in model.views) != ("mean", "none")
        or tuple(v.pca_fit for v in model.views) != ("target_means", "source_rows")
        or any(v.n_pcs != 2 or v.weight != 1 for v in model.views)
    ):
        raise ValueError("Full grouped diagnosis requires the declared group-mean four-PC baseline")
    return {
        "baseline": model,
        "pcs_2": replace(model, n_pcs=2),
        "pcs_3": replace(model, n_pcs=3),
        "starts_100": replace(model, n_init=100),
        "shared_4": replace(model, covariance_type="tied"),
        "source_4": replace(
            model, views=(replace(model.views[0], pca_fit="source_rows"), model.views[1])
        ),
    }
