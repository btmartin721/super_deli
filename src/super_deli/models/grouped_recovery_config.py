"""Frozen grouped diagnostic settings; no calibration or production policy changes."""

from dataclasses import dataclass, replace

from super_deli.models.recovery_config import COHORTS as COHORTS
from super_deli.models.recovery_config import RecoveryConfig

VARIANTS = (
    "baseline",
    "pcs_3",
    "pcs_4",
    "ecology_0.5",
    "ecology_2",
    "ecology_4",
    "starts_100",
    "views_4_joint_2",
    "views_4_joint_4",
    "first_specimen",
)


@dataclass(frozen=True)
class GroupedRecoveryConfig(RecoveryConfig):
    random_seed: int = 20261104


def variants(model):
    if (
        model.covariance_type != "tied"
        or model.n_bootstrap
        or model.n_pcs != 2
        or model.max_k != 4
        or model.min_cluster_size != 5
        or model.n_init != 10
        or model.target_level not in {"population", "locality"}
        or tuple(v.kind for v in model.views) != ("morphological", "ecological")
        or tuple(v.aggregation for v in model.views) != ("mean", "none")
        or any(v.n_pcs != 2 or v.weight != 1 for v in model.views)
    ):
        raise ValueError("Grouped diagnosis requires the declared shared two-view baseline")
    result = {"baseline": model, "pcs_3": replace(model, n_pcs=3), "pcs_4": replace(model, n_pcs=4)}
    for weight in (0.5, 2, 4):
        result[f"ecology_{weight}"] = replace(
            model,
            views=tuple(
                replace(v, weight=weight) if v.kind == "ecological" else v for v in model.views
            ),
        )
    result["starts_100"] = replace(model, n_init=100)
    for pcs in (2, 4):
        result[f"views_4_joint_{pcs}"] = replace(
            model, views=tuple(replace(v, n_pcs=4) for v in model.views), n_pcs=pcs
        )
    result["first_specimen"] = model
    return result
