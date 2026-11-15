"""Explicit full-coordinate preset over unchanged native integrative estimators."""

from dataclasses import replace

from super_deli.models.integrative_config import IntegrativeConfig


def full_retention_model(model, *, min_cluster_size):
    """Retain every declared view PC; require an explicitly compatible size policy.

    Full-SVD joint PCA still centers and rotates the complete score space.
    This preset changes neither covariance family nor numerical fitting settings.
    """
    if type(model) is not IntegrativeConfig:
        raise ValueError("A native IntegrativeConfig is required")
    return replace(
        model,
        n_pcs=sum(v.n_pcs for v in model.views),
        min_cluster_size=min_cluster_size,
        fusion_policy="retain_all_v1",
    )


def full_retention_analysis(config, *, arm_name, min_cluster_size):
    """Replace exactly one named integrative arm in a matched multimodal plan."""
    from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig

    if type(config) is not MultimodalAnalysisConfig:
        raise ValueError("A MultimodalAnalysisConfig is required")
    chosen = [a for a in config.arms if a.name == arm_name]
    if len(chosen) != 1 or chosen[0].family != "integrative":
        raise ValueError("Select exactly one declared integrative arm by name")
    return replace(
        config,
        arms=tuple(
            replace(a, model=full_retention_model(a.model, min_cluster_size=min_cluster_size))
            if a.name == arm_name
            else a
            for a in config.arms
        ),
    )
