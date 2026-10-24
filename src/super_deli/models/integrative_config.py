"""Dependency-free declarations for common-unit, weighted score integration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

from super_deli.models.cluster_config import ClusterConfig
from super_deli.models.config import BaselineConfig
from super_deli.models.shape_config import ShapeConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig


def configuration_dict(value):
    """Preserve v1 full-covariance configuration bytes at every nesting level."""
    return asdict(
        value,
        dict_factory=lambda pairs: {
            key: item
            for key, item in pairs
            if not (
                (key == "covariance_type" and item == "full")
                or (key == "pca_fit" and item == "source_rows")
                or (key == "selection_criterion" and item == "bic")
            )
        },
    )


@dataclass(frozen=True)
class ViewConfig:
    modality: str
    kind: str
    n_pcs: int = 2
    weight: float = 1.0
    aggregation: str = "none"
    feature_names: tuple[str, ...] = ()
    missing_policy: str = "error"
    scaling: str | None = None
    count_transform: str = "identity"
    ploidy: int | None = None
    shape: ShapeConfig | None = None
    pca_fit: str = "source_rows"

    def __post_init__(self):
        if (
            not isinstance(self.modality, str)
            or not self.modality
            or self.modality != self.modality.strip()
        ):
            raise ValueError("View modality must be a nonempty, unpadded identifier")
        if not isinstance(self.kind, str) or self.kind not in {
            "genomic",
            "morphological",
            "ecological",
            "morphometric",
        }:
            raise ValueError("Unsupported integrative view kind")
        if type(self.n_pcs) is not int or self.n_pcs < 1:
            raise ValueError("View n_pcs must be a positive integer")
        try:
            finite_weight = type(self.weight) in (int, float) and math.isfinite(float(self.weight))
        except OverflowError:
            finite_weight = False
        if not finite_weight or self.weight <= 0:
            raise ValueError("View weight must be positive and finite")
        if not isinstance(self.aggregation, str) or self.aggregation not in {"none", "mean"}:
            raise ValueError("View aggregation must be none or mean")
        if (
            not isinstance(self.feature_names, tuple)
            or any(not isinstance(n, str) or not n or n != n.strip() for n in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("feature_names must contain unique, unpadded names")
        if not isinstance(self.pca_fit, str) or self.pca_fit not in {"source_rows", "target_means"}:
            raise ValueError("pca_fit must be source_rows or target_means")
        if self.pca_fit == "target_means" and (
            self.kind != "morphological"
            or self.aggregation != "mean"
            or self.missing_policy != "error"
            or self.count_transform != "identity"
        ):
            raise ValueError(
                "Target-mean PCA requires complete continuous morphology and mean aggregation"
            )
        if self.kind in {"morphological", "ecological"}:
            if self.ploidy is not None or self.shape is not None:
                raise ValueError("Tabular views cannot contain genomic/shape options")
            self.tabular_config()
        elif self.kind == "genomic":
            if self.feature_names or self.count_transform != "identity" or self.shape is not None:
                raise ValueError("Genomic views cannot contain tabular/shape options")
            self.genomic_config()
        else:
            if (
                self.feature_names
                or self.ploidy is not None
                or self.scaling is not None
                or self.missing_policy != "error"
                or self.count_transform != "identity"
            ):
                raise ValueError("Morphometric policies belong in the shape object")
            if not isinstance(self.shape, ShapeConfig):
                raise ValueError("Morphometric views require an explicit shape configuration")
            if self.shape.repeated_policy != "error" or self.shape.n_components not in (
                None,
                self.n_pcs,
            ):
                raise ValueError(
                    "Shape repeats must be rejected and n_components must match view n_pcs"
                )

    def tabular_config(self, max_working_bytes=268435456):
        return TabularClusterConfig(
            True,
            n_pcs=self.n_pcs,
            min_cluster_size=self.n_pcs + 1,
            n_bootstrap=0,
            feature_names=self.feature_names,
            missing_policy=self.missing_policy,
            scaling="standard" if self.scaling is None else self.scaling,
            count_transform=self.count_transform,
            max_working_bytes=max_working_bytes,
        )

    def genomic_config(self):
        return BaselineConfig(
            self.ploidy,
            self.missing_policy,
            True,
            n_pcs=self.n_pcs,
            min_cluster_size=max(2, self.n_pcs + 1),
            n_bootstrap=0,
            scaling="patterson" if self.scaling is None else self.scaling,
        )

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("View must be an object containing only supported view fields")
        if not {"modality", "kind"} <= value.keys():
            raise ValueError("Each view requires modality and kind")
        options = dict(value)
        if "feature_names" in options:
            if not isinstance(options["feature_names"], (tuple, list)):
                raise ValueError("feature_names must be an array")
            options["feature_names"] = tuple(options["feature_names"])
        if options.get("shape") is not None:
            options["shape"] = ShapeConfig.from_dict(options["shape"])
        return cls(**options)


@dataclass(frozen=True)
class IntegrativeConfig(ClusterConfig):
    views: tuple[ViewConfig, ...] = ()
    target_level: str = "specimen"
    missing_views: str = "error"
    rank_tol: float = 1e-12
    max_working_bytes: int = 268435456
    schema_version: int = 1
    covariance_type: str = "full"
    selection_criterion: str = "bic"

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.covariance_type, str) or self.covariance_type not in {
            "full",
            "tied",
        }:
            raise ValueError("Integrative covariance_type must be full or tied")
        if (
            not isinstance(self.views, tuple)
            or not 2 <= len(self.views) <= 4
            or any(not isinstance(v, ViewConfig) for v in self.views)
        ):
            raise ValueError("Integration requires two to four declared views")
        if len({v.modality for v in self.views}) != len(self.views) or len(
            {v.kind for v in self.views}
        ) != len(self.views):
            raise ValueError("Select unique modalities and at most one view of each kind")
        if not isinstance(self.target_level, str) or self.target_level not in {
            "specimen",
            "population",
            "locality",
        }:
            raise ValueError("target_level must be specimen, population or locality")
        if not isinstance(self.missing_views, str) or self.missing_views not in {
            "error",
            "complete_cases",
        }:
            raise ValueError("missing_views must be error or complete_cases")
        if (
            type(self.rank_tol) not in (int, float)
            or not 0 < self.rank_tol < 1
            or not math.isfinite(self.rank_tol)
        ):
            raise ValueError("rank_tol must be finite and between zero and one")
        if type(self.max_working_bytes) is not int or self.max_working_bytes < 1:
            raise ValueError("max_working_bytes must be positive")
        if any(v.pca_fit == "target_means" for v in self.views) and self.target_level not in {
            "population",
            "locality",
        }:
            raise ValueError("Target-mean PCA requires explicit population/locality integration")
        if not isinstance(self.selection_criterion, str) or self.selection_criterion not in {
            "bic",
            "aic",
        }:
            raise ValueError("selection_criterion must be bic or aic")
        if self.selection_criterion == "aic" and (
            self.target_level not in {"population", "locality"}
            or {v.kind for v in self.views} != {"morphological", "ecological"}
            or any(
                v.pca_fit != ("target_means" if v.kind == "morphological" else "source_rows")
                or v.aggregation != ("mean" if v.kind == "morphological" else "none")
                or v.missing_policy != "error"
                or v.count_transform != "identity"
                for v in self.views
            )
        ):
            raise ValueError("AIC requires grouped continuous views with group-mean morphology PCA")
        if (
            self.selection_criterion == "aic"
            and self.covariance_type == "tied"
            and self.n_bootstrap
        ):
            raise ValueError("Shared-family AIC stability is not validated")
        if self.n_pcs > sum(v.n_pcs for v in self.views):
            raise ValueError("Joint n_pcs exceeds the declared score dimensions")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or "target_level" not in value or "views" not in value:
            raise ValueError("Integrative configuration requires views and target_level")
        if not isinstance(value["views"], (list, tuple)):
            raise ValueError("views must be an array")
        value = {**value, "views": tuple(ViewConfig.from_dict(v) for v in value["views"])}
        return super().from_dict(value)

    def to_dict(self):
        return configuration_dict(self)


def require_calibrated_pca(model):
    """Keep new grouped preprocessing outside unvalidated significance workflows."""
    if model.selection_criterion != "bic":
        raise ValueError("AIC requires the separate grouped likelihood calibration workflow")
    if any(v.pca_fit != "source_rows" for v in model.views):
        raise ValueError("Target-mean PCA significance calibration is not yet supported")
