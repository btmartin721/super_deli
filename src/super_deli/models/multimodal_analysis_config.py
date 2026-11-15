"""Versioned, dependency-free declarations for matched fixed-landmark comparisons."""

from dataclasses import dataclass, replace

from super_deli.models.analysis_contracts import validate_arm_name
from super_deli.models.config import BaselineConfig
from super_deli.models.genomic_analysis_config import TYPES, GenomicAnalysisArm
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig

MODEL_TYPES = {**TYPES, "shape": ShapeClusterConfig}


@dataclass(frozen=True)
class MultimodalAnalysisArm:
    name: str
    family: str
    modalities: tuple[str, ...]
    model: BaselineConfig | TabularClusterConfig | ShapeClusterConfig | IntegrativeConfig
    block_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        validate_arm_name(self.name)
        if (
            not isinstance(self.family, str)
            or self.family not in MODEL_TYPES
            or type(self.model) is not MODEL_TYPES[self.family]
        ):
            raise ValueError("Declare a supported native PCA/GMM family and configuration")
        m = MODEL_TYPES[self.family].from_dict(self.model.to_dict())
        if self.family in {"genomic", "tabular"}:
            GenomicAnalysisArm(self.name, self.family, self.modalities, m, self.block_ids)
            return
        if (
            not isinstance(self.modalities, tuple)
            or not self.modalities
            or any(not isinstance(n, str) or not n or n != n.strip() for n in self.modalities)
            or len(set(self.modalities)) != len(self.modalities)
        ):
            raise ValueError("Declare unique unpadded modality names")
        if self.block_ids is not None or m.assignment_threshold != 0 or m.n_threads != 1:
            raise ValueError(
                "Use native zero-threshold, single-thread fits; only genomic arms accept blocks"
            )
        if self.family == "shape":
            if len(self.modalities) != 1 or m.shape.missing_policy != "error":
                raise ValueError("Shape singleton requires one complete fixed-landmark source")
        else:
            if (
                self.modalities != tuple(v.modality for v in m.views)
                or m.target_level != "specimen"
                or m.missing_views != "error"
                or m.selection_criterion != "bic"
            ):
                raise ValueError("Integration requires complete matched specimen views and BIC")
            for v in m.views:
                if (
                    v.aggregation != "none"
                    or v.missing_policy != "error"
                    or v.pca_fit != "source_rows"
                    or v.count_transform != "identity"
                    or (v.kind == "morphometric" and v.shape.missing_policy != "error")
                ):
                    raise ValueError("Use complete unaggregated raw specimen views")

    def to_dict(self):
        return dict(
            name=self.name,
            family=self.family,
            modalities=list(self.modalities),
            model=self.model.to_dict(),
            block_ids=None if self.block_ids is None else list(self.block_ids),
        )

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or set(value) != {"name", "family", "modalities", "model", "block_ids"}
            or not isinstance(value["family"], str)
            or value["family"] not in MODEL_TYPES
            or not isinstance(value["modalities"], (list, tuple))
            or (
                value["block_ids"] is not None and not isinstance(value["block_ids"], (list, tuple))
            )
        ):
            raise ValueError("Malformed multimodal arm")
        return cls(
            value["name"],
            value["family"],
            tuple(value["modalities"]),
            MODEL_TYPES[value["family"]].from_dict(value["model"]),
            None if value["block_ids"] is None else tuple(value["block_ids"]),
        )


def _check_view(view, arm):
    model = arm.model
    if view.n_pcs != model.n_pcs:
        raise ValueError("Singleton and combined view PCA dimensions must match")
    if view.kind == "morphometric":
        if (
            arm.family != "shape"
            or replace(view.shape, n_components=view.n_pcs) != model.shape_config
        ):
            raise ValueError(
                "Shape view requires the same declared GPA/PCA policies as its singleton"
            )
    elif view.kind == "genomic":
        if (
            arm.family != "genomic"
            or view.ploidy != model.ploidy
            or (view.scaling or "patterson") != model.scaling
        ):
            raise ValueError("Genomic view requires matching ploidy and preprocessing")
    elif (
        arm.family != "tabular"
        or view.feature_names != model.feature_names
        or (view.scaling or "standard") != model.scaling
    ):
        raise ValueError("Tabular view requires matching continuous feature order and scaling")


@dataclass(frozen=True)
class MultimodalAnalysisConfig:
    arms: tuple[MultimodalAnalysisArm, ...]
    ecology_measurement_scope: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported multimodal analysis schema")
        if not isinstance(self.arms, tuple) or not 3 <= len(self.arms) <= 8:
            raise ValueError("Declare three to eight prespecified singleton/combined arms")
        for arm in self.arms:
            if type(arm) is not MultimodalAnalysisArm:
                raise ValueError("Invalid multimodal arm")
            MultimodalAnalysisArm.from_dict(arm.to_dict())
        if len({a.name for a in self.arms}) != len(self.arms):
            raise ValueError("Duplicate arm names")
        singletons = [a for a in self.arms if a.family != "integrative"]
        if (
            not 2 <= len(singletons) <= 4
            or sum(a.family == "shape" for a in singletons) != 1
            or len({a.modalities[0] for a in singletons}) != len(singletons)
        ):
            raise ValueError("Declare shape and one to three other singleton views")
        by_name = {a.modalities[0]: a for a in singletons}
        combined = [a for a in self.arms if a.family == "integrative"]
        if not any(set(a.modalities) == set(by_name) for a in combined):
            raise ValueError("Declare an all-view combined arm")
        kinds = {}
        for arm in combined:
            for view in arm.model.views:
                if view.modality not in by_name:
                    raise ValueError("Every combined view requires a singleton reference arm")
                if view.modality in kinds and kinds[view.modality] != view.kind:
                    raise ValueError("Conflicting view kinds")
                kinds[view.modality] = view.kind
                _check_view(view, by_name[view.modality])
        if len(set(kinds.values())) != len(kinds):
            raise ValueError("Use at most one source of each modality kind")
        if "ecological" in kinds.values() and self.ecology_measurement_scope != "individual":
            raise ValueError("Explicit individual-level ecological measurements are required")
        if self.ecology_measurement_scope not in (None, "individual"):
            raise ValueError("Unsupported ecology scope")

    def to_dict(self):
        return dict(
            schema_version=self.schema_version,
            ecology_measurement_scope=self.ecology_measurement_scope,
            arms=[a.to_dict() for a in self.arms],
        )

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "ecology_measurement_scope", "arms"}
            or not isinstance(value["arms"], (list, tuple))
        ):
            raise ValueError("Malformed multimodal analysis configuration")
        return cls(
            tuple(MultimodalAnalysisArm.from_dict(a) for a in value["arms"]),
            value["ecology_measurement_scope"],
            value["schema_version"],
        )
