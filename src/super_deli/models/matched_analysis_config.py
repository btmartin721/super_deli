"""Separate, dependency-free orchestration contract for native matched GMM arms."""

from dataclasses import dataclass

from super_deli.models.analysis_contracts import validate_arm_name
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.separate_gmm_config import SeparateGMMConfig


@dataclass(frozen=True)
class MatchedAnalysisArm:
    name: str
    family: str
    model: IntegrativeConfig | SeparateGMMConfig

    def __post_init__(self):
        validate_arm_name(self.name)
        if self.family == "pca_gmm" and type(self.model) is IntegrativeConfig:
            m = IntegrativeConfig.from_dict(self.model.to_dict())
            if (
                m.target_level != "specimen"
                or m.missing_views != "error"
                or m.covariance_type != "tied"
                or m.selection_criterion != "bic"
                or m.assignment_threshold != 0
                or m.n_threads != 1
                or tuple(v.kind for v in m.views) != ("morphological", "ecological")
            ):
                raise ValueError("PCA arms require complete specimen-level shared-covariance BIC")
            for v in m.views:
                if (
                    not v.feature_names
                    or v.aggregation != "none"
                    or v.pca_fit != "source_rows"
                    or v.missing_policy != "error"
                    or v.count_transform != "identity"
                    or v.tabular_config().scaling != "standard"
                ):
                    raise ValueError(
                        "Explicit continuous features and native standard PCA required"
                    )
        elif self.family == "separate_gmm" and type(self.model) is SeparateGMMConfig:
            SeparateGMMConfig.from_dict(self.model.to_dict())
        else:
            raise ValueError("Use pca_gmm/IntegrativeConfig or separate_gmm/SeparateGMMConfig")
        # Reuse existing identity, independence, feature and memory constraints.
        self.alignment_config()

    def alignment_config(self):
        if self.family == "separate_gmm":
            return self.model
        a, b = self.model.views
        return SeparateGMMConfig(
            a.modality,
            b.modality,
            a.feature_names,
            b.feature_names,
            self.model.assume_independent_units,
            "individual",
            max_working_bytes=self.model.max_working_bytes,
            max_pairwise_bytes=self.model.max_pairwise_bytes,
        )

    def signature(self):
        m = self.alignment_config()
        return (m.morphology, m.ecology, m.morphology_features, m.ecology_features)

    def to_dict(self):
        return dict(name=self.name, family=self.family, model=self.model.to_dict())

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"name", "family", "model"}:
            raise ValueError("Matched arm requires exactly name, family and model")
        family = value["family"]
        if family not in ("pca_gmm", "separate_gmm"):
            raise ValueError("Unsupported matched analysis family")
        kind = IntegrativeConfig if family == "pca_gmm" else SeparateGMMConfig
        return cls(value["name"], family, kind.from_dict(value["model"]))


@dataclass(frozen=True)
class MatchedAnalysisConfig:
    arms: tuple[MatchedAnalysisArm, ...]
    ecology_measurement_scope: str
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported matched analysis schema")
        if self.ecology_measurement_scope != "individual":
            raise ValueError("Explicit individually measured ecology is required")
        if not isinstance(self.arms, tuple) or not 1 <= len(self.arms) <= 16:
            raise ValueError("Declare one to sixteen prespecified matched arms")
        for arm in self.arms:
            if type(arm) is not MatchedAnalysisArm:
                raise ValueError("Invalid matched analysis arm")
            MatchedAnalysisArm.from_dict(arm.to_dict())
        if len({a.name for a in self.arms}) != len(self.arms):
            raise ValueError("Duplicate matched arm names")
        if len({a.signature() for a in self.arms}) != 1:
            raise ValueError("Arms require identical views and ordered selected features")

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
            or not isinstance(value["arms"], (tuple, list))
        ):
            raise ValueError(
                "Matched analysis requires schema_version, ecology_measurement_scope and arms"
            )
        return cls(
            tuple(MatchedAnalysisArm.from_dict(a) for a in value["arms"]),
            value["ecology_measurement_scope"],
            value["schema_version"],
        )
