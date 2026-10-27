"""Versioned grouped orchestration; all scientific choices remain native contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from super_deli.models.analysis_contracts import validate_arm_name
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_config import PredictiveScreenConfig
from super_deli.models.predictive_fold_size import PredictiveFoldSizePolicy


@dataclass(frozen=True)
class GroupedAnalysisArm:
    name: str
    kind: str
    model: IntegrativeConfig
    screen: PredictiveScreenConfig | None = None
    bootstrap: PredictiveBootstrapConfig | None = None
    fold_size_policy: PredictiveFoldSizePolicy | None = None

    def __post_init__(self):
        validate_arm_name(self.name)
        if type(self.model) is not IntegrativeConfig:
            raise ValueError("An explicit native integrative model is required")
        model = IntegrativeConfig.from_dict(self.model.to_dict())
        if (
            model.target_level not in {"population", "locality"}
            or model.missing_views != "error"
            or model.assignment_threshold != 0
            or model.n_threads != 1
            or tuple(v.kind for v in model.views) != ("morphological", "ecological")
        ):
            raise ValueError(
                "Grouped analysis requires two ordered complete views and zero threshold"
            )
        for view in model.views:
            morphology = view.kind == "morphological"
            if (
                not view.feature_names
                or view.missing_policy != "error"
                or view.count_transform != "identity"
                or view.tabular_config().scaling != "standard"
                or view.aggregation != ("mean" if morphology else "none")
                or view.pca_fit != ("target_means" if morphology else "source_rows")
            ):
                raise ValueError(
                    "Use explicit features, standard scaling and group-mean morphology PCA"
                )
        if self.kind == "ordinary":
            if any(v is not None for v in (self.screen, self.bootstrap, self.fold_size_policy)):
                raise ValueError("Ordinary arms use native model stability only")
        elif self.kind == "predictive":
            if type(self.screen) is not PredictiveScreenConfig:
                raise ValueError("Predictive arms require an explicit screen configuration")
            PredictiveScreenConfig.from_dict(self.screen.to_dict()).validate_model(model)
            if self.bootstrap is not None:
                if type(self.bootstrap) is not PredictiveBootstrapConfig:
                    raise ValueError("Invalid predictive bootstrap configuration")
                PredictiveBootstrapConfig.from_dict(self.bootstrap.to_dict()).validate_model(
                    model, self.screen
                )
            if self.fold_size_policy is not None:
                if type(self.fold_size_policy) is not PredictiveFoldSizePolicy:
                    raise ValueError("Invalid fold-size policy")
                PredictiveFoldSizePolicy.from_dict(self.fold_size_policy.to_dict()).validate_model(
                    model, self.screen
                )
        else:
            raise ValueError("Arm kind must be ordinary or predictive")

    def signature(self):
        views = []
        for v in self.model.views:
            item = asdict(v)
            item["scaling"] = v.tabular_config().scaling
            views.append(item)
        return json.dumps(
            {
                "target": self.model.target_level,
                "views": views,
                "n_pcs": self.model.n_pcs,
                "rank_tol": self.model.rank_tol,
                "missing_views": self.model.missing_views,
            },
            sort_keys=True,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.model.to_dict(),
            "screen": self.screen.to_dict() if self.screen is not None else None,
            "bootstrap": self.bootstrap.to_dict() if self.bootstrap is not None else None,
            "fold_size_policy": self.fold_size_policy.to_dict()
            if self.fold_size_policy is not None
            else None,
        }

    @classmethod
    def from_dict(cls, value):
        allowed = {"name", "kind", "model", "screen", "bootstrap", "fold_size_policy"}
        if (
            not isinstance(value, dict)
            or not {"name", "kind", "model"} <= value.keys()
            or value.keys() - allowed
        ):
            raise ValueError("Invalid grouped analysis arm fields")
        extras = {}
        for key, model in (
            ("screen", PredictiveScreenConfig),
            ("bootstrap", PredictiveBootstrapConfig),
            ("fold_size_policy", PredictiveFoldSizePolicy),
        ):
            extras[key] = model.from_dict(value[key]) if value.get(key) is not None else None
        return cls(
            value["name"], value["kind"], IntegrativeConfig.from_dict(value["model"]), **extras
        )


@dataclass(frozen=True)
class GroupedAnalysisConfig:
    arms: tuple[GroupedAnalysisArm, ...]
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported grouped analysis schema")
        if not isinstance(self.arms, tuple) or not 1 <= len(self.arms) <= 16:
            raise ValueError("Declare one to sixteen prespecified grouped arms")
        for arm in self.arms:
            if type(arm) is not GroupedAnalysisArm:
                raise ValueError("Invalid grouped arm")
            GroupedAnalysisArm.from_dict(arm.to_dict())
        if len({a.name for a in self.arms}) != len(self.arms):
            raise ValueError("Duplicate arm name")
        if len({a.signature() for a in self.arms}) != 1:
            raise ValueError("Grouped arms require identical targets, features and preprocessing")

    def to_dict(self):
        return {"schema_version": self.schema_version, "arms": [a.to_dict() for a in self.arms]}

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "arms"}
            or not isinstance(value["arms"], (list, tuple))
        ):
            raise ValueError("Grouped analysis requires exactly schema_version and an arms array")
        return cls(
            tuple(GroupedAnalysisArm.from_dict(a) for a in value["arms"]), value["schema_version"]
        )
