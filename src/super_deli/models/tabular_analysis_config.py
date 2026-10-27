"""Dependency-free orchestration contracts for prespecified tabular model arms."""

from __future__ import annotations

from dataclasses import dataclass

from super_deli.models.analysis_contracts import validate_arm_name
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig


@dataclass(frozen=True)
class TabularAnalysisArm:
    name: str
    model: TabularClusterConfig | HDBSCANConfig

    def __post_init__(self):
        validate_arm_name(self.name)
        if type(self.model) not in (TabularClusterConfig, HDBSCANConfig):
            raise ValueError("Unsupported tabular comparison model")
        type(self.model).from_dict(self.model.to_dict())
        if not self.model.feature_names:
            raise ValueError("Explicit ordered feature_names are required for comparison")
        if isinstance(self.model, TabularClusterConfig) and (
            self.model.scaling != "standard"
            or self.model.missing_policy != "error"
            or self.model.count_transform != "identity"
            or self.model.n_threads != 1
        ):
            raise ValueError("Comparison requires standard scaling, complete traits and one thread")

    @property
    def family(self):
        return "gmm" if isinstance(self.model, TabularClusterConfig) else "hdbscan"

    def to_dict(self):
        return {"name": self.name, "family": self.family, "model": self.model.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"name", "family", "model"}:
            raise ValueError("Arm requires exactly name, family and model")
        families = {"gmm": TabularClusterConfig, "hdbscan": HDBSCANConfig}
        if not isinstance(value["family"], str) or value["family"] not in families:
            raise ValueError("Unsupported tabular comparison family")
        return cls(value["name"], families[value["family"]].from_dict(value["model"]))


@dataclass(frozen=True)
class TabularAnalysisConfig:
    arms: tuple[TabularAnalysisArm, ...]
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported tabular analysis schema")
        if not isinstance(self.arms, tuple) or not 1 <= len(self.arms) <= 16:
            raise ValueError("Declare one to sixteen prespecified model arms")
        for arm in self.arms:
            if type(arm) is not TabularAnalysisArm:
                raise ValueError("Invalid analysis arm")
            TabularAnalysisArm.from_dict(arm.to_dict())
        if len({a.name for a in self.arms}) != len(self.arms):
            raise ValueError("Duplicate arm name")
        signatures = {(a.model.feature_names, a.model.n_pcs, a.model.rank_tol) for a in self.arms}
        if len(signatures) != 1:
            raise ValueError("All arms require identical ordered features, n_pcs and rank_tol")

    def to_dict(self):
        return {"schema_version": self.schema_version, "arms": [a.to_dict() for a in self.arms]}

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, dict) and value.get("schema_version") == 2:
            from super_deli.models.tabular_missing_config import TabularMissingAnalysisConfig

            return TabularMissingAnalysisConfig.from_dict(value)
        if not isinstance(value, dict) or set(value) != {"schema_version", "arms"}:
            raise ValueError("Analysis requires exactly schema_version and arms")
        if not isinstance(value["arms"], (list, tuple)):
            raise ValueError("arms must be an array")
        return cls(
            tuple(TabularAnalysisArm.from_dict(a) for a in value["arms"]), value["schema_version"]
        )
