"""Dependency-free assumptions for clustering fixed-landmark shapes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from super_deli.models.cluster_config import ClusterConfig
from super_deli.models.shape_config import ShapeConfig


@dataclass(frozen=True)
class ShapeClusterConfig(ClusterConfig):
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    schema_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.shape, ShapeConfig):
            raise ValueError("shape must be a ShapeConfig")
        if self.shape.repeated_policy != "error":
            raise ValueError("Clustering requires repeated_policy='error'; aggregate units first")
        if self.shape.n_components not in (None, self.n_pcs):
            raise ValueError("shape.n_components must be null or equal n_pcs")

    @property
    def shape_config(self) -> ShapeConfig:
        return replace(self.shape, n_components=self.n_pcs)

    @classmethod
    def from_dict(cls, value: object) -> ShapeClusterConfig:
        if isinstance(value, dict) and "shape" in value:
            value = {**value, "shape": ShapeConfig.from_dict(value["shape"])}
        return super().from_dict(value)
