"""Paired validation of one prespecified fold-size policy, with no policy selection."""

from dataclasses import dataclass, field

from super_deli.models.empirical_config import _options
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy
from super_deli.models.predictive_config import PredictiveScreenConfig
from super_deli.models.predictive_fold_size import PredictiveFoldSizePolicy
from super_deli.models.predictive_study_config import PredictiveCell


@dataclass(frozen=True)
class ScaledFoldStudy(PredictiveBootstrapStudy):
    random_seed: int = 20261226
    max_artifact_bytes: int = 6 * 1024**3
    fold_size_policy: PredictiveFoldSizePolicy = field(default_factory=PredictiveFoldSizePolicy)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.fold_size_policy, PredictiveFoldSizePolicy):
            raise ValueError("Explicit fold-size policy required")
        self.fold_size_policy.validate_model(self.model, self.screening)

    @classmethod
    def from_dict(cls, value):
        d = _options(
            cls, value, {"schema_version", "model", "screening", "cells", "fold_size_policy"}
        )
        d["model"] = IntegrativeConfig.from_dict(d["model"])
        d["screening"] = PredictiveScreenConfig.from_dict(d["screening"])
        d["cells"] = tuple(PredictiveCell.from_dict(c) for c in d["cells"])
        d["fold_size_policy"] = PredictiveFoldSizePolicy.from_dict(d["fold_size_policy"])
        return cls(**d)

    def to_dict(self):
        return {**super().to_dict(), "fold_size_policy": self.fold_size_policy.to_dict()}

    def plan(self):
        return [dict(r, planned_fits=2 * r["planned_fits"]) for r in super().plan()]

    def probe(self, seed=20261227):
        return super().probe(seed)
