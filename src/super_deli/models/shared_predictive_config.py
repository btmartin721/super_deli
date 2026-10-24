"""Paired fixed-family validation, without a family-selection procedure."""

from dataclasses import dataclass, replace

from super_deli.models.predictive_study_config import PredictiveStudyConfig


@dataclass(frozen=True)
class SharedPredictiveConfig:
    study: PredictiveStudyConfig
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported paired predictive schema")
        if (
            not isinstance(self.study, PredictiveStudyConfig)
            or self.study.model.covariance_type != "full"
        ):
            raise ValueError("Declare the unchanged full-family reference model")
        if any(c.law == "gaussian_scores" for c in self.study.cells):
            raise ValueError("Paired shared validation requires raw grouped views")
        if self.study.screening.n_folds != 5:
            raise ValueError("Use five prespecified whole-unit folds")

    def family(self, name):
        if name not in {"full", "tied"}:
            raise ValueError("Prespecify full or tied covariance")
        return replace(self.study, model=replace(self.study.model, covariance_type=name))

    def plan(self):
        return [{**r, "planned_fits": 2 * r["planned_fits"]} for r in self.study.plan()]

    def probe(self, seed=20261205):
        return replace(self, study=self.study.probe(seed))

    def to_dict(self):
        return {"schema_version": 1, "study": self.study.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"schema_version", "study"}:
            raise ValueError("Explicit paired study/schema required")
        return cls(PredictiveStudyConfig.from_dict(value["study"]), value["schema_version"])
