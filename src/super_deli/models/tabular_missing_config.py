"""Separate opt-in contract; legacy complete-data schemas are unchanged."""

from dataclasses import dataclass

from super_deli.models.tabular_analysis_config import TabularAnalysisArm, TabularAnalysisConfig


@dataclass(frozen=True)
class TabularMissingAnalysisConfig:
    arms: tuple[TabularAnalysisArm, ...]
    policies: tuple[str, ...]
    schema_version: int = 2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("Missing-data analysis requires schema_version=2")
        TabularAnalysisConfig(self.arms)
        if (
            not isinstance(self.policies, tuple)
            or not 1 <= len(self.policies) <= 2
            or any(
                not isinstance(p, str) or p not in ("complete_case", "mean") for p in self.policies
            )
            or len(set(self.policies)) != len(self.policies)
        ):
            raise ValueError("Declare unique complete_case and/or mean policies")

    def to_dict(self):
        return dict(
            schema_version=2, policies=list(self.policies), arms=[a.to_dict() for a in self.arms]
        )

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"schema_version", "policies", "arms"}:
            raise ValueError("Missing-data analysis requires version, policies and arms")
        if not isinstance(value["policies"], list) or not isinstance(value["arms"], list):
            raise ValueError("Policies and arms must be arrays")
        return cls(
            tuple(TabularAnalysisArm.from_dict(a) for a in value["arms"]),
            tuple(value["policies"]),
            value["schema_version"],
        )
