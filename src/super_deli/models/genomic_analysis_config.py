"""Dependency-free contract for prespecified native matched PCA/GMM comparisons."""

from dataclasses import dataclass

from super_deli.models.analysis_contracts import validate_arm_name
from super_deli.models.config import BaselineConfig
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig

TYPES = dict(genomic=BaselineConfig, tabular=TabularClusterConfig, integrative=IntegrativeConfig)


@dataclass(frozen=True)
class GenomicAnalysisArm:
    name: str
    family: str
    modalities: tuple[str, ...]
    model: BaselineConfig | TabularClusterConfig | IntegrativeConfig
    block_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        validate_arm_name(self.name)
        if (
            not isinstance(self.family, str)
            or self.family not in TYPES
            or type(self.model) is not TYPES[self.family]
        ):
            raise ValueError("Use the declared native genomic, tabular or integrative model")
        m = TYPES[self.family].from_dict(self.model.to_dict())
        if (
            not isinstance(self.modalities, tuple)
            or not self.modalities
            or any(not isinstance(n, str) or not n or n != n.strip() for n in self.modalities)
            or len(set(self.modalities)) != len(self.modalities)
        ):
            raise ValueError("Declare unique unpadded modality names")
        if m.assignment_threshold != 0 or m.n_threads != 1:
            raise ValueError("Comparison requires zero membership threshold and one native thread")
        if self.family == "integrative":
            if (
                self.modalities != tuple(v.modality for v in m.views)
                or m.target_level != "specimen"
                or m.missing_views != "error"
                or m.selection_criterion != "bic"
            ):
                raise ValueError("Integration requires complete matched specimen views and BIC")
            for v in m.views:
                if (
                    v.kind not in {"genomic", "morphological", "ecological"}
                    or v.aggregation != "none"
                    or v.missing_policy != "error"
                    or v.pca_fit != "source_rows"
                    or v.count_transform != "identity"
                ):
                    raise ValueError(
                        "Only complete unaggregated genomic/continuous views are supported"
                    )
        else:
            if len(self.modalities) != 1 or m.missing_policy != "error":
                raise ValueError("Native singleton arms require complete observations")
            if self.family == "tabular" and (
                not m.feature_names or m.count_transform != "identity"
            ):
                raise ValueError("Declare continuous features without count transformation")
        if self.block_ids is not None and (
            self.family != "genomic"
            or not isinstance(self.block_ids, tuple)
            or not self.block_ids
            or any(not isinstance(b, str) or not b or b != b.strip() for b in self.block_ids)
        ):
            raise ValueError("Only the genomic arm accepts original-source-locus block IDs")
        if self.family == "genomic" and m.n_bootstrap and not (self.block_ids or m.assume_unlinked):
            raise ValueError(
                "Genomic stability requires declared blocks or explicit unlinked assumption"
            )

    def to_dict(self):
        return dict(
            name=self.name,
            family=self.family,
            modalities=list(self.modalities),
            model=self.model.to_dict(),
            block_ids=list(self.block_ids) if self.block_ids else None,
        )

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or set(value) != {"name", "family", "modalities", "model", "block_ids"}
            or not isinstance(value["modalities"], (list, tuple))
            or not isinstance(value["family"], str)
            or value["family"] not in TYPES
            or (
                value["block_ids"] is not None and not isinstance(value["block_ids"], (list, tuple))
            )
        ):
            raise ValueError("Malformed genomic comparison arm")
        return cls(
            value["name"],
            value["family"],
            tuple(value["modalities"]),
            TYPES[value["family"]].from_dict(value["model"]),
            None if value["block_ids"] is None else tuple(value["block_ids"]),
        )


@dataclass(frozen=True)
class GenomicAnalysisConfig:
    arms: tuple[GenomicAnalysisArm, ...]
    ecology_measurement_scope: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported genomic comparison schema")
        if not isinstance(self.arms, tuple) or not 3 <= len(self.arms) <= 5:
            raise ValueError("Declare three to five prespecified singleton/combined arms")
        for arm in self.arms:
            if type(arm) is not GenomicAnalysisArm:
                raise ValueError("Invalid comparison arm")
            GenomicAnalysisArm.from_dict(arm.to_dict())
        if len({a.name for a in self.arms}) != len(self.arms):
            raise ValueError("Duplicate arm names")
        singletons = [a for a in self.arms if a.family != "integrative"]
        g = [a for a in singletons if a.family == "genomic"]
        if (
            len(g) != 1
            or not 2 <= len(singletons) <= 3
            or len({a.modalities[0] for a in singletons}) != len(singletons)
        ):
            raise ValueError("Declare one genomic and one/two distinct phenotype singleton arms")
        by_name = {a.modalities[0]: a for a in singletons}
        names = set(by_name)
        combined = [a for a in self.arms if a.family == "integrative"]
        if not any(set(a.modalities) == names for a in combined):
            raise ValueError("Declare an all-view integrative arm")
        kinds = {}
        for arm in combined:
            if not set(arm.modalities) <= names:
                raise ValueError("Every integrated view requires its singleton reference arm")
            for view in arm.model.views:
                if view.modality in kinds and kinds[view.modality] != view.kind:
                    raise ValueError("Conflicting view kinds")
                kinds[view.modality] = view.kind
                single = by_name[view.modality]
                m = single.model
                if (
                    view.n_pcs != m.n_pcs
                    or view.missing_policy != m.missing_policy
                    or (view.scaling or ("patterson" if view.kind == "genomic" else "standard"))
                    != m.scaling
                ):
                    raise ValueError(
                        "Native and integrated views require identical PCA preprocessing d"
                        "eclarations"
                    )
                if view.kind == "genomic":
                    if single.family != "genomic" or view.ploidy != m.ploidy:
                        raise ValueError("Genomic kind/ploidy mismatch")
                elif single.family != "tabular" or view.feature_names != m.feature_names:
                    raise ValueError("Phenotype feature identity/order mismatch")
        if len(set(kinds.values())) != len(kinds) or kinds.get(g[0].modalities[0]) != "genomic":
            raise ValueError("Use at most one view of each declared kind")
        if "ecological" in kinds.values() and self.ecology_measurement_scope != "individual":
            raise ValueError("Explicit individually measured ecology is required")
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
            or not isinstance(value["arms"], (tuple, list))
        ):
            raise ValueError("Malformed genomic comparison configuration")
        return cls(
            tuple(GenomicAnalysisArm.from_dict(a) for a in value["arms"]),
            value["ecology_measurement_scope"],
            value["schema_version"],
        )
