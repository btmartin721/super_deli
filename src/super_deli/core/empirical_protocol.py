"""Immutable, non-executable evaluation plans with explicit applicability gates."""

from dataclasses import asdict, dataclass

from super_deli.core.empirical import checked_fields
from super_deli.core.provenance import configuration_hash


@dataclass(frozen=True)
class EmpiricalProtocol:
    schema_version: int
    dataset_id: str
    source_sha256: str
    profile_sha256: str
    phase: str
    pipeline_reference_commit: str
    scaling: str
    missing_policy: str
    view_weights: str
    selection_criterion: str
    comparison_rule: str
    n_threads: int
    fitting_authorized: bool
    independent_units_confirmed: bool
    reference_independence_confirmed: bool
    cohort_rule: str
    target_level: str
    point_arms: tuple[str, ...]
    n_pcs_per_view: int
    joint_n_pcs: int
    max_k: int
    min_cluster_size: int
    n_init: int
    max_iter: int
    tol: float
    reg_covar: float
    random_seed: int
    significance_draws: int
    stability_draws: int
    reference_metrics: tuple[str, ...]

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported empirical evaluation protocol")
        if self.phase != "curation_only" or self.target_level != "specimen":
            raise ValueError("Only matched capture-record curation is authorized")
        for name in (
            "fitting_authorized",
            "independent_units_confirmed",
            "reference_independence_confirmed",
        ):
            if type(getattr(self, name)) is not bool or getattr(self, name):
                raise ValueError("Curation does not establish fitting or biological readiness")
        if self.cohort_rule != "complete_declared_features_only":
            raise ValueError("Cohort selection cannot use reference labels or covariates")
        if self.point_arms != (
            "morphology_full_bic",
            "ecology_full_bic",
            "integrative_full_bic",
            "integrative_tied_bic",
        ):
            raise ValueError("Prespecified point comparisons required")
        frozen_text = dict(
            pipeline_reference_commit="54fe5675b0308079f52c340cf801c59d8e3b7ea3",
            scaling="standard",
            missing_policy="error",
            view_weights="equal",
            selection_criterion="bic",
            comparison_rule="descriptive_per_arm_only",
        )
        if any(getattr(self, k) != v for k, v in frozen_text.items()):
            raise ValueError("Preprocessing and comparison settings are frozen")
        expected = dict(
            n_threads=1,
            n_pcs_per_view=2,
            joint_n_pcs=2,
            max_k=4,
            min_cluster_size=5,
            n_init=10,
            max_iter=500,
            random_seed=20261230,
            significance_draws=0,
            stability_draws=0,
        )
        if any(
            type(getattr(self, k)) is not int or getattr(self, k) != v for k, v in expected.items()
        ):
            raise ValueError("The frozen point protocol cannot be tuned or run during curation")
        if (
            type(self.tol) not in (int, float)
            or self.tol != 0.001
            or (type(self.reg_covar) not in (int, float) or self.reg_covar != 1e-6)
        ):
            raise ValueError("Retain existing numerical settings")
        if self.reference_metrics != (
            "selected_k",
            "adjusted_rand",
            "adjusted_mutual_info",
            "reference_coverage",
            "failures",
        ):
            raise ValueError("Reference concordance, coverage and failures must be separate")

    def check_profile(self, profile):
        if (
            self.dataset_id != profile.dataset_id
            or self.source_sha256 != profile.source.sha256
            or self.profile_sha256 != configuration_hash(profile.to_dict())
        ):
            raise ValueError(
                "Protocol must bind the exact dataset, source bytes and "
                "feature/identity declarations"
            )
        if any(
            sum(f.kind == k for f in profile.features) < self.n_pcs_per_view
            for k in ("morphological", "ecological")
        ):
            raise ValueError("The future fixed PCA dimensions exceed declared feature counts")

    def to_dict(self):
        import json

        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        v = checked_fields(cls, value)
        for name in ("point_arms", "reference_metrics"):
            if not isinstance(v[name], list):
                raise ValueError("Protocol sequences must be arrays")
            v[name] = tuple(v[name])
        return cls(**v)
