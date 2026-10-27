"""Non-executable, source-bound amphipod feasibility protocol."""

from dataclasses import asdict, dataclass

from super_deli.core.correspondence import digest
from super_deli.core.empirical import NAME, checked_fields
from super_deli.core.trait_audit import TRAITS

CORE19 = TRAITS[:19]
SOMATIC24 = TRAITS[:-1]
PANELS = {"core19": CORE19, "somatic24": SOMATIC24}
SEXES = ("male", "female")
REASONS = (
    "ambiguous_identity",
    "outside_family",
    "outside_sex_stage",
    "formula_measurement",
    "uncertain_measurement",
    "missing_measurement",
    "nonpositive_measurement",
)


def future_point_design():
    """Descriptive protocol fields, deliberately not executable model configs."""
    return dict(
        phase="future_design_only",
        fitting_authorized=False,
        independent_units_confirmed=False,
        scaling="standard",
        n_pcs=2,
        random_seed=79147,
        significance_draws=0,
        bootstrap_draws=0,
        subsampling_draws=0,
        arms=[
            dict(
                family="gmm",
                covariance_type="full",
                selection="bic",
                max_k=4,
                min_cluster_size=5,
                n_init=10,
                max_iter=500,
                tol=0.001,
                reg_covar=1e-6,
            ),
            dict(
                family="hdbscan",
                min_cluster_size=5,
                min_samples=5,
                cluster_selection_method="eom",
                allow_single_cluster=True,
                alpha=1.0,
                cluster_selection_epsilon=0.0,
            ),
        ],
        comparison="descriptive_each_partition_separately_no_winner",
        future_metrics=["pairwise_covered_ari", "pairwise_covered_ami", "coverage", "noise"],
    )


@dataclass(frozen=True)
class TraitPilotProtocol:
    protocol_id: str
    evidence_specification_sha256: str
    evidence_checksums_sha256: str
    schema_version: int = 1
    phase: str = "trait_pilot_feasibility_only"
    cohort_policy: str = "niphargidae_separate_male_female_core19_somatic24_raw_complete_positive"
    reference_policy: str = "attach_after_selection_separate_namespaces_no_species_truth"
    contact_policy: str = "no_contact"
    fitting_authorized: bool = False
    independent_units_confirmed: bool = False
    reference_species_truth_confirmed: bool = False

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported trait pilot protocol")
        if not isinstance(self.protocol_id, str) or not NAME.fullmatch(self.protocol_id):
            raise ValueError("Neutral protocol ID required")
        for name in ("evidence_specification_sha256", "evidence_checksums_sha256"):
            digest(getattr(self, name))
        expected = dict(
            phase="trait_pilot_feasibility_only",
            cohort_policy="niphargidae_separate_male_female_core19_somatic24_raw_complete_positive",
            reference_policy="attach_after_selection_separate_namespaces_no_species_truth",
            contact_policy="no_contact",
        )
        if any(getattr(self, k) != v for k, v in expected.items()):
            raise ValueError("The approved pilot policies are fixed")
        for name in (
            "fitting_authorized",
            "independent_units_confirmed",
            "reference_species_truth_confirmed",
        ):
            if type(getattr(self, name)) is not bool or getattr(self, name):
                raise ValueError("Fit-free feasibility cannot promote biological readiness")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**checked_fields(cls, value))
