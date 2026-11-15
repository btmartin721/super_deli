"""Matched shape-capable orchestration through unchanged native PCA/GMM fitters."""

from super_deli.io.multimodal_analysis import freeze_multimodal_inputs
from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig
from super_deli.workflows.genomic_analysis_adapters import read_genomic_arm, run_genomic_arm
from super_deli.workflows.genomic_analysis_report import write_genomic_report
from super_deli.workflows.matched_comparison import run_comparison

LIMITATIONS = [
    "Candidate partitions do not establish species; no automatic winner or consensus.",
    "Reviewed fixed homologous landmarks and independent matched specimens are assumptions.",
    "GPA removes translation, rotation and positive scale; it does not"
    " correct allometry or digitization error.",
    "Genomic diagonal, shape/tabular full and declared integrative covariance families differ.",
    "Compare BIC within each native search only; memberships are conditional fitted assignments.",
    "Whole-specimen shape and paired-view bootstraps refit GPA/PCA and"
    " retain all landmarks together.",
    "Locus-block stability holds specimens fixed; specimen stability h"
    "olds source loci fixed. Do not pool them.",
    "Correlated views can emphasize shared signal; added views need not improve recovery.",
    "No missingness, grouped/unmatched data, image detection, semiland"
    "mark sliding or new p-values.",
    "Existing defaults, historical p-values and empirical evidence gates remain unchanged.",
]


def _read_arm(arm, directory, cohort, frozen):
    from super_deli.io.config import read_json

    native = read_genomic_arm(arm, directory, cohort, frozen)
    summary = read_json(directory / "summary.json")
    expected = {
        v["modality"]: dict(zip(cohort["unit_ids"], v["observation_ids"], strict=True))
        for v in cohort["views"]
    }
    if arm.family == "integrative":
        for view in summary["views"]:
            actual = {m["target_id"]: m["observation_ids"] for m in view["members"]}
            if actual != {u: [o] for u, o in expected[view["modality"]].items()}:
                raise ValueError("Native combined observation/unit binding mismatch")
        native["evidence_type"] = "paired_specimen_bootstrap_refitting_selected_views"
    else:
        ids = summary["specimen_ids"] if arm.family == "genomic" else summary["unit_ids"]
        observations = (
            summary["sample_ids"] if arm.family == "genomic" else summary["observation_ids"]
        )
        if dict(zip(ids, observations, strict=True)) != expected[arm.modalities[0]]:
            raise ValueError("Native singleton observation/unit binding mismatch")
    return native


def _report(summary, points, output):
    write_genomic_report(summary, points, output, title="multimodal landmark-shape")


def run_multimodal_analysis(dataset, *, analysis_config, base_dir, output_dir):
    return run_comparison(
        dataset,
        analysis_config=analysis_config,
        base_dir=base_dir,
        output_dir=output_dir,
        config_type=MultimodalAnalysisConfig,
        freeze_inputs=freeze_multimodal_inputs,
        run_arm=run_genomic_arm,
        read_arm=_read_arm,
        render=_report,
        limitations=LIMITATIONS,
    )
