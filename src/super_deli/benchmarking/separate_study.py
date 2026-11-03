"""Prespecified separate-encoder protocol on the shared audited lifecycle."""

from super_deli.benchmarking import neural_lifecycle as engine


def design():
    from super_deli.benchmarking.separate_audit import audit_replay
    from super_deli.benchmarking.separate_config import (
        ARMS,
        CASES,
        NATIVE,
        PRIMARY,
        ROOTS,
        SEEDS,
        SeparateStudy,
        counts,
        model_for,
        tasks,
    )
    from super_deli.models.neural_config import TRAINING
    from super_deli.models.separate_neural import SEPARATE_TRAINING

    return dict(
        config_type=SeparateStudy,
        arms=ARMS,
        native_arms=NATIVE,
        counts=counts,
        model_for=model_for,
        tasks=tasks,
        audit_replay=audit_replay,
        runner="super_deli.benchmarking.separate_run",
        auditor="super_deli.benchmarking.separate_audit",
        branch_parity=True,
        training_counts={
            arm: 2
            if arm.startswith("separate_")
            else 0
            if arm in ("pca4", "integrative_gmm")
            else 1
            for arm in ARMS
        },
        replay_prefix="joint2_",
        settings=dict(
            cases=CASES,
            primary_cases=PRIMARY,
            neural_seeds=SEEDS,
            rng_roots=ROOTS,
            primary_rates=108,
            paired_comparisons=72,
            features_per_view=[4, 8],
            subsampling_fraction=0.8,
            latent_rank_tolerance=1e-12,
            resource_projection=(
                "per-N/arm maxima of training groups, refits, workflow/source overhead "
                "and storage; 25% allowance; no parallel speed credit"
            ),
            fusion=(
                "four concatenated standardized coordinates; no joint reduction or learned weights"
            ),
        ),
        training={
            "separate": SEPARATE_TRAINING,
            "joint2": TRAINING,
            "joint4": {**TRAINING, "encoder_widths": [16, 8, 4], "latent_dimensions": 4},
            "pca4": "two standardized covariance PCs per view; concatenate without reduction",
        },
    )


def declaration(root, config):
    return engine.declaration(root, config, design())


def freeze(root, config):
    return engine.freeze(root, config, design())


def check_frozen(root):
    return engine.check_frozen(root, design())


def run_phase(root, source, phase, *, workers=1):
    return engine.run_phase(root, source, phase, workers=workers, design=design())


def audit_phase(root, source, phase):
    return engine.audit_phase(root, source, phase, design())


def gate(root, *, staging_bytes):
    return engine.gate(root, staging_bytes=staging_bytes, design=design())
