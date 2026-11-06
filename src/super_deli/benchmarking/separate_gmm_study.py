"""Fresh shared-encoder comparison on the existing source-frozen lifecycle."""

from super_deli.benchmarking import neural_lifecycle as engine


def design():
    from super_deli.benchmarking.separate_gmm_audit import audit_replay
    from super_deli.benchmarking.separate_gmm_budget import project
    from super_deli.benchmarking.separate_gmm_config import (
        ARMS,
        CASES,
        NATIVE,
        PRIMARY,
        ROOTS,
        SEEDS,
        SharedEncoderStudy,
        counts,
        model_for,
        tasks,
    )
    from super_deli.models.separate_neural import SEPARATE_TRAINING

    return dict(
        config_type=SharedEncoderStudy,
        arms=ARMS,
        native_arms=NATIVE,
        counts=counts,
        model_for=model_for,
        tasks=tasks,
        audit_replay=audit_replay,
        runner="super_deli.benchmarking.separate_gmm_run",
        auditor="super_deli.benchmarking.separate_gmm_audit",
        serial_runtime_guard=True,
        resource_gate=project,
        settings=dict(
            cases=CASES,
            primary_cases=PRIMARY,
            neural_seeds=SEEDS,
            rng_roots=ROOTS,
            primary_rates=108,
            paired_comparisons=72,
            features_per_view=[4, 8],
            subsampling_fraction=0.8,
            unequal_group_standard_deviations=[0.5, 1.0, 1.5, 2.0],
            noise_sd=0.15,
            representation_refits=(
                "one pair per seed and sampled input; all heads share exact coordinates"
            ),
        ),
        training=SEPARATE_TRAINING,
    )


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
