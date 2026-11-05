"""Frozen archival diagnostic on the shared source-bound experiment lifecycle."""

from super_deli.benchmarking import neural_lifecycle as engine


def declaration_base(root, config):
    from super_deli.benchmarking.neural_moons_study import runtime, source_hashes
    from super_deli.core.provenance import fingerprint_file

    return dict(
        configuration=config.to_dict(),
        runtime=runtime(),
        source_sha256=source_hashes(),
        plan_sha256=fingerprint_file(root / "approved-plan.md").sha256,
    )


def design():
    from super_deli.benchmarking.coordinate_config import (
        SETTINGS,
        SPACES,
        CoordinateStudy,
        MixtureSettings,
        counts,
        model_for,
        tasks,
    )

    return dict(
        config_type=CoordinateStudy,
        arms=SPACES,
        native_arms=SPACES,
        counts=counts,
        model_for=model_for,
        tasks=tasks,
        declaration_base=declaration_base,
        archival=True,
        serial_runtime_guard=True,
        settings={
            **SETTINGS,
            "mixtures": {kind: MixtureSettings(kind).to_dict() for kind in ("full", "tied")},
        },
        training=dict(
            neural_fits=0,
            new_datasets=0,
            resampling=0,
            existing_gmm_optimization="unchanged ten-start searches",
        ),
        runner="super_deli.benchmarking.coordinate_run",
        auditor="super_deli.benchmarking.coordinate_audit",
        audit_replay=None,
        audit_boundary=(
            "Frozen raw-to-score transforms, source density replay, independent "
            "geometry/hierarchy, Gaussian densities, convergence, eligibility, "
            "BIC and accounting; no neural optimization is run or re-audited"
        ),
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
