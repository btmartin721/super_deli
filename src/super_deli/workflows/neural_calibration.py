"""Full raw-data refits for an approximate fitted-Gaussian neural clustering test."""

import time
from functools import partial
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import write_json
from super_deli.models.neural_calibration_config import NeuralCalibrationConfig, raw_null_policy
from super_deli.workflows.calibration_artifacts import fit_pipeline
from super_deli.workflows.empirical_calibration import TAIL_OPTIONS, run_fitted_calibration

SCOPE = (
    "Approximate plug-in joint raw-Gaussian departure test for complete independent matched "
    "morphology/ecology. Each draw refits ingestion, scaling, both separate encoders, coordinate "
    "standardization/fusion and the entire tied-covariance BIC GMM K search. Neural/mixture "
    "seeds are fixed; no seed/model selection is calibrated. Not selected-K, cluster or species "
    "significance; no general size guarantee. Stability remains separate."
)


def fit_input(dataset, *, model_config, base_dir, output_dir, point_function=None):
    """The native numerical point pipeline, with compact artifacts for streamed draws."""
    import numpy as np

    from super_deli.io.tabular import ingest_tabular
    from super_deli.models import separate_gmm
    from super_deli.models.matched_preprocessing import align_tables
    from super_deli.models.tabular_neural import point_arrays

    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(exist_ok=False)
    model = type(model_config).from_dict(model_config.to_dict())
    loaded = ingest_tabular(
        dataset,
        base_dir=base_dir,
        output_dir=out / "ingestion",
        modality_names=[model.morphology, model.ecology],
    )
    x, alignment = align_tables(loaded.tables, model, minimum=10)
    write_json(alignment, out / "alignment.json")
    np.savez_compressed(out / "input.npz", values=x)
    try:
        point = (point_function or separate_gmm.fit_shared_point)(x, config=model)
    except Exception as exc:
        np.savez_compressed(out / "failed_point.npz", **getattr(exc, "arrays", {}))
        write_json(
            {"error": str(exc), "diagnostics": getattr(exc, "diagnostics", {})},
            out / "failure.json",
        )
        raise
    np.savez_compressed(out / "point.npz", **point_arrays(point))
    np.savez_compressed(out / "fusion.npz", scores=point.scores)
    write_json(point.diagnostics["candidates"], out / "candidates.json")
    result = dict(
        selected_k=point.selected_k,
        unit_ids=alignment["unit_ids"],
        point_seed=model.random_seed,
        mixture_seed=model.mixture_seed,
        point_pipeline_seconds=time.monotonic() - started,
        candidates=point.diagnostics["candidates"],
        numerical_pipeline=point.diagnostics.get(
            "benchmark_control", "Unchanged native separate-encoder/shared-GMM point fit"
        ),
        provenance={
            "mixture_sha256": fingerprint_file(out / "point.npz").sha256,
            "fusion_sha256": fingerprint_file(out / "fusion.npz").sha256,
        },
    )
    write_json(result, out / "summary.json")
    return result


def load_raw_null(dataset, config, source, destination, fit_null):
    from super_deli.io.tabular import ingest_tabular
    from super_deli.models.matched_preprocessing import align_tables

    loaded = ingest_tabular(
        dataset,
        base_dir=source,
        output_dir=destination / "null_ingestion",
        modality_names=[config.model.morphology, config.model.ecology],
    )
    # Validate matching and one-to-one identities before estimating any null.
    align_tables(loaded.tables, config.model, minimum=10)
    return fit_null(dataset, loaded.tables, raw_null_policy(config))


def run_neural_calibration(dataset, *, config, base_dir, output_dir):
    config = NeuralCalibrationConfig.from_dict(config.to_dict())
    try:
        import numpy as np
        import torch  # noqa: F401

        from super_deli.models.empirical_null import fit_null, simulate_null, write_raw_views
        from super_deli.models.monte_carlo import summarize_upper_tail
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[neural]' for neural calibration") from exc
    return run_fitted_calibration(
        dataset,
        config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        dependencies=(
            np,
            fit_null,
            simulate_null,
            write_raw_views,
            summarize_upper_tail,
            fit_input,
        ),
        scope=SCOPE,
        tail_options=TAIL_OPTIONS,
        seed_namespace="separate_raw_gaussian_v1",
        target="joint_gaussian_raw_continuous_views_neural_pipeline",
        null_hypothesis="Independent matched raw continuous rows follow one full Gaussian",
        load_null=load_raw_null,
        point_fitter=partial(fit_pipeline, error_stage="separate_encoder_shared_gmm_pipeline"),
        evidence_method="Approximate full-pipeline neural fitted-Gaussian BIC bootstrap",
        software_names=("super-deli", "numpy", "scipy", "scikit-learn", "torch", "threadpoolctl"),
    )
