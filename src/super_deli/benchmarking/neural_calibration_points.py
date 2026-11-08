"""Paired PCA reference and retained candidate densities; no head retraining."""

from functools import partial

import numpy as np

from super_deli.benchmarking.coordinate_fit import mixture
from super_deli.benchmarking.separate_controls import fit_pca_four
from super_deli.io.config import write_json
from super_deli.workflows.neural_calibration import fit_input


def pca_runner(dataset, *, model_config, base_dir, output_dir, evidence_dir):
    captured = {}

    def cluster(scores, unused_config):
        row, arrays = mixture(scores, "tied")
        captured.update(row=row, arrays=arrays)
        if row["status"] != "available":
            raise ValueError(row.get("error", "PCA GMM search failed"))
        return arrays["labels"], arrays["membership"].max(1)

    def point(x, *, config):
        result = fit_pca_four(x, config=config, clusterer=cluster)
        result.diagnostics["candidates"] = captured["row"]["candidates"]
        return result

    evidence_dir.mkdir(parents=True)
    try:
        return fit_input(
            dataset,
            model_config=model_config,
            base_dir=base_dir,
            output_dir=output_dir,
            point_function=point,
        )
    finally:
        np.savez_compressed(evidence_dir / "fit.npz", **captured.get("arrays", {}))
        write_json(captured.get("row", dict(status="unattempted")), evidence_dir / "execution.json")


def pca_calibration(dataset, *, config, base_dir, output_dir):
    from super_deli.models.empirical_null import fit_null, simulate_null, write_raw_views
    from super_deli.models.monte_carlo import summarize_upper_tail
    from super_deli.workflows.calibration_artifacts import fit_pipeline
    from super_deli.workflows.empirical_calibration import TAIL_OPTIONS, run_fitted_calibration
    from super_deli.workflows.neural_calibration import load_raw_null

    counter = []

    def runner(*args, **kwargs):
        i = len(counter)
        counter.append(i)
        return pca_runner(*args, **kwargs, evidence_dir=output_dir / "head-evidence" / str(i))

    return run_fitted_calibration(
        dataset,
        config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        dependencies=(np, fit_null, simulate_null, write_raw_views, summarize_upper_tail, runner),
        scope="Benchmark-only paired per-view PCA/shared-GMM raw-Gaussian full-pipeline test",
        tail_options=TAIL_OPTIONS,
        seed_namespace="separate_raw_gaussian_v1",
        target="joint_gaussian_raw_continuous_views_pca_reference",
        null_hypothesis="Independent matched raw continuous rows follow one full Gaussian",
        load_null=load_raw_null,
        point_fitter=partial(fit_pipeline, error_stage="pca_shared_gmm_pipeline"),
    )
