"""Opt-in denoising workflow with numeric, inference-only reload."""

from super_deli.models.denoising_config import DENOISING_TRAINING, DenoisingConfig
from super_deli.workflows._neural_common import load_numeric_neural_model, run_neural_workflow


def run_tabular_denoising(dataset, *, model_config, base_dir, output_dir, modality_name=None):
    def fit(table, *, config):
        from super_deli.models.tabular_denoising import fit_tabular_denoising

        return fit_tabular_denoising(table, config=config)

    return run_neural_workflow(
        dataset,
        model_config=model_config,
        config_type=DenoisingConfig,
        fit_function=fit,
        training=DENOISING_TRAINING,
        base_dir=base_dir,
        output_dir=output_dir,
        modality_name=modality_name,
    )


def load_denoising_model(directory):
    from pathlib import Path

    import numpy as np

    from super_deli.io.config import read_json

    summary = read_json(Path(directory) / "summary.json")
    DenoisingConfig.from_dict(summary["configuration"]["model"])
    if summary["configuration"]["training"] != DENOISING_TRAINING:
        raise ValueError("Unsupported denoising training recipe")
    model = load_numeric_neural_model(directory, family="tabular_denoising_autoencoder_hdbscan")
    for name, shape in (("losses", (501,)), ("corrupted_objectives", (500,))):
        a = model.arrays[name]
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError("Invalid denoising objective history")
    return model
