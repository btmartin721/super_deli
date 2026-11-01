"""Ordinary neural workflow; shared I/O preserves the existing contract."""

from super_deli.models.neural_config import TRAINING, NeuralConfig
from super_deli.workflows._neural_common import load_numeric_neural_model, run_neural_workflow


def run_tabular_neural(dataset, *, model_config, base_dir, output_dir, modality_name=None):
    def fit(table, *, config):
        from super_deli.models.tabular_neural import fit_tabular_neural

        return fit_tabular_neural(table, config=config)

    return run_neural_workflow(
        dataset,
        model_config=model_config,
        config_type=NeuralConfig,
        fit_function=fit,
        training=TRAINING,
        base_dir=base_dir,
        output_dir=output_dir,
        modality_name=modality_name,
    )


def load_neural_model(directory):
    return load_numeric_neural_model(directory, family="tabular_autoencoder_hdbscan")
