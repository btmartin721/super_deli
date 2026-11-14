"""Dependency-light command registration; numerical imports stay lazy."""

import sys
from contextlib import redirect_stdout
from pathlib import Path


def register(commands):
    for name, help_text in (
        ("cluster-genomic-neural", "Categorical genotype autoencoder/shared GMM"),
        ("analyze-genomic-models", "Compare genomic PCA and categorical neural models"),
    ):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("config", type=Path)
        parser.add_argument("--model-config", type=Path, required=True)
        parser.add_argument("--modality")
        parser.add_argument("--output-dir", type=Path, required=True)


def dispatch(args):
    from super_deli.core.schema import config_from_dict
    from super_deli.io.config import read_json
    from super_deli.models.genomic_neural_config import GenomicComparisonConfig, GenomicNeuralConfig
    from super_deli.workflows.genomic_neural import run_genomic_comparison, run_genomic_neural

    source = args.config.absolute()
    dataset = config_from_dict(read_json(source))
    compare = args.command == "analyze-genomic-models"
    model = (GenomicComparisonConfig if compare else GenomicNeuralConfig).from_dict(
        read_json(args.model_config)
    )
    with redirect_stdout(sys.stderr):
        return (run_genomic_comparison if compare else run_genomic_neural)(
            dataset,
            **{("analysis_config" if compare else "model_config"): model},
            base_dir=source.parent,
            output_dir=args.output_dir,
            modality_name=args.modality,
        )
