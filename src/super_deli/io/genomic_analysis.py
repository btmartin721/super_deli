"""Backward-compatible genomic comparison input adapter."""

from super_deli.io.multimodal_analysis import freeze_multimodal_inputs


def freeze_genomic_analysis(dataset, config, base_dir, directory):
    return freeze_multimodal_inputs(dataset, config, base_dir, directory)
