"""Invented installed-package smoke: no real study inputs or contact."""

import argparse
from pathlib import Path


def main():
    from super_deli.benchmarking.density_study import generate_raw, write_raw
    from super_deli.models.hdbscan_config import HDBSCANConfig
    from super_deli.workflows.tabular_hdbscan import run_tabular_hdbscan

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    x, _ = generate_raw("four_spherical", 64, 91817)
    dataset = write_raw(x, args.output_dir / "raw")
    result = run_tabular_hdbscan(
        dataset,
        model_config=HDBSCANConfig(True, n_subsamples=3, random_seed=91783),
        base_dir=args.output_dir / "raw",
        output_dir=args.output_dir / "fit",
    )
    assert result["stability"]["successful"] == 3
    assert result["inference"]["p_value"] is None
    assert result["model"] == "tabular_pca_hdbscan"
    print("Density workflow and inclusion-conditional stability verified")


if __name__ == "__main__":
    main()
