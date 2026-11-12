"""Installed-package checks with invented specimens, never empirical inference."""

import argparse
import json

from super_deli.benchmarking.residual_config import model_for
from super_deli.benchmarking.residual_fixture import reference_example, smoke
from super_deli.workflows.residual_reference import predict_matched_residual, train_matched_residual


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study", action="store_true")
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir()
    dataset = reference_example(root / "example")
    train = train_matched_residual(
        dataset,
        labels_path=root / "example/references.csv",
        model_config=model_for("residual", 0),
        base_dir=root / "example/raw",
        output_dir=root / "trained",
    )
    predicted = predict_matched_residual(
        dataset,
        reference_run=root / "trained",
        base_dir=root / "example/raw",
        output_dir=root / "predicted",
    )
    assert train["unit_ids"] == predicted["unit_ids"]
    result = dict(passed=True, reference_units=len(train["unit_ids"]), scope=predicted["scope"])
    if args.study:
        result["study"] = smoke(root / "lifecycle", workers=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
