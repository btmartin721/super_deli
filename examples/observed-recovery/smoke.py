"""Installed continuation accounting smoke on invented, non-study coordinates."""

import argparse
from dataclasses import replace

import numpy as np

from super_deli.benchmarking.observed_recovery import continued_start, put_parameters
from super_deli.io.config import write_json
from super_deli.models.observed_config import ObservedGMMConfig
from super_deli.models.observed_gmm import fit_start, initialize
from super_deli.models.observed_likelihood import patterns


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    x = np.random.default_rng(92111).normal(size=(32, 2))
    x[:8, 0] = np.nan
    config = ObservedGMMConfig(("a", "b"), max_iter=3, tol=1e-20, fixed_point_tol=1e-20)
    initial = initialize(x, 2, 92112, 1e-4)
    record, evidence = fit_start(x, initial, patterns(x), config)
    record.update(start=0, seed=92112)
    arrays = {"k2_s0_trace": evidence["trace"]}
    for stage in ("initial", "previous", "final"):
        put_parameters(arrays, "k2_s0_" + stage + "_", evidence[stage])
    result, extended = continued_start(x, arrays, "k2_s0_", record, config, total_cap=9)
    _, direct = fit_start(x, initial, patterns(x), replace(config, max_iter=9))
    np.testing.assert_array_equal(extended["k2_s0_trace"], direct["trace"])
    assert result["iterations"] == 9 and record["iterations"] == 3
    write_json(
        dict(
            status="passed",
            production_defaults_unchanged=True,
            original_iterations=3,
            cumulative_iterations=9,
            oracle_used=False,
        ),
        root / "summary.json",
    )
    print("Observed-data recovery diagnostic installed smoke passed")


if __name__ == "__main__":
    main()
