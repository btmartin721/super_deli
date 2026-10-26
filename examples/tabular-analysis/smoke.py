"""Installed-package smoke: run from outside the checkout, without source data."""

import argparse
import json
from pathlib import Path

from super_deli.benchmarking.density_study import generate_raw, write_raw
from super_deli.benchmarking.tabular_comparison import model_arms
from super_deli.benchmarking.tabular_comparison_audit import audit_analysis
from super_deli.cli import main
from super_deli.io.config import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True, type=Path)
args = parser.parse_args()
out = args.output_dir
out.mkdir()
x, _ = generate_raw("four_spherical", 64, 79181)
write_raw(x, out / "input")
write_json(model_arms(2).to_dict(), out / "analysis.json")
status = main(
    [
        "analyze-tabular",
        str(out / "input/dataset.json"),
        "--analysis-config",
        str(out / "analysis.json"),
        "--output-dir",
        str(out / "run"),
    ]
)
assert status == 0
result = audit_analysis(out / "run")
assert result == {"models": 4, "draws": 8, "comparisons": 6}
print(json.dumps({"installed_unified_tabular_smoke": "passed", **result}))
