"""Small installed opt-in check using invented measurements only."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path


def main():
    import super_deli
    from super_deli.core.schema import config_from_dict
    from super_deli.io.config import read_json
    from super_deli.models.tabular_analysis_config import TabularAnalysisConfig
    from super_deli.workflows.tabular_analysis import run_tabular_analysis

    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    assert version("super-deli") == super_deli.__version__
    r = run_tabular_analysis(
        config_from_dict(read_json(here / "dataset.json")),
        analysis_config=TabularAnalysisConfig.from_dict(read_json(here / "analysis.json")),
        base_dir=here,
        output_dir=a.output_dir,
    )
    assert r["status"] == "complete"
    assert r["policies"]["complete_case"]["eligible_units"] == 29
    assert r["policies"]["mean"]["eligible_units"] == 31
    assert r["inference"]["p_value"] is None
    print(
        json.dumps(
            dict(
                status="passed",
                installed_missing_tabular_smoke="passed",
                original_units=32,
                complete_case=29,
                mean=31,
                new_pvalues=0,
            )
        )
    )


if __name__ == "__main__":
    main()
