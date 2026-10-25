"""Offline installed CLI check with entirely invented records and evidence."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    guard = (
        "import sys; "
        "blocked={'numpy','scipy','sklearn','snpio','requests'}; "
        'exec("def guard(event,args):\\n'
        " if event=='socket.connect' or (event=='import' and args[0].split('.')[0] in blocked):\\n"
        "  raise RuntimeError('Forbidden numerical import or network event')\"); "
        "sys.addaudithook(guard); from super_deli.cli import main; sys.exit(main())"
    )
    common = [sys.executable, "-c", guard]
    command = [
        *common,
        "audit-replacement-empirical",
        str(here / "fixture/plan.json"),
        "--source-dir",
        str(here / "fixture"),
        "--output-dir",
        str(root / "audit"),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    summary = json.loads(result.stdout)
    assert summary["metadata_records"] == 4 and summary["reference_records"] == 4
    assert (
        summary["empirical_fits"]
        == summary["pvalues_generated"]
        == summary["stability_draws"]
        == summary["significance_draws"]
        == 0
    )
    assert summary["reference_link_counts"] == {"candidate_namespace": 3, "unmatched": 1}
    assert summary["readiness"]["independent_unit_fitting"].startswith("blocked")
    print(
        json.dumps(
            dict(
                status="passed",
                synthetic_source_rows=4,
                included_rows=4,
                empirical_fits=0,
                stability_draws=0,
                significance_draws=0,
                network_requests=0,
            )
        )
    )


if __name__ == "__main__":
    main()
