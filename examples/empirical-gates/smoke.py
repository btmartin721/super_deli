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
    fixture = here.parent / "empirical-foundation/fixture"
    guard = (
        "import sys; "
        "blocked={'numpy','scipy','sklearn','snpio','requests'}; "
        'exec("def guard(event,args):\\n'
        " if event=='socket.connect' or (event=='import' and args[0].split('.')[0] in blocked):\\n"
        "  raise RuntimeError('Forbidden numerical import or network event')\"); "
        "sys.addaudithook(guard); from super_deli.cli import main; sys.exit(main())"
    )
    common = [sys.executable, "-c", guard]
    arguments = [
        str(fixture / "profile.json"),
        "--source",
        str(fixture / "source.csv"),
        "--protocol",
        str(fixture / "protocol.json"),
    ]
    for command in [
        [*common, "curate-empirical", *arguments, "--output-dir", str(root / "curated")],
        [
            *common,
            "assess-empirical",
            *arguments,
            "--evidence",
            str(here / "fixture/evidence.json"),
            "--evidence-dir",
            str(here / "fixture"),
            "--curated-dir",
            str(root / "curated"),
            "--output-dir",
            str(root / "assessment"),
        ],
    ]:
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
    summary = json.loads(result.stdout)
    assert summary["source_rows"] == 12 and summary["included_rows"] == 10
    assert (
        summary["empirical_fits"]
        == summary["stability_draws"]
        == summary["significance_draws"]
        == 0
    )
    assert summary["readiness"]["independent_unit_point_fitting"]["status"] == "blocked"
    assert summary["readiness"]["grouped_predictive_screen"]["status"] == "inapplicable"
    assert not any(k.split(".")[0] in {"numpy", "scipy", "sklearn", "snpio"} for k in sys.modules)
    print(
        json.dumps(
            dict(
                status="passed",
                synthetic_source_rows=12,
                included_rows=10,
                empirical_fits=0,
                stability_draws=0,
                significance_draws=0,
                network_requests=0,
            )
        )
    )


if __name__ == "__main__":
    main()
