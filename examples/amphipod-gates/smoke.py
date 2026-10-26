"""Installed offline evidence assessment using entirely invented sources."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    prior = here.parent / "amphipod-audit/fixture"
    guard = (
        "import sys; blocked={'numpy','scipy','sklearn','snpio','requests'}; "
        'exec("def guard(event,args):\\n'
        " if event=='socket.connect' or (event=='import' and args[0].split('.')[0] in blocked):\\n"
        "  raise RuntimeError('Forbidden numerical import or network event')\"); "
        "sys.addaudithook(guard); from super_deli.cli import main; sys.exit(main())"
    )
    common = [sys.executable, "-c", guard]
    commands = [
        [
            *common,
            "audit-trait-source",
            str(prior / "plan.json"),
            "--source-dir",
            str(prior),
            "--output-dir",
            str(args.output_dir / "prior"),
        ],
        [
            *common,
            "assess-trait-evidence",
            str(here / "fixture/plan.json"),
            "--prior-plan",
            str(prior / "plan.json"),
            "--source-dir",
            str(prior),
            "--prior-audit-dir",
            str(args.output_dir / "prior"),
            "--evidence-dir",
            str(here / "fixture"),
            "--output-dir",
            str(args.output_dir / "assessment"),
        ],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
    summary = json.loads(result.stdout)
    assert summary["source_rows"] == 6 and summary["documented_specimen_links"] == 2
    assert summary["explicit_table_corroborations"] == 1
    assert summary["cohort_created"] is summary["contact_performed"] is False
    assert summary["empirical_fits"] == summary["pvalues_generated"] == 0
    assert summary["readiness"]["species_boundary_validation"]["status"] == "blocked"
    print(
        json.dumps(
            dict(
                status="passed",
                synthetic_source_rows=6,
                empirical_fits=0,
                network_requests=0,
                contact_performed=False,
            )
        )
    )


if __name__ == "__main__":
    main()
