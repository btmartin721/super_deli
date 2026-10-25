"""Installed, offline correspondence assessment with invented documentary records."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    prior = here.parent / "replacement-empirical/fixture"
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
            "audit-replacement-empirical",
            str(prior / "plan.json"),
            "--source-dir",
            str(prior),
            "--output-dir",
            str(root / "prior"),
        ],
        [
            *common,
            "assess-correspondence",
            str(here / "fixture/plan.json"),
            "--prior-plan",
            str(prior / "plan.json"),
            "--source-dir",
            str(prior),
            "--prior-audit-dir",
            str(root / "prior"),
            "--evidence-dir",
            str(here / "fixture"),
            "--output-dir",
            str(root / "assessment"),
        ],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
    summary = json.loads(result.stdout)
    assert summary["resolved_voucher_specimens"] == summary["resolved_haul_specimens"] == 4
    assert summary["readiness"]["empirical_analysis"].startswith("not_authorized")
    assert summary["contact_performed"] is False
    assert summary["empirical_fits"] == summary["pvalues_generated"] == 0
    print(
        json.dumps(
            dict(
                status="passed",
                synthetic_specimens=4,
                empirical_fits=0,
                network_requests=0,
                contact_performed=False,
            )
        )
    )


if __name__ == "__main__":
    main()
