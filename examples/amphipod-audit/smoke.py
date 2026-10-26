"""Installed fit-free workflow, with invented records and blocked network/models."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent / "fixture"
    guard = (
        "import sys; blocked={'numpy','scipy','sklearn','snpio','requests'}; "
        'exec("def guard(event,args):\\n'
        " if event=='socket.connect' or (event=='import' and args[0].split('.')[0] in blocked):\\n"
        "  raise RuntimeError('Forbidden numerical import or network event')\"); "
        "sys.addaudithook(guard); from super_deli.cli import main; sys.exit(main())"
    )
    command = [
        sys.executable,
        "-c",
        guard,
        "audit-trait-source",
        str(here / "plan.json"),
        "--source-dir",
        str(here),
        "--output-dir",
        str(args.output_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    a = json.loads(result.stdout)
    assert a["source_rows"] == 6 and a["continuous_traits"] == 25
    assert a["unflagged_literal_links"] == 1
    assert a["empirical_fits"] == a["pvalues_generated"] == a["bootstrap_draws"] == 0
    assert a["contact_performed"] is a["email_sent"] is False
    assert a["readiness"]["reference_label_independence"].startswith("not_established")
    print(
        json.dumps(
            dict(
                status="passed",
                synthetic_specimens=6,
                empirical_fits=0,
                network_requests=0,
                contact_performed=False,
            )
        )
    )


if __name__ == "__main__":
    main()
