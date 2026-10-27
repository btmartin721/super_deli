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
            str(here.parent / "amphipod-gates/fixture/plan.json"),
            "--prior-plan",
            str(prior / "plan.json"),
            "--source-dir",
            str(prior),
            "--prior-audit-dir",
            str(args.output_dir / "prior"),
            "--evidence-dir",
            str(here.parent / "amphipod-gates/fixture"),
            "--output-dir",
            str(args.output_dir / "assessment"),
        ],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
    import hashlib

    from super_deli.core.provenance import configuration_hash
    from super_deli.core.trait_pilot import TraitPilotProtocol

    prior_assessment = args.output_dir / "assessment"
    evidence_plan = here.parent / "amphipod-gates/fixture/plan.json"
    plan = TraitPilotProtocol(
        "invented_pilot",
        configuration_hash(json.loads(evidence_plan.read_text())),
        hashlib.sha256((prior_assessment / "checksums.json").read_bytes()).hexdigest(),
    )
    pilot_plan = args.output_dir / "pilot-plan.json"
    pilot_plan.write_text(json.dumps(plan.to_dict()))
    result = subprocess.run(
        [
            *common,
            "assess-trait-pilot",
            str(pilot_plan),
            "--prior-plan",
            str(prior / "plan.json"),
            "--evidence-plan",
            str(evidence_plan),
            "--source-dir",
            str(prior),
            "--prior-audit-dir",
            str(args.output_dir / "prior"),
            "--evidence-dir",
            str(evidence_plan.parent),
            "--prior-assessment-dir",
            str(prior_assessment),
            "--output-dir",
            str(args.output_dir / "pilot"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    summary = json.loads(result.stdout)
    assert summary["source_rows"] == 6 and len(summary["rosters"]) == 4
    assert summary["fitting_authorized"] is summary["contact_performed"] is False
    assert summary["empirical_fits"] == summary["pvalues_generated"] == 0
    assert summary["readiness"]["species_boundary_validation"]["status"] == "blocked"
    print(
        json.dumps(
            dict(
                status="passed",
                synthetic_source_rows=6,
                installed_trait_pilot_smoke="passed",
                empirical_fits=0,
                network_requests=0,
                contact_performed=False,
            )
        )
    )


if __name__ == "__main__":
    main()
