"""Dependency-free secondary audit of every retained statistic and paired tail."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

CASES = (
    "independent_gaussian",
    "correlated_gaussian",
    "zero_conditional_gaussian",
    "rank_two_gaussian",
    "weak_conditional_gaussian",
)


def read(path):
    return json.loads(path.read_text())


def check(root, table, archived=None):
    path = Path(__file__).parents[1] / "grouped-likelihood/check_tails.py"
    spec = importlib.util.spec_from_file_location("independent_candidate_equations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        (r["cohort"], r["case"], int(r["repetition"]), r["arm"], r["statistic_kind"]): r
        for r in rows
    }
    if len(lookup) != len(rows):
        raise ValueError("Duplicate reported tails")
    observed_count = verified = checked = withheld = 0
    sources = [("fresh", root)]
    if archived is not None:
        sources.insert(0, ("archived", archived))
    for cohort, source in sources:
        for case in CASES:
            study = read(source / case / "study.json")
            policy = study["calibration"]
            m, d = policy["n_simulations"], policy["model"]["n_pcs"]
            for i in range(study["n_datasets"]):
                base = source / case / f"dataset_{i:04d}"
                observed_count += 1
                arms = ("fitted", "known") if cohort == "fresh" else ("fitted",)
                shared = None
                for arm in arms:
                    directory = base / ("calibration" if arm == "fitted" else "known")
                    point_path = directory / "observed.json"
                    point = read(point_path) if point_path.exists() else {"status": "failed"}
                    if arm == "fitted":
                        shared = point
                    elif point_path.exists() and point != shared:
                        raise ValueError("Shared observed fit differs")
                    observed = module.calculate(point, d)
                    if arm != "known":
                        verified += observed is not None
                    records = [read(p) for p in sorted(directory.glob("draw_*.json"))]
                    values = [module.calculate(r, d) for r in records]
                    verified += sum(v is not None for v in values)
                    if len(records) > m or any(r["replicate"] != j for j, r in enumerate(records)):
                        raise ValueError("Requested draw sequence differs")
                    complete = (
                        observed is not None
                        and len(values) == m
                        and all(v is not None for v in values)
                    )
                    for statistic, key in (("likelihood", "aic"), ("bic", "bic")):
                        row = lookup[cohort, case, i, arm, statistic]
                        checked += 1
                        x = (
                            sum(v[key] >= observed[key] for v in values if v is not None)
                            if observed is not None
                            else None
                        )
                        p = (1 + x) / (m + 1) if complete else None
                        if p is None:
                            withheld += 1
                            if row["p_value"] or row["reject"]:
                                raise ValueError("Incomplete test exposes inference")
                        elif (
                            float(row["p_value"]) != p
                            or int(row["exceedances"]) != x
                            or (row["reject"] == "True") != (p <= policy["alpha"])
                        ):
                            raise ValueError("Independent exact tail differs")
                        if row.get("unattempted_draws", "") != "" and int(
                            row["unattempted_draws"]
                        ) != m - len(values):
                            raise ValueError("Unattempted draw count differs")
                    if (directory / "summary.json").exists():
                        expected = (
                            (1 + sum(v["aic"] >= observed["aic"] for v in values)) / (1 + m)
                            if complete
                            else None
                        )
                        if read(directory / "summary.json")["inference"]["p_value"] != expected:
                            raise ValueError("Primary stored p-value differs")
    if checked != len(rows):
        raise ValueError("Unexpected or missing tail rows")
    return {
        "observations": observed_count,
        "verified_unique_pipeline_statistics": verified,
        "tails_checked": checked,
        "withheld_tails": withheld,
        "scope": (
            "Independent candidate equations and all upper tails; copied observed fits count once"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--tails", type=Path, required=True)
    parser.add_argument("--archived-study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as handle:
        json.dump(check(args.study_root, args.tails, args.archived_study), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
