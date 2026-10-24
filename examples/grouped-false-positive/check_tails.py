"""Check every retained tail and selected statistic using only Python's standard library."""

import argparse
import csv
import json
from pathlib import Path


def statistic(record):
    if record["status"] != "complete":
        return None
    eligible = [c for c in record["candidates"] if c["status"] == "eligible"]
    one = next(c for c in eligible if c["k"] == 1)
    ll = 2 * (max(c["log_likelihood"] for c in eligible) - one["log_likelihood"])
    bic = one["bic"] - min(c["bic"] for c in eligible)
    if ll != record["statistic"] or bic != record["bic_statistic"]:
        raise ValueError("Retained likelihood/BIC statistic differs")
    return ll, bic


def check(root, tails):
    expected, fits, observations = {}, 0, 0
    for case in sorted(root.iterdir()):
        if not case.is_dir():
            continue
        config = json.loads((case / "study.json").read_text())
        requested = config["calibration"]["n_simulations"]
        alpha = config["calibration"]["alpha"]
        ledgers = [json.loads(s) for s in (case / "datasets.jsonl").read_text().splitlines()]
        if len(ledgers) != config["n_datasets"]:
            raise ValueError("Observed denominator differs")
        for i, ledger in enumerate(ledgers):
            observations += 1
            point_count = 0
            for arm, directory in [("fitted", "calibration"), ("known", "known")]:
                base = case / f"dataset_{i:04d}" / directory
                if not (base / "observed.json").exists():
                    for kind in ("likelihood", "bic"):
                        expected[(case.name, str(i), arm, kind)] = (None, None)
                    continue
                point = json.loads((base / "observed.json").read_text())
                obs = statistic(point)
                draws = [
                    statistic(json.loads(p.read_text())) for p in sorted(base.glob("draw_*.json"))
                ]
                point_count += len(draws) + (arm == "fitted")
                complete = (
                    obs is not None
                    and len(draws) == requested
                    and all(v is not None for v in draws)
                )
                for j, kind in enumerate(("likelihood", "bic")):
                    exceed = (
                        sum(v[j] >= obs[j] for v in draws if v is not None)
                        if obs is not None
                        else None
                    )
                    p_value = (1 + exceed) / (1 + requested) if complete else None
                    expected[(case.name, str(i), arm, kind)] = (
                        p_value,
                        p_value <= alpha if complete else None,
                    )
            if point_count != ledger["attempted_fits"]:
                raise ValueError("Unique point accounting differs")
            fits += point_count
    rows = list(csv.DictReader(tails.open()))
    keys = [(r["case"], r["repetition"], r["arm"], r["statistic_kind"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("Tail identities missing or duplicated")
    for row, key in zip(rows, keys, strict=True):
        p, decision = expected[key]
        actual = float(row["p_value"]) if row["p_value"] else None
        if actual != p or row["reject"] != (str(decision) if decision is not None else ""):
            raise ValueError(f"Independent tail differs: {key}")
    return {
        "observations": observations,
        "unique_pipeline_records": fits,
        "tails_checked": len(rows),
        "withheld_tails": sum(v[0] is None for v in expected.values()),
        "passed": True,
        "dependencies": "Python standard library only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--tails", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.study_root, args.tails)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
