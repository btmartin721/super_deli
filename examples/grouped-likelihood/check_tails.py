"""Independently recompute all candidate criteria, statistics and paired tails."""

import argparse
import csv
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def calculate(record, d):
    if record["status"] != "complete":
        return None
    n = record["n_retained"]
    candidates = []
    for row in record["candidates"]:
        if row["status"] == "insufficient_samples":
            continue
        k, ll = row["k"], row["log_likelihood"]
        q = k - 1 + k * d + k * d * (d + 1) // 2
        aic, bic = -2 * ll + 2 * q, -2 * ll + q * math.log(n)
        if not math.isclose(aic, row["aic"], abs_tol=1e-8, rel_tol=1e-12) or not math.isclose(
            bic, row["bic"], abs_tol=1e-8
        ):
            raise ValueError("Independent candidate criterion mismatch")
        if row["status"] == "eligible":
            candidates.append(row)
    one = next(c for c in candidates if c["k"] == 1)
    values = {
        "aic": 2 * (max(c["log_likelihood"] for c in candidates) - one["log_likelihood"]),
        "bic": one["bic"] - min(c["bic"] for c in candidates),
    }
    for name, key in (("aic", "statistic"), ("bic", "bic_statistic")):
        if values[name] != record[key]:
            raise ValueError("Independent null/point statistic mismatch")
        selected = min(candidates, key=lambda c: (c[name], c["k"]))["k"]
        if selected != record["selected_k" if name == "aic" else "bic_selected_k"]:
            raise ValueError("Independent component selection mismatch")
    return values


def check(root, paired_table):
    with paired_table.open(newline="") as handle:
        paired = {(r["case"], r["repetition"], r["criterion"]): r for r in csv.DictReader(handle)}
    observed_count, verified, withheld = 0, 0, 0
    for shard in sorted(p for p in root.iterdir() if p.is_dir()):
        declaration = read(shard / "study.json")
        d = declaration["calibration"]["model"]["n_pcs"]
        expected = declaration["calibration"]["n_simulations"]
        for i in range(declaration["n_datasets"]):
            base = shard / f"dataset_{i:04d}/calibration"
            observed_count += 1
            if (base / "failure.json").exists():
                for name in ("aic", "bic"):
                    row = paired[shard.name, str(i), name]
                    if row["status"] != "failed" or row.get("p_value"):
                        raise ValueError("Outer failure exposes inference")
                    withheld += 1
                continue
            result = read(base / "summary.json")
            point = read(base / "observed.json")
            draws = read(base / "null_replicates.json")
            observed = calculate(point, d)
            if observed is None:
                if draws or result["inference"]["p_value"] is not None:
                    raise ValueError("Failed point attempted draws or exposes inference")
                withheld += 2
                continue
            if len(draws) != expected:
                raise ValueError("Requested draw count mismatch")
            verified += 1
            values = [calculate(r, d) for r in draws]
            verified += sum(v is not None for v in values)
            failed = sum(v is None for v in values)
            for name in ("aic", "bic"):
                row = paired[shard.name, str(i), name]
                x = sum(v[name] >= observed[name] for v in values if v is not None)
                p = (1 + x) / (1 + expected) if not failed else None
                if p is None:
                    if row["p_value"] or row["reject"]:
                        raise ValueError("Failed draw exposes inference")
                    withheld += 1
                elif (
                    float(row["p_value"]) != p
                    or int(row["exceedances"]) != x
                    or (row["reject"] == "True") != (p <= declaration["calibration"]["alpha"])
                ):
                    raise ValueError("Independent tail count/p-value/decision mismatch")
                if name == "aic" and result["inference"]["p_value"] != p:
                    raise ValueError("Primary inference differs from independent tail")
    if len(paired) != 2 * observed_count:
        raise ValueError("Paired table has missing or extra observations")
    return {
        "observations": observed_count,
        "verified_successful_pipeline_statistics": verified,
        "paired_tests": len(paired),
        "withheld_tests": withheld,
        "scope": (
            "Independent candidate arithmetic, all point/null statistics "
            "and upper-tail counts; no fits"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--paired-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as handle:
        json.dump(check(args.study_root, args.paired_table), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
