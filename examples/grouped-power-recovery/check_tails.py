"""Standalone standard-library reconstruction of compact-study criteria and tails."""

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path, PurePosixPath


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def statistics(record, dimensions, max_k, minimum):
    if record["status"] != "complete":
        if record["status"] != "failed" or not record.get("error"):
            raise ValueError("Malformed failed point record")
        return None
    if [c["k"] for c in record["candidates"]] != list(range(1, max_k + 1)):
        raise ValueError("Candidate K search differs")
    eligible = []
    for candidate in record["candidates"]:
        if candidate["status"] == "insufficient_samples":
            if record["n_retained"] >= candidate["k"] * minimum:
                raise ValueError("False point sample-size infeasibility")
            continue
        k, likelihood = candidate["k"], candidate["log_likelihood"]
        q = k - 1 + k * dimensions + k * dimensions * (dimensions + 1) // 2
        for name, expected in (
            ("aic", -2 * likelihood + 2 * q),
            ("bic", -2 * likelihood + q * math.log(record["n_retained"])),
        ):
            if not math.isclose(candidate[name], expected, rel_tol=1e-10, abs_tol=1e-7):
                raise ValueError("Independent information-criterion equation differs")
        sizes = candidate["distinct_unit_sizes"]
        if (
            sum(sizes) != record["n_retained"]
            or sum(candidate["hard_cluster_sizes"]) != record["n_retained"]
        ):
            raise ValueError("Point membership counts differ")
        admissible = candidate["converged"] and min(sizes) >= minimum
        if admissible != (candidate["status"] == "eligible"):
            raise ValueError("Point candidate eligibility differs")
        if admissible:
            eligible.append(candidate)
    one = next(c for c in eligible if c["k"] == 1)
    aic = min(eligible, key=lambda c: (c["aic"], c["k"]))["k"]
    bic = min(eligible, key=lambda c: (c["bic"], c["k"]))["k"]
    values = {
        "likelihood": 2 * (max(c["log_likelihood"] for c in eligible) - one["log_likelihood"]),
        "bic": one["bic"] - min(c["bic"] for c in eligible),
    }
    if (
        (record["selected_k"], record["bic_selected_k"]) != (aic, bic)
        or record["statistic"] != values["likelihood"]
        or record["bic_statistic"] != values["bic"]
    ):
        raise ValueError("Independent selection/statistic differs")
    return values


def check(root, audit):
    study = json.loads((root / "study.json").read_text())
    model = study["calibration"]["model"]
    requested = study["calibration"]["n_simulations"]
    rows = [json.loads(s) for s in (audit / "observations.jsonl").read_text().splitlines()]
    audited = {(r["cell"], r["repetition"]): r for r in rows}
    ledger = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    if len(audited) != len(ledger) or len(rows) != len(ledger):
        raise ValueError("Independent checker requires one audited row per observation")
    count = tails = withheld = 0
    for entry in ledger:
        path = root / entry["archive"]
        if digest(path) != entry["archive_sha256"]:
            raise ValueError("Compact evidence fingerprint differs")
        manifest_path = root / entry["manifest"]
        if digest(manifest_path) != entry["manifest_sha256"]:
            raise ValueError("Compact member manifest fingerprint differs")
        manifest = json.loads(manifest_path.read_text())["files"]
        wanted, names = {}, set()
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or not member.isfile()
                    or member.name in names
                    or member.name not in manifest
                ):
                    raise ValueError("Invalid compact archive membership")
                names.add(member.name)
                if member.name in {
                    "calibration/observed.json",
                    "calibration/null_replicates.json",
                    "calibration/summary.json",
                    "record.json",
                } or (
                    member.name.startswith("calibration/draw_") and member.name.endswith(".json")
                ):
                    data = archive.extractfile(member).read()
                    if hashlib.sha256(data).hexdigest() != manifest[member.name]["sha256"]:
                        raise ValueError("Selected raw record fingerprint differs")
                    wanted[member.name] = json.loads(data)
        if names != manifest.keys():
            raise ValueError("Missing compact archive members")
        result = audited[entry["cell"], entry["repetition"]]
        observed = wanted.get("calibration/observed.json")
        draws = [wanted[name] for name in sorted(wanted) if name.startswith("calibration/draw_")]
        if (
            "calibration/null_replicates.json" in wanted
            and wanted["calibration/null_replicates.json"] != draws
        ):
            raise ValueError("Null replicate ledgers differ")
        record = wanted["record.json"]
        arguments = model["n_pcs"], model["max_k"], model["min_cluster_size"]
        point = statistics(observed, *arguments) if observed else None
        reference = [statistics(r, *arguments) for r in draws]
        complete = (
            "calibration/summary.json" in wanted
            and point is not None
            and len(reference) == requested
            and all(r is not None for r in reference)
        )
        count += (observed is not None) + len(draws)
        for index, key in enumerate(("likelihood", "bic")):
            exceedances = (
                sum(r[key] >= point[key] for r in reference if r is not None)
                if point is not None
                else None
            )
            p = (1 + exceedances) / (requested + 1) if complete else None
            row = result["tails"][index]
            if row["p_value"] != p or row["reject"] != (
                p <= study["calibration"]["alpha"] if p is not None else None
            ):
                raise ValueError("Independent plus-one upper tail/decision differs")
            if point is not None and row["exceedances"] != exceedances:
                raise ValueError("Independent exceedance count differs")
            if key == "likelihood" and record["calibration"]["p_value"] != p:
                raise ValueError("Execution likelihood p-value differs")
            tails += 1
            withheld += p is None
    return {
        "passed": True,
        "dependencies": "Python standard library only",
        "observations": len(ledger),
        "calibration_pipeline_records": count,
        "tails_checked": tails,
        "withheld_tails": withheld,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.study_root, args.audit_root)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
