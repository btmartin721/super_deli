"""Standard-library archive, component-criterion and rate-accounting audit."""

import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def verify(study_root, audit_root):
    root = Path(study_root)
    config = load(root / "configuration.json")
    plan = load(root / "plan.json")
    ledger = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    assert len(ledger) == len(plan)
    expected = {(r["cell"], r["repetition"]): r for r in plan}
    assert len(expected) == len(plan)
    observed = set()
    events, complete, failures = Counter(), Counter(), Counter()
    candidate_count = 0
    for record in ledger:
        key = record["cell"], record["repetition"]
        assert key not in observed
        observed.add(key)
        assert all(record[k] == v for k, v in expected[key].items())
        for purpose, field in (("training_data", "data_seed"), ("heldout_data", "test_seed")):
            payload = json.dumps([config["random_seed"], *key, purpose], separators=(",", ":"))
            seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")
            assert record[field] == seed
        archive = root / record["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == record["archive_sha256"]
        assert archive.stat().st_size == record["size_bytes"]
        manifest_path = root / record["manifest"]
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == record["manifest_sha256"]
        manifest = load(manifest_path)["files"]
        with tarfile.open(archive) as packed:
            members = packed.getmembers()
            assert len(members) == len({m.name for m in members}) == len(manifest)
            for member in members:
                assert member.isfile() and not Path(member.name).is_absolute()
                assert ".." not in Path(member.name).parts
                data = packed.extractfile(member).read()
                assert len(data) == manifest[member.name]["size_bytes"]
                assert hashlib.sha256(data).hexdigest() == manifest[member.name]["sha256"]

            def read(name):
                return json.loads(packed.extractfile(name).read())

            inner = read("record.json")
            assert inner["points"] == record["points"]
            for point in inner["points"]:
                group = record["cell"], point["arm"]
                if point["status"] != "complete":
                    assert point["aic_k"] is None and point["error"]
                    failures[group] += 1
                    continue
                complete[group] += 1
                arm = point["arm"]
                metrics = read(f"{arm}/metrics.json")
                path = f"{arm}/" + ("" if arm == "scores" else "run/")
                candidates = read(path + "candidates.json")
                assert [c["k"] for c in candidates] == [1, 2, 3, 4]
                eligible = []
                one = candidates[0]
                for c in candidates:
                    candidate_count += 1
                    if c["status"] == "insufficient_samples":
                        assert record["n"] < c["k"] * config["model"]["min_cluster_size"]
                        continue
                    k, d = c["k"], 4
                    q = k - 1 + k * d + k * d * (d + 1) // 2
                    ll = c["log_likelihood"]
                    assert math.isclose(c["aic"], -2 * ll + 2 * q, abs_tol=1e-7)
                    assert math.isclose(c["bic"], -2 * ll + math.log(record["n"]) * q, abs_tol=1e-7)
                    admissible = c["converged"] and min(c["distinct_unit_sizes"]) >= 5
                    assert (c["status"] == "eligible") == admissible
                    if admissible:
                        eligible.append(c)
                    metric = next(m for m in metrics["candidates"] if m["k"] == k)
                    assert math.isclose(
                        metric["twice_loglikelihood_gain_over_one"],
                        2 * (ll - one["log_likelihood"]),
                        abs_tol=1e-7,
                    )
                    assert metric["parameters"] == q
                for criterion in ("aic", "bic"):
                    chosen = min(eligible, key=lambda c: (c[criterion], c["k"]))["k"]
                    assert metrics[criterion + "_k"] == point[criterion + "_k"] == chosen
                events[group] += point["aic_k"] > 1
    assert observed == set(expected)
    summary = load(Path(audit_root) / "summary.json")
    assert summary["point_fits"] == sum(complete.values()) + sum(failures.values())
    assert summary["successful_fits"] == sum(complete.values())
    assert summary["failed_fits"] == sum(failures.values())
    assert summary["primary_family_size"] == len(summary["cells"])
    for row in summary["cells"]:
        group = row["cell"], row["arm"]
        rate = row["primary_aic_multiple_components"]
        assert rate["events"] == events[group]
        assert rate["available"] == complete[group]
        assert rate["withheld"] == failures[group]
        assert sum(row["aic_k_counts"].values()) == complete[group]
    return {
        "standard_library_check_passed": True,
        "observations": len(ledger),
        "successful_searches": sum(complete.values()),
        "failed_searches": sum(failures.values()),
        "candidates_checked": candidate_count,
        "no_refits": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.study_root, args.audit_root)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
