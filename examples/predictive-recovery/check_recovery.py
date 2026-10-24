"""Standard-library verification of new archives, counterfactuals, budgets and rates."""

import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location(
    "predictive_bytes_checker", Path(__file__).parents[1] / "predictive-aic/check_screening.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def seed(master, cell, rep, purpose):
    raw = json.dumps([master, cell, rep, purpose], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def criteria(candidates, raw_pool, minimum):
    with ZipFile(io.BytesIO(raw_pool)) as p:
        header, _ = common.npy(p.read("scores.npy"))
    n, d = header["shape"]
    for c in candidates:
        k = c["k"]
        if c["status"] == "insufficient_samples":
            assert n < k * minimum
            continue
        q = k - 1 + k * d + k * d * (d + 1) // 2
        assert math.isclose(c["aic"], -2 * c["log_likelihood"] + 2 * q, abs_tol=1e-7)
        assert math.isclose(c["bic"], -2 * c["log_likelihood"] + math.log(n) * q, abs_tol=1e-7)
        assert sum(c["distinct_unit_sizes"]) == n
        assert (c["status"] == "eligible") == (
            c["converged"] and min(c["distinct_unit_sizes"]) >= minimum
        )


def part(z, name, row, config):
    def data(relative):
        return z.extractfile(name + "/" + relative).read()

    def load(relative):
        return json.loads(data(relative))

    summary = load("workflow/summary.json")
    selection = summary["selection"]
    full = load("workflow/full/run/candidates.json")
    units = load("workflow/full/run/summary.json")["unit_ids"]
    assignments = load("workflow/fold_assignments.json")
    assert len(assignments) == 5
    assert sorted(j for f in assignments for j in f["validation_indices"]) == list(
        range(len(units))
    )
    for f in assignments:
        assert set(f["training_indices"]).isdisjoint(f["validation_indices"])
        assert sorted(f["training_indices"] + f["validation_indices"]) == list(range(len(units)))
        for key in ("training", "validation"):
            assert f[key + "_unit_ids"] == [units[j] for j in f[key + "_indices"]]
    decisions = {rule: [] for rule in ("ordinary", "eligibility_only", "gain_only", "actual")}
    reported = load("diagnostics/metrics.json")["gates"]
    for c in full:
        k = c["k"]
        full_eligible = c["status"] == "eligible"
        every = True
        gains = []
        for f in assignments:
            prefix = f"workflow/fold_{f['fold']:03d}"
            pool = load(prefix + "/run/candidates.json")
            current = next(v for v in pool if v["k"] == k)
            every = every and current["status"] == "eligible"
            if current["status"] != "insufficient_samples":
                a, b = (
                    common.floats(data(prefix + "/validation.npz"), key)
                    for key in ("k" + str(k), "k1")
                )
                assert len(a) == len(b) == len(f["validation_indices"])
                gains.extend(x - y for x, y in zip(a, b, strict=True))
        gain = math.fsum(gains) / len(units) if len(gains) == len(units) else None
        positive = gain is not None and gain > 0
        flags = {
            "ordinary": full_eligible,
            "eligibility_only": full_eligible and every,
            "gain_only": full_eligible and (k == 1 or positive),
            "actual": full_eligible and every and (k == 1 or positive),
        }
        diagnosis = next(v for v in reported["candidates"] if v["k"] == k)
        assert diagnosis["passes"] == flags
        if gain is None:
            assert diagnosis["pooled_gain_all_fitted_folds"] is None
        else:
            assert math.isclose(diagnosis["pooled_gain_all_fitted_folds"], gain, abs_tol=1e-10)
        for rule, yes in flags.items():
            if yes:
                decisions[rule].append(c)
    for rule, cs in decisions.items():
        assert reported[rule + "_k"] == min(cs, key=lambda c: (c["aic"], c["k"]))["k"]
    assert selection["ordinary_aic_k"] == reported["ordinary_k"]
    assert selection["screened_k"] == reported["actual_k"]
    for prefix in ["full"] + [f"fold_{i:03d}" for i in range(5)]:
        criteria(
            load("workflow/" + prefix + "/run/candidates.json"),
            data("workflow/" + prefix + "/run/gmm_candidates.npz"),
            config["study"]["model"]["min_cluster_size"],
        )
    assert row["selection"] == selection and row["gates"] == reported
    assert row["performance"] == load("performance.json")
    for method, key in (
        ("ordinary", "ordinary_aic_k"),
        ("screened", "screened_k"),
        ("bic", "descriptive_bic_k"),
    ):
        assert row["performance"][method]["k"] == selection[key]
    assert "workflow/selected_model.npz" in [
        n[len(name) + 1 :] for n in z.getnames() if n.startswith(name + "/")
    ]
    return row["performance"]


def verify(study, audit):
    root = Path(study)
    config = common.read(root / "configuration.json")
    plan = common.read(root / "plan.json")
    rows = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    expected = {(r["cell"], r["repetition"]): r for r in plan}
    seen = set()
    selections = {}
    counts = {}
    for row in rows:
        key = row["cell"], row["repetition"]
        assert key not in seen
        seen.add(key)
        assert all(row[k] == v for k, v in expected[key].items())
        master = config["study"]["random_seed"]
        for field, purpose in (
            ("data_seed", "training_data"),
            ("test_seed", "external_data"),
            ("split_seed", "fold_split"),
        ):
            assert row[field] == seed(master, *key, purpose)
        assert row["partition_seeds"] == [row["split_seed"]] + [
            seed(master, *key, f"extra_split_{j}") for j in range(1, config["partitions"])
        ]
        with common.verified_archive(root, row) as z:
            inner = json.load(z.extractfile("record.json"))
            assert all(row[k] == v for k, v in inner.items())
            first = {}
            for p in row["parts"]:
                name = p["name"]
                reused = p["shared_full_search"]
                files = {
                    m.name[len(name + "/workflow/full/") :]: hashlib.sha256(
                        z.extractfile(m).read()
                    ).hexdigest()
                    for m in z.getmembers()
                    if m.name.startswith(name + "/workflow/full/")
                }
                if name == "part_000":
                    first = files
                if reused:
                    assert files == first
                records = [
                    json.load(z.extractfile(m))
                    for m in z.getmembers()
                    if m.name.startswith(name + "/workflow/")
                    and m.name.count("/") == 3
                    and m.name.endswith("/record.json")
                    and not (reused and "/full/" in m.name)
                ]
                for k, v in {
                    "attempted_searches": len(records),
                    "successful_searches": sum(r["status"] == "complete" for r in records),
                    "failed_searches": sum(r["status"] != "complete" for r in records),
                }.items():
                    counts[k] = counts.get(k, 0) + v
                if p["selection"]["status"] == "complete":
                    performance = part(z, name, p, config)
                else:
                    assert p["selection"].get("screened_k") is None
                    performance = p["performance"]
                    assert performance["screened"] is None
                if name in {"part_000", "part_extended"}:
                    selections[key, name] = performance
            oracles = [
                json.load(z.extractfile(m))
                for m in z.getmembers()
                if m.name.startswith("oracles/") and m.name.endswith("/result.json")
            ]
            assert len(oracles) == row["attempted_oracles"]
            for oracle in oracles:
                assert oracle["attempted"] and oracle["k"] == 4 and oracle["partition"] == "four"
                assert "never selection" in oracle["role"]
            for k, v in {
                "attempted_oracles": len(oracles),
                "successful_oracles": sum(o["status"] == "complete" for o in oracles),
                "failed_oracles": sum(o["status"] != "complete" for o in oracles),
            }.items():
                counts[k] = counts.get(k, 0) + v
    assert seen == set(expected) and len(rows) == len(plan)
    result = common.read(Path(audit) / "summary.json")
    for k, v in counts.items():
        assert result[k] == v
    assert (
        len(result["primary_rates"])
        == result["primary_family_size"]
        == 4 * len(config["study"]["cells"])
    )
    for rate in result["primary_rates"]:
        arm, method = rate["method"].split("_")
        name = "part_000" if arm == "baseline" else "part_extended"
        group = [
            v[name][method]
            for key in seen
            if key[0] == rate["cell"]
            for v in [{name: selections[key, name]}]
        ]
        good = [v for v in group if v is not None]
        events = sum(v["k"] == 4 if rate["event"] == "K=4" else v["k"] > 1 for v in good)
        assert (rate["events"], rate["available"], rate["requested"]) == (
            events,
            len(good),
            len(group),
        )
    return {
        "standard_library_check_passed": True,
        "observations": len(rows),
        **counts,
        "primary_rates": len(result["primary_rates"]),
        "no_refits": True,
        "shared_full_fit_and_counterfactuals_checked": True,
        "oracle_separation_checked": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.study_root, args.audit_root)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
