"""Independent standard-library audit of archive bytes, folds, gates and stability counts."""

import argparse
import hashlib
import importlib.util
import io
import json
import math
import struct
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location(
    "paired_bytes", Path(__file__).parents[1] / "shared-predictive/check_shared.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def array(raw, key):
    with ZipFile(io.BytesIO(raw)) as z:
        header, payload = common.common.npy(z.read(key + ".npy"))
    dtype = header["descr"]
    fmt = {"i8": "q", "u8": "Q", "f8": "d", "i4": "i", "u4": "I"}[dtype[1:]]
    assert not header["fortran_order"]
    values = [v[0] for v in struct.iter_unpack((">" if dtype[0] == ">" else "<") + fmt, payload)]
    assert len(values) == math.prod(header["shape"])
    return header["shape"], values


def derive(*args):
    return int.from_bytes(
        hashlib.sha256(json.dumps(args, separators=(",", ":")).encode()).digest()[:8], "little"
    )


def verify_screen(files, prefix, model, indices=None, assignments=None):
    def load(name):
        return json.loads(files[prefix + "/" + name])

    result = load("summary.json")
    records = result["searches"]
    assert result["attempted_fits"] == len(records)
    assert result["successful_fits"] == sum(r["status"] == "complete" for r in records)
    assert result["failed_fits"] == sum(r["status"] != "complete" for r in records)
    assert result["unattempted_fits"] == 6 - len(records)
    if records[0]["status"] != "complete":
        assert result["selection"]["screened_k"] is None
        return result
    pools = {}
    for record in records:
        name = record.get("archive_prefix", record.get("prefix"))
        if record["status"] != "complete":
            assert record["error"]
            continue
        pool = files[prefix + "/" + name + "/run/gmm_candidates.npz"]
        shape, _ = array(pool, "scores")
        n, d = shape
        original = (
            list(range(n))
            if indices is None
            else (indices if name == "full" else load(name + "/draw.json")["original_indices"])
        )
        assert len(original) == n
        candidates = load(name + "/run/candidates.json")
        pools[name] = candidates
        for c in candidates:
            k = c["k"]
            if c["status"] == "insufficient_samples":
                assert len(set(original)) < k * model["min_cluster_size"]
                continue
            _, labels = array(pool, f"k{k}_labels")
            sizes = [labels.count(j) for j in range(k)]
            unique = [
                len({u for u, label in zip(original, labels, strict=True) if label == j})
                for j in range(k)
            ]
            assert sizes == c["hard_cluster_sizes"] and unique == c["distinct_unit_sizes"]
            assert (c["status"] == "eligible") == (
                c["converged"] and min(unique) >= model["min_cluster_size"]
            )
            q = (
                k
                - 1
                + k * d
                + (1 if model.get("covariance_type", "full") == "tied" else k) * d * (d + 1) // 2
            )
            assert math.isclose(c["aic"], -2 * c["log_likelihood"] + 2 * q, abs_tol=1e-7)
            assert math.isclose(c["bic"], -2 * c["log_likelihood"] + math.log(n) * q, abs_tol=1e-7)
    folds = []
    retained = load("folds.json") if indices is not None else None
    failed = False
    for i in range(5):
        name = f"fold_{i:03d}"
        if retained is None:
            row = load(name + "/validation_record.json")
            if row["status"] != "complete":
                failed = True
                continue
            candidates = pools[name]
            test = row["validation_indices"]
        else:
            row = retained[i]
            test = [j for j, u in enumerate(indices) if u in assignments[i]["validation_indices"]]
            train = sorted(set(range(len(indices))) - set(test))
            assert row["training_indices"] == train and row["validation_indices"] == test
            assert row["original_training_indices"] == [indices[j] for j in train]
            assert row["original_validation_indices"] == [indices[j] for j in test]
            assert not set(row["original_training_indices"]) & set(
                row["original_validation_indices"]
            )
            if row["status"] != "complete":
                assert row["error"]
                failed = True
                continue
            candidates = row["candidates"]
        raw = files[prefix + "/" + name + "/validation.npz"]
        density = {
            str(c["k"]): array(raw, "k" + str(c["k"]))[1]
            for c in candidates
            if c["status"] != "insufficient_samples"
        }
        assert all(
            len(v) == len(test) and all(math.isfinite(x) for x in v) for v in density.values()
        )
        folds.append(dict(candidates=candidates, density=density))
    if failed:
        assert result["selection"]["status"] == "withheld"
        return result
    valid = [c for c in pools["full"] if c["status"] == "eligible"]
    allow = []
    decisions = []
    for c in pools["full"]:
        k = c["k"]
        states = [next(v["status"] for v in f["candidates"] if v["k"] == k) for f in folds]
        gain = None
        if c["status"] == "eligible" and all(s == "eligible" for s in states):
            gain = math.fsum(
                a - b
                for f in folds
                for a, b in zip(f["density"][str(k)], f["density"]["1"], strict=True)
            ) / sum(len(f["density"]["1"]) for f in folds)
        passes = k == 1 or (gain is not None and gain > 0)
        decisions.append((k, gain, passes, states))
        if passes:
            allow.append(c)
    actual = result["selection"]
    assert actual["ordinary_aic_k"] == min(valid, key=lambda c: (c["aic"], c["k"]))["k"]
    assert actual["screened_k"] == min(allow, key=lambda c: (c["aic"], c["k"]))["k"]
    for (k, gain, passes, states), row in zip(decisions, actual["candidates"], strict=True):
        assert (k, passes, states) == (row["k"], row["passes"], row["fold_statuses"])
        assert gain is row["pooled_validation_gain"] or (
            gain is not None and math.isclose(gain, row["pooled_validation_gain"], abs_tol=1e-12)
        )
    return result


def verify(study):
    root = Path(study)
    config = json.loads((root / "configuration.json").read_text())
    rows = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    plan = json.loads((root / "plan.json").read_text())
    lookup = {(r["cell"], r["repetition"]): r for r in plan}
    seen = set()
    searches = draws = failed = 0
    for outer in rows:
        identity = outer["cell"], outer["repetition"]
        assert identity not in seen
        seen.add(identity)
        assert all(outer[k] == v for k, v in lookup[identity].items())
        for key, purpose in (
            ("data_seed", "training_data"),
            ("split_seed", "fold_split"),
            ("bootstrap_seed", "bootstrap"),
        ):
            assert outer[key] == derive(config["random_seed"], *identity, purpose)
        files = common.unpack(
            (root / outer["archive"]).read_bytes(), (root / outer["manifest"]).read_bytes(), outer
        )
        result = json.loads(files["workflow/summary.json"])
        point = verify_screen(files, "workflow/point", config["model"])
        assignments = json.loads(files["workflow/point/fold_assignments.json"])
        n = len(result["unit_ids"])
        assert sorted(j for a in assignments for j in a["validation_indices"]) == list(range(n))
        for a in assignments:
            assert sorted(a["training_indices"] + a["validation_indices"]) == list(range(n))
            assert not set(a["training_indices"]) & set(a["validation_indices"])
        counts = [0] * (n * n)
        oobcounts = counts.copy()
        requested = counts.copy()
        success = counts.copy()
        kpoint = point["selection"]["screened_k"]
        _, pointlabels = array(files["workflow/point/selected_memberships.npz"], "labels")
        groups = [{j for j, label in enumerate(pointlabels) if label == k} for k in range(kpoint)]
        exact = [0] * kpoint
        ks = Counter()
        successful = 0
        searches += 6
        for i, row in enumerate(result["replicates"]):
            draw = row["training_indices"]
            draws += 1
            assert row["replicate"] == i and row["draw_seed"] == derive(
                outer["bootstrap_seed"], "predictive_locality_bootstrap_v1", i, "draw"
            )
            assert len(draw) == n and all(type(j) is int and 0 <= j < n for j in draw)
            assert row["multiplicities"] == [draw.count(j) for j in range(n)]
            missing = set(range(n)) - set(draw)
            assert sorted(missing) == row["oob_indices"]
            for a in missing:
                for b in missing:
                    requested[a * n + b] += 1
            record = verify_screen(
                files, "workflow/" + row["prefix"], config["model"], draw, assignments
            )
            searches += record["attempted_fits"]
            if row["status"] != "complete":
                failed += 1
                continue
            labels = row["labels"]
            k = row["selected_k"]
            ks[k] += 1
            successful += 1
            assert len(labels) == n and all(0 <= j < k for j in labels)
            _, predicted = array(
                files["workflow/" + row["prefix"] + "/original_predictions.npz"], "labels"
            )
            assert labels == predicted
            for a in range(n):
                for b in range(n):
                    index = a * n + b
                    same = labels[a] == labels[b]
                    counts[index] += same
                    if a in missing and b in missing:
                        oobcounts[index] += same
                        success[index] += 1
            for j, group in enumerate(groups):
                exact[j] += any(
                    group == {u for u, label in enumerate(labels) if label == v} for v in range(k)
                )
        raw = files["workflow/stability.npz"]
        for key, expected in [
            ("all_counts", counts),
            ("oob_counts", oobcounts),
            ("oob_requested", requested),
            ("oob_successful", success),
            ("exact_cluster_counts", exact),
        ]:
            _, actual = array(raw, key)
            assert actual == expected
        _, actual = array(raw, "selected_k_counts")
        assert actual == [ks[k] for k in range(1, len(actual) + 1)]
        for name, numerators, denominators, completed, threshold in (
            ("all", counts, [len(result["replicates"])] * (n * n), [successful] * (n * n), 1),
            ("oob", oobcounts, requested, success, 10),
        ):
            _, freq = array(raw, name + "_frequency")
            _, conditional = array(raw, name + "_conditional_frequency")
            _, low = array(raw, name + "_failure_lower")
            _, high = array(raw, name + "_failure_upper")
            for j, (num, den, done) in enumerate(
                zip(numerators, denominators, completed, strict=True)
            ):
                if den:
                    assert math.isclose(low[j], num / den, abs_tol=1e-14)
                    assert math.isclose(high[j], (num + den - done) / den, abs_tol=1e-14)
                else:
                    assert math.isnan(low[j]) and math.isnan(high[j])
                if done >= threshold:
                    assert math.isclose(conditional[j], num / done, abs_tol=1e-14)
                else:
                    assert math.isnan(conditional[j])
                if den == done and done >= threshold:
                    assert math.isclose(freq[j], num / den, abs_tol=1e-14)
                else:
                    assert math.isnan(freq[j])
        for j, cluster in enumerate(result["stability"]["clusters"]):
            assert cluster["exact_recovery_count"] == exact[j]
            for name, mask in [
                ("within_cohesion", [(a, b) for a in groups[j] for b in groups[j] if a != b]),
                (
                    "cross_merging",
                    [(a, b) for a in groups[j] for b in range(n) if b not in groups[j]],
                ),
            ]:
                _, freq = array(raw, "all_frequency")
                vals = [freq[a * n + b] for a, b in mask if math.isfinite(freq[a * n + b])]
                actual = cluster["all_" + name]
                assert (actual is None and not vals) or math.isclose(
                    actual, math.fsum(vals) / len(vals), abs_tol=1e-12
                )
    assert seen == lookup.keys()
    return dict(
        status="passed",
        observations=len(rows),
        audited_searches=searches,
        draws=draws,
        failed_draws=failed,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("--output-file", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.study)
    with args.output_file.open("x") as h:
        json.dump(result, h, indent=2, sort_keys=True)
        h.write("\n")
    print(json.dumps(result, sort_keys=True))
