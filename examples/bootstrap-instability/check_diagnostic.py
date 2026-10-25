"""Standard-library archive, identity, gate and counter verification, without fitting."""

import argparse
import hashlib
import importlib.util
import io
import json
import math
import struct
from collections import Counter
from itertools import permutations
from pathlib import Path
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location(
    "paired_bytes", Path(__file__).parents[1] / "shared-predictive/check_shared.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def integers(raw, name):
    with ZipFile(io.BytesIO(raw)) as archive:
        header, data = common.common.npy(archive.read(name + ".npy"))
    assert len(header["shape"]) == 1
    endian = ">" if header["descr"][0] == ">" else "<"
    formats = {"i8": "q", "i4": "i", "u8": "Q", "u4": "I"}
    result = [v[0] for v in struct.iter_unpack(endian + formats[header["descr"][1:]], data)]
    assert len(result) == header["shape"][0]
    return result


def check_assignments(files, prefix, diagnostic, reference):
    if diagnostic["selection"]["status"] != "complete":
        assert diagnostic["point_assignment_changes"] is None
        return
    name = (
        "original_predictions.npz"
        if prefix + "/original_predictions.npz" in files
        else "selected_memberships.npz"
    )
    labels = integers(files[prefix + "/" + name], "labels")
    a, b = sorted(set(reference)), sorted(set(labels))
    width = max(len(a), len(b))
    table = [[0] * width for _ in range(width)]
    for x, y in zip(reference, labels, strict=True):
        table[a.index(x)][b.index(y)] += 1
    same = max(
        sum(table[i][j] for i, j in enumerate(order)) for order in permutations(range(width))
    )
    assert diagnostic["point_assignment_changes"] == len(labels) - same


def choose(rows):
    return min(rows, key=lambda c: (c["aic"], c["k"]))["k"]


def verify_gate(files, prefix, diagnostic, indices, minimum):
    def load(name):
        return json.loads(files[prefix + "/" + name])

    original = load("summary.json")["selection"]
    assert diagnostic["selection"] == original
    assert diagnostic["draw_indices"] == indices
    assert diagnostic["original_multiplicities"] == [
        Counter(indices)[i] for i in range(diagnostic["n_original"])
    ]
    assert diagnostic["global_oob"] == sorted(set(range(diagnostic["n_original"])) - set(indices))
    gate = diagnostic["gate"]
    if original["status"] != "complete":
        assert gate["status"] == "unavailable" and gate["selected_k"] is None
        return
    if prefix + "/folds.json" in files:
        folds = load("folds.json")
    else:
        folds = []
        for assignment in load("fold_assignments.json"):
            f = assignment["fold"]
            folds.append(
                dict(
                    fold=f,
                    original_training_indices=[indices[j] for j in assignment["training_indices"]],
                    original_validation_indices=[
                        indices[j] for j in assignment["validation_indices"]
                    ],
                    candidates=load(f"fold_{f:03d}/run/candidates.json"),
                )
            )
    assert len(folds) == 5
    assert sorted(i for f in folds for i in f["original_validation_indices"]) == sorted(indices)
    for f in folds:
        assert set(f["original_training_indices"]).isdisjoint(f["original_validation_indices"])
        assert sorted(f["original_training_indices"] + f["original_validation_indices"]) == sorted(
            indices
        )
    full = load("full/run/candidates.json")
    allowed = []
    for c, row in zip(full, gate["candidates"], strict=True):
        k = c["k"]
        candidates = [c] + [next(v for v in f["candidates"] if v["k"] == k) for f in folds]
        counts = [len(set(indices))] + [len(set(f["original_training_indices"])) for f in folds]
        assert row["full_and_fold_distinct_counts"] == counts
        assert row["full_distinct_feasible"] == (counts[0] >= k * minimum)
        assert row["all_fold_distinct_feasible"] == all(n >= k * minimum for n in counts[1:])
        assert row["full_eligible"] == (c["status"] == "eligible")
        assert row["all_folds_eligible"] == all(v["status"] == "eligible" for v in candidates[1:])
        assert row["full_distinct_sizes"] == c.get("distinct_unit_sizes")
        assert row["fold_distinct_sizes"] == [v.get("distinct_unit_sizes") for v in candidates[1:]]
        fitted = all(v["status"] != "insufficient_samples" for v in candidates)
        gains, distinct_gains = [], []
        if fitted:
            for f in folds:
                raw = files[prefix + f"/fold_{f['fold']:03d}/validation.npz"]
                logs = common.common.floats(raw, "k" + str(k))
                one = common.common.floats(raw, "k1")
                delta = [a - b for a, b in zip(logs, one, strict=True)]
                assert len(delta) == len(f["original_validation_indices"])
                gains.extend(delta)
                seen = {}
                for u, value in zip(f["original_validation_indices"], delta, strict=True):
                    if u in seen:
                        assert math.isclose(value, seen[u], abs_tol=1e-12)
                    else:
                        seen[u] = value
                distinct_gains.extend(seen.values())
            gain, equal = (
                math.fsum(gains) / len(gains),
                math.fsum(distinct_gains) / len(distinct_gains),
            )
            assert math.isclose(row["pooled_validation_gain"], gain, abs_tol=1e-11)
            assert math.isclose(row["equal_original_validation_gain"], equal, abs_tol=1e-11)
        else:
            gain = equal = None
            assert row["pooled_validation_gain"] is None
        positive = k == 1 or gain is not None and gain > 0
        passed = row["full_eligible"] and row["all_folds_eligible"] and positive
        assert row["passes"] == passed
        first = (
            "distinct_feasibility"
            if min(counts) < k * minimum
            else "full_eligibility"
            if not row["full_eligible"]
            else "fold_eligibility"
            if not row["all_folds_eligible"]
            else "nonpositive_gain"
            if not positive
            else "selected"
            if k == original["screened_k"]
            else "aic_ranking"
        )
        assert row["first_failing_gate"] == first
        if passed:
            allowed.append(c)
    assert gate["selected_k"] == original["screened_k"] == choose(allowed)
    for policy, result in gate["counterfactuals"].items():
        missing, allow = [], []
        for c, row in zip(full, gate["candidates"], strict=True):
            if (
                policy in {"remove_full_size", "remove_fold_size", "occurrence_sizes"}
                and not row["fitted_in_all_searches"]
            ):
                missing.append(c["k"])
                continue
            a = row["full_eligible"] or policy == "remove_full_size"
            b = row["all_folds_eligible"] or policy == "remove_fold_size"
            gain = (
                row["equal_original_validation_gain"]
                if policy == "equal_validation_weights"
                else row["pooled_validation_gain"]
            )
            if policy == "occurrence_sizes":
                a = min(row["full_hard_sizes"]) >= minimum
                b = all(min(v) >= minimum for v in row["fold_hard_sizes"])
            if (
                a
                and b
                and (c["k"] == 1 or policy == "remove_gain" or gain is not None and gain > 0)
            ):
                allow.append(c)
        if missing:
            assert result == dict(status="unavailable", missing_candidates=missing, selected_k=None)
        else:
            assert result == dict(status="available", selected_k=choose(allow))


def verify(study):
    root = Path(study)
    config = json.loads((root / "configuration.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    rows = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    expected = {(r["cell"], r["repetition"]): r for r in plan}
    assert len(expected) == len(rows)
    searches = oracles = 0
    seen = set()
    for row in rows:
        identity = row["cell"], row["repetition"]
        assert identity not in seen
        seen.add(identity)
        assert all(row[k] == v for k, v in expected[identity].items())
        for field, purpose in (
            ("data_seed", "training_data"),
            ("split_seed", "fold_split"),
            ("draw_stream_seed", "bootstrap"),
        ):
            payload = json.dumps(
                [config["random_seed"], *identity, purpose], separators=(",", ":")
            ).encode()
            assert row[field] == int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        files = common.unpack(
            (root / row["archive"]).read_bytes(), (root / row["manifest"]).read_bytes(), row
        )

        def load(name, files=files):
            return json.loads(files[name])

        draws = load("draws.json")
        assert len(draws) == config["n_draws"]
        for draw in draws:
            assert draw["support"] == list(dict.fromkeys(draw["indices"]))
        point_reference = (
            integers(files["baseline/point/selected_memberships.npz"], "labels")
            if "baseline/point/selected_memberships.npz" in files
            else None
        )
        workflows = load("workflows.json")
        assert len(workflows) == 2 + 2 * config["n_draws"] + config["extended_draws"]
        for w in workflows:
            arm, i = w["arm"], w["replicate"]
            indices = (
                list(range(row["cell_config"]["n_units"]))
                if i == -1
                else draws[i]["support" if arm == "unique" else "indices"]
            )
            assert indices == w["indices"]
            name = "point" if i == -1 else f"draw_{i:04d}"
            verify_gate(
                files,
                w["prefix"],
                load(f"diagnostics/{arm}/{name}.json"),
                indices,
                config["model"]["min_cluster_size"],
            )
            if point_reference is not None:
                check_assignments(
                    files, w["prefix"], load(f"diagnostics/{arm}/{name}.json"), point_reference
                )
            summary = load(w["prefix"] + "/summary.json")
            assert summary["attempted_fits"] + summary["unattempted_fits"] == 6
            searches += summary["attempted_fits"]
        oracle_rows = load("oracles.json")
        assert len(oracle_rows) == row["planned_oracles"]
        for oracle in oracle_rows:
            entry = load(f"oracles/{oracle['workflow']}/{oracle['prefix']}/diagnostic.json")
            assert entry == {k: v for k, v in oracle.items() if k != "workflow"}
            if oracle["status"] == "complete":
                labels = integers(
                    files[f"oracles/{oracle['workflow']}/{oracle['prefix']}/mixture.npz"], "labels"
                )
                sizes = [
                    len(
                        {
                            u
                            for u, k in zip(oracle["original_indices"], labels, strict=True)
                            if k == j
                        }
                    )
                    for j in range(4)
                ]
                assert sizes == oracle["distinct_unit_sizes"]
                assert oracle["eligible_original_ids"] == (
                    min(oracle["distinct_unit_sizes"]) >= config["model"]["min_cluster_size"]
                )
            else:
                assert oracle["error"]
        oracles += len(oracle_rows)
    summary = json.loads((root / "summary.json").read_text())
    assert searches == summary["attempted_fits"]
    assert oracles == summary["planned_oracles"]
    return dict(
        status="verified",
        observations=len(rows),
        audited_searches=searches,
        audited_oracles=oracles,
        scope="Byte/identity/fold/gate verification without numerical fitting",
    )


def verify_archival(study, source):
    root, original = Path(study), Path(source)
    rows = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    source_rows = {
        (r["cell"], r["repetition"]): r
        for r in map(json.loads, (original / "observations.jsonl").read_text().splitlines())
    }
    config = json.loads((original / "configuration.json").read_text())
    assert len(rows) == len(source_rows)
    workflows = 0
    for row in rows:
        source_row = source_rows[row["cell"], row["repetition"]]
        files = common.unpack(
            (original / source_row["archive"]).read_bytes(),
            (original / source_row["manifest"]).read_bytes(),
            source_row,
        )
        derived = common.unpack(
            (root / row["archive"]).read_bytes(), (root / row["manifest"]).read_bytes(), row
        )
        assert json.loads(derived["source.json"])["source_archive"] == source_row
        selected = json.loads(files["workflow/summary.json"])
        point_reference = integers(files["workflow/point/selected_memberships.npz"], "labels")
        n = row["cell_config"]["n_units"]
        verify_gate(
            files,
            "workflow/point",
            json.loads(derived["point.json"]),
            list(range(n)),
            config["model"]["min_cluster_size"],
        )
        for record in selected["replicates"]:
            name = record["prefix"]
            diagnostic = json.loads(derived[name + ".json"])
            verify_gate(
                files,
                "workflow/" + name,
                diagnostic,
                record["training_indices"],
                config["model"]["min_cluster_size"],
            )
            check_assignments(files, "workflow/" + name, diagnostic, point_reference)
            workflows += 1
        workflows += 1
    assert workflows == len(rows) * (1 + config["n_bootstrap"])
    return dict(
        status="verified",
        observations=len(rows),
        draws=workflows - len(rows),
        verified_workflows=workflows,
        refits=0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path)
    args = p.parse_args()
    result = verify_archival(args.study, args.source) if args.source else verify(args.study)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
