"""Standard-library verification of every retained byte, fold, threshold and gate."""

import argparse
import hashlib
import importlib.util
import io
import json
import math
import struct
import tarfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location(
    "screening_bytes", Path(__file__).parents[1] / "predictive-aic/check_screening.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def integers(npz, name):
    with ZipFile(io.BytesIO(npz)) as z:
        header, raw = common.npy(z.read(name + ".npy"))
    code = {"i8": "q", "i4": "i", "u8": "Q", "u4": "I"}[header["descr"][1:]]
    result = [
        v[0] for v in struct.iter_unpack((">" if header["descr"][0] == ">" else "<") + code, raw)
    ]
    assert len(result) == math.prod(header["shape"])
    return result


def wanted(name):
    if name in {"declaration.json", "record.json", "generating_layout.json"}:
        return True
    parts = name.split("/")
    if parts[0] not in {"legacy", "scaled"}:
        return False
    if len(parts) == 2:
        return (
            parts[1] in {"summary.json", "stability.npz"}
            or parts[1].startswith("draw_")
            and parts[1].endswith(".json")
        )
    return parts[-1] in {
        "candidates.json",
        "model.json",
        "fold_support.json",
        "mixture_settings.json",
        "gmm_candidates.npz",
        "validation.npz",
        "selected_memberships.npz",
        "original_predictions.npz",
        "summary.json",
        "fold_assignments.json",
        "folds.json",
        "record.json",
    }


def contents(root, row):
    archive = root / row["archive"]
    manifest = root / row["manifest"]
    assert archive.stat().st_size == row["size_bytes"]

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                h.update(block)
        return h.hexdigest()

    assert digest(archive) == row["archive_sha256"] and digest(manifest) == row["manifest_sha256"]
    expected = json.loads(manifest.read_text())["files"]
    assert (
        len(expected) == row["files"]
        and sum(v["size_bytes"] for v in expected.values()) == row["unpacked_bytes"]
    )
    files = {}
    seen = set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = member.name
            assert member.isfile() and name not in seen and name in expected
            assert (
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and Path(name).as_posix() == name
            )
            assert member.size == expected[name]["size_bytes"]
            h = hashlib.sha256()
            blocks = []
            retain = wanted(name)
            handle = stream.extractfile(member)
            while block := handle.read(1024 * 1024):
                h.update(block)
                if retain:
                    blocks.append(block)
            assert h.hexdigest() == expected[name]["sha256"]
            if retain:
                files[name] = b"".join(blocks)
            seen.add(name)
    assert seen == set(expected)
    return files


def select(candidates):
    return min(candidates, key=lambda c: (c["aic"], c["k"]))["k"]


def workflow(files, prefix, indices, assignments, model, scaled, point=False):
    def load(name):
        return json.loads(files[prefix + "/" + name])

    if prefix + "/summary.json" not in files:
        if point:
            record = json.loads(files[prefix.split("/")[0] + "/summary.json"])["point"]["selection"]
        else:
            record = json.loads(files[prefix + ".json"])
        assert record["status"] != "complete" and record["error"]
        return sum(
            prefix + f"/{p}/record.json" in files
            for p in ["full"] + [f"fold_{i:03d}" for i in range(5)]
        )
    summary = load("summary.json")
    selection = summary["selection"]
    if selection["status"] != "complete":
        assert selection.get("screened_k") is None and selection.get("error")
        return sum(
            prefix + f"/{p}/record.json" in files
            for p in ["full"] + [f"fold_{i:03d}" for i in range(5)]
        )
    assert load("full/model.json") == model
    assert ("fold_size_policy" in summary) == scaled
    folds = []
    full = load("full/run/candidates.json")
    for assignment in assignments:
        i = assignment["fold"]
        part = f"fold_{i:03d}"
        valid = set(assignment["validation_indices"])
        train = [j for j, u in enumerate(indices) if u not in valid]
        test = [j for j, u in enumerate(indices) if u in valid]
        assert (
            test and train and set(indices[j] for j in train).isdisjoint(indices[j] for j in test)
        )
        if not point:
            row = next(r for r in load("folds.json") if r["fold"] == i)
            assert row["training_indices"] == train and row["validation_indices"] == test
            assert row["original_training_indices"] == [indices[j] for j in train]
            assert row["original_validation_indices"] == [indices[j] for j in test]
        u, t = len(set(indices)), len({indices[j] for j in train})
        m = model["min_cluster_size"]
        minimum = max(2, -(-m * t // u)) if scaled else m
        if scaled:
            e = load(part + "/fold_support.json")
            assert [
                e[k]
                for k in ("full_distinct", "training_distinct", "full_minimum", "training_minimum")
            ] == [u, t, m, minimum]
            assert e["schema_version"] == 1 and e["policy"] == "proportional_distinct"
            effective = load(part + "/run/mixture_settings.json")
            assert effective["min_cluster_size"] == minimum
            for key in (
                "max_k",
                "n_init",
                "max_iter",
                "tol",
                "reg_covar",
                "selection_criterion",
                "max_working_bytes",
            ):
                assert effective[key] == model[key]
        else:
            assert prefix + "/" + part + "/fold_support.json" not in files
        folds.append(
            dict(
                part=part,
                indices=[indices[j] for j in train],
                test=test,
                minimum=minimum,
                candidates=load(part + "/run/candidates.json"),
            )
        )
    for part, original, minimum, candidates in [
        ("full", indices, model["min_cluster_size"], full)
    ] + [(f["part"], f["indices"], f["minimum"], f["candidates"]) for f in folds]:
        pool = files[prefix + "/" + part + "/run/gmm_candidates.npz"]
        with ZipFile(io.BytesIO(pool)) as z:
            header, _ = common.npy(z.read("scores.npy"))
        n, d = header["shape"]
        assert n == len(original)
        for c in candidates:
            k = c["k"]
            if c["status"] == "insufficient_samples":
                assert len(set(original)) < k * minimum
                continue
            assert len(set(original)) >= k * minimum and c["converged"]
            labels = integers(pool, f"k{k}_labels")
            assert len(labels) == len(original)
            hard = [sum(v == j for v in labels) for j in range(k)]
            distinct = [
                len({u for u, label in zip(original, labels, strict=True) if label == j})
                for j in range(k)
            ]
            assert hard == c["hard_cluster_sizes"] and distinct == c["distinct_unit_sizes"]
            assert (c["status"] == "eligible") == (min(distinct) >= minimum)
            q = k - 1 + k * d + d * (d + 1) // 2
            assert math.isclose(c["aic"], -2 * c["log_likelihood"] + 2 * q, abs_tol=1e-7)
            assert math.isclose(c["bic"], -2 * c["log_likelihood"] + math.log(n) * q, abs_tol=1e-7)
    passing = []
    for c in full:
        k = c["k"]
        eligible = c["status"] == "eligible" and all(
            next(v for v in f["candidates"] if v["k"] == k)["status"] == "eligible" for f in folds
        )
        gain = None
        if eligible:
            values = []
            for f in folds:
                raw = files[prefix + "/" + f["part"] + "/validation.npz"]
                a, b = common.floats(raw, "k" + str(k)), common.floats(raw, "k1")
                assert len(a) == len(b) == len(f["test"])
                values.extend(x - y for x, y in zip(a, b, strict=True))
            gain = math.fsum(values) / len(indices)
        actual = next(v for v in selection["candidates"] if v["k"] == k)
        passed = k == 1 or gain is not None and gain > 0
        assert actual["passes"] == passed
        if gain is None:
            assert actual["pooled_validation_gain"] is None
        else:
            assert math.isclose(actual["pooled_validation_gain"], gain, abs_tol=1e-11)
        if passed:
            passing.append(c)
    assert selection["screened_k"] == select(passing)
    eligible = [c for c in full if c["status"] == "eligible"]
    assert selection["ordinary_aic_k"] == select(eligible)
    return 6


def verify(study):
    root = Path(study)
    config = json.loads((root / "configuration.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    rows = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    lookup = {(r["cell"], r["repetition"]): r for r in plan}
    assert len(lookup) == len(rows)
    seen = set()
    checked = draws = 0
    for row in rows:
        identity = row["cell"], row["repetition"]
        assert identity not in seen
        seen.add(identity)
        assert all(row[k] == v for k, v in lookup[identity].items())
        for key, purpose in (
            ("data_seed", "training_data"),
            ("split_seed", "fold_split"),
            ("bootstrap_seed", "bootstrap"),
        ):
            raw = json.dumps(
                [config["random_seed"], *identity, purpose], separators=(",", ":")
            ).encode()
            assert row[key] == int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")
        files = contents(root, row)

        def load(name, files=files):
            return json.loads(files[name])

        n = row["cell_config"]["n_units"]
        candidates = [
            load(arm + "/point/fold_assignments.json")
            for arm in ("legacy", "scaled")
            if arm + "/point/fold_assignments.json" in files
        ]
        assignments = candidates[0] if candidates else None
        if assignments is not None:
            assert sorted(j for f in assignments for j in f["validation_indices"]) == list(range(n))
            assert all(value == assignments for value in candidates)
        pair = []
        for arm in ("legacy", "scaled"):
            result = load(arm + "/summary.json")
            checked += workflow(
                files,
                arm + "/point",
                list(range(n)),
                assignments,
                config["model"],
                arm == "scaled",
                True,
            )
            records = result["replicates"]
            pair.append(records)
            for i, record in enumerate(records):
                payload = json.dumps(
                    [row["bootstrap_seed"], "predictive_locality_bootstrap_v1", i, "draw"],
                    separators=(",", ":"),
                ).encode()
                assert record["draw_seed"] == int.from_bytes(
                    hashlib.sha256(payload).digest()[:8], "little"
                )
                ids = record["training_indices"]
                assert len(ids) == n and all(type(v) is int and 0 <= v < n for v in ids)
                counts = Counter(ids)
                assert record["multiplicities"] == [counts[j] for j in range(n)]
                assert record["oob_indices"] == [j for j in range(n) if not counts[j]]
                checked += workflow(
                    files,
                    arm + "/" + record["prefix"],
                    ids,
                    assignments,
                    config["model"],
                    arm == "scaled",
                )
                if record["status"] == "complete":
                    assert (
                        integers(
                            files[arm + "/" + record["prefix"] + "/original_predictions.npz"],
                            "labels",
                        )
                        == record["labels"]
                    )
            draws += len(records)
            if arm + "/stability.npz" in files:
                pool = files[arm + "/stability.npz"]
                ks = Counter(r["selected_k"] for r in records if r["status"] == "complete")
                selected = integers(pool, "selected_k_counts")
                assert selected == [ks[k] for k in range(1, len(selected) + 1)]
                assert all(v == len(records) for v in integers(pool, "selected_k_requested"))
                assert all(v == sum(ks.values()) for v in integers(pool, "selected_k_successful"))
        for records in pair:
            assert len(records) <= config["n_bootstrap"]
        for a, b in zip(
            pair[0][: min(map(len, pair))], pair[1][: min(map(len, pair))], strict=True
        ):
            for key in ("training_indices", "multiplicities", "draw_seed", "oob_indices"):
                assert a[key] == b[key]
    summary = json.loads((root / "summary.json").read_text())
    assert checked == summary["attempted_fits"]
    return dict(
        status="verified",
        observations=len(rows),
        audited_searches=checked,
        draw_declarations=draws,
        every_archive_member_verified=True,
        paired_ids_thresholds_and_gates_verified=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.study)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
