"""Independent standard-library verification of paired archive bytes and selection."""

import argparse
import hashlib
import importlib.util
import io
import json
import math
import tarfile
from pathlib import Path
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location(
    "screen_bytes", Path(__file__).parents[1] / "predictive-aic/check_screening.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def contents(raw, manifest):
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r|gz") as archive:
        for member in archive:
            name = member.name
            assert member.isfile() and not Path(name).is_absolute() and ".." not in Path(name).parts
            assert name not in result and name in manifest
            data = archive.extractfile(member).read()
            assert member.size == len(data) == manifest[name]["size_bytes"]
            assert hashlib.sha256(data).hexdigest() == manifest[name]["sha256"]
            result[name] = data
    assert set(result) == set(manifest)
    return result


def unpack(raw, raw_manifest, row):
    assert len(raw) == row["size_bytes"]
    assert hashlib.sha256(raw).hexdigest() == row["archive_sha256"]
    assert hashlib.sha256(raw_manifest).hexdigest() == row["manifest_sha256"]
    return contents(raw, json.loads(raw_manifest)["files"])


def verify_arm(files, row, model):
    def load(name):
        return json.loads(files[name])

    selection = load("workflow/summary.json")["selection"]
    assert selection == row["selection"]
    records = load("workflow/summary.json")["searches"]
    for record in records:
        if record["status"] != "complete":
            assert record["error"]
            continue
        prefix = record.get("archive_prefix", record.get("prefix"))
        candidates = load(f"workflow/{prefix}/run/candidates.json")
        with ZipFile(io.BytesIO(files[f"workflow/{prefix}/run/gmm_candidates.npz"])) as pool:
            header, _ = common.npy(pool.read("scores.npy"))
            n, d = header["shape"]
            for c in candidates:
                k = c["k"]
                if c["status"] == "insufficient_samples":
                    assert n < k * model["min_cluster_size"]
                    continue
                covariances = 1 if model.get("covariance_type", "full") == "tied" else k
                q = k - 1 + k * d + covariances * d * (d + 1) // 2
                assert math.isclose(c["aic"], -2 * c["log_likelihood"] + 2 * q, abs_tol=1e-7)
                assert math.isclose(
                    c["bic"], -2 * c["log_likelihood"] + math.log(n) * q, abs_tol=1e-7
                )
                assert (c["status"] == "eligible") == (
                    c["converged"] and min(c["distinct_unit_sizes"]) >= model["min_cluster_size"]
                )
    if selection["status"] != "complete":
        assert selection["screened_k"] is None and "workflow/selected_model.npz" not in files
        assert row["performance"]["screened"] is None
        return
    full = load("workflow/full/run/candidates.json")
    units = load("workflow/full/run/summary.json")["unit_ids"]
    folds = load("workflow/fold_assignments.json")
    assert len(folds) == 5
    assert sorted(j for f in folds for j in f["validation_indices"]) == list(range(len(units)))
    passing = []
    for candidate in full:
        k = candidate["k"]
        usable = candidate["status"] == "eligible"
        contributions = []
        for f in folds:
            train, valid = f["training_indices"], f["validation_indices"]
            assert set(train).isdisjoint(valid) and sorted(train + valid) == list(range(len(units)))
            assert f["training_unit_ids"] == [units[j] for j in train]
            assert f["validation_unit_ids"] == [units[j] for j in valid]
            prefix = f"workflow/fold_{f['fold']:03d}"
            cs = load(prefix + "/run/candidates.json")
            eligible = next(c for c in cs if c["k"] == k)["status"] == "eligible"
            usable = usable and eligible
            if eligible:
                raw = files[prefix + "/validation.npz"]
                a, b = common.floats(raw, "k" + str(k)), common.floats(raw, "k1")
                assert len(a) == len(b) == len(valid)
                contributions.extend(x - y for x, y in zip(a, b, strict=True))
        gain = math.fsum(contributions) / len(units) if usable else None
        actual = next(c for c in selection["candidates"] if c["k"] == k)
        passed = k == 1 or (gain is not None and gain > 0)
        assert actual["passes"] == passed
        if gain is None:
            assert actual["pooled_validation_gain"] is None
        else:
            assert math.isclose(actual["pooled_validation_gain"], gain, abs_tol=1e-11)
        if passed:
            passing.append(candidate)
    assert selection["screened_k"] == min(passing, key=lambda c: (c["aic"], c["k"]))["k"]
    eligible = [c for c in full if c["status"] == "eligible"]
    assert selection["ordinary_aic_k"] == min(eligible, key=lambda c: (c["aic"], c["k"]))["k"]


def verify(study, audit):
    root = Path(study)

    def read(name):
        return json.loads((root / name).read_text())

    config = read("configuration.json")["study"]
    plan = {(r["cell"], r["repetition"]): r for r in read("plan.json")}
    rows = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    seen, counts = (
        set(),
        {k: 0 for k in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")},
    )
    for row in rows:
        key = row["cell"], row["repetition"]
        assert key not in seen and all(row[k] == v for k, v in plan[key].items())
        seen.add(key)
        for field, purpose in (
            ("data_seed", "training_data"),
            ("test_seed", "external_data"),
            ("split_seed", "fold_split"),
        ):
            payload = json.dumps(
                [config["random_seed"], *key, purpose], separators=(",", ":")
            ).encode()
            assert row[field] == int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        outer = unpack(
            (root / row["archive"]).read_bytes(), (root / row["manifest"]).read_bytes(), row
        )
        assert all(row[k] == v for k, v in json.loads(outer["record.json"]).items())
        arms = {}
        for family in ("full", "tied"):
            r = row["arms"][family]
            files = unpack(
                outer[family + "/" + r["archive"]], outer[family + "/" + r["manifest"]], r
            )
            model = {**config["model"], **({"covariance_type": "tied"} if family == "tied" else {})}
            verify_arm(files, r, model)
            assert json.loads(files["workflow/configuration.json"])["predictive_model"] == model
            arms[family] = files
            for k in counts:
                counts[k] += r[k]
        for name in arms["full"]:
            if name.endswith(("/pca.npz", "/profiles.npz", "/fusion.npz", "/source_values.npy")):
                if name in arms["tied"]:
                    if name.endswith(".npy"):
                        assert arms["full"][name] == arms["tied"][name]
                    else:
                        with (
                            ZipFile(io.BytesIO(arms["full"][name])) as x,
                            ZipFile(io.BytesIO(arms["tied"][name])) as y,
                        ):
                            assert x.namelist() == y.namelist()
                            for n in x.namelist():
                                assert x.read(n) == y.read(n)
    assert seen == set(plan) and len(rows) == len(plan)
    result = json.loads((Path(audit) / "summary.json").read_text())
    assert all(result[k] == v for k, v in counts.items())
    assert len(result["primary_rates"]) == result["primary_family_size"]
    for rate in result["primary_rates"]:
        family, method = rate["method"].split("_")
        group = [
            r["arms"][family]["performance"][method] for r in rows if r["cell"] == rate["cell"]
        ]
        available = [v for v in group if v is not None]
        events = sum(v["k"] == 4 if rate["event"] == "K=4" else v["k"] > 1 for v in available)
        assert (rate["events"], rate["available"], rate["requested"]) == (
            events,
            len(available),
            len(group),
        )
    return {
        "standard_library_check_passed": True,
        "observations": len(rows),
        **counts,
        "primary_rates": len(result["primary_rates"]),
        "paired_comparisons": len(result["paired"]),
        "no_refits": True,
        "scope": "Archive/criteria/gate/totals audit independent of numerical libraries",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.study_root, args.audit_root)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
