"""Standard-library verification of retained bytes, fold gains, criteria and totals."""

import argparse
import ast
import hashlib
import io
import json
import math
import struct
import tarfile
from pathlib import Path
from zipfile import ZipFile


def read(path):
    return json.loads(Path(path).read_text())


def npy(raw):
    assert raw[:6] == b"\x93NUMPY"
    major = raw[6]
    assert major in (1, 2, 3)
    width = 2 if major == 1 else 4
    length = int.from_bytes(raw[8 : 8 + width], "little")
    end = 8 + width + length
    header = ast.literal_eval(raw[8 + width : end].decode("utf-8" if major == 3 else "latin1"))
    return header, raw[end:]


def floats(npz, name):
    with ZipFile(io.BytesIO(npz)) as z:
        h, data = npy(z.read(name + ".npy"))
    assert len(h["shape"]) == 1 and h["descr"] in ("<f8", ">f8", "=f8")
    values = [x[0] for x in struct.iter_unpack((">" if h["descr"][0] == ">" else "<") + "d", data)]
    assert len(values) == h["shape"][0] and all(math.isfinite(x) for x in values)
    return values


def verified_archive(root, row):
    archive = root / row["archive"]
    manifest = root / row["manifest"]
    assert archive.stat().st_size == row["size_bytes"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["archive_sha256"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == row["manifest_sha256"]
    expected = read(manifest)["files"]
    packed = tarfile.open(archive)
    members = packed.getmembers()
    assert len(members) == len(expected) == len({m.name for m in members})
    for m in members:
        assert m.isfile() and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts
        data = packed.extractfile(m).read()
        assert len(data) == expected[m.name]["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == expected[m.name]["sha256"]
    return packed


def verify(study, audit):
    root = Path(study)
    config = read(root / "configuration.json")
    rows = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    plan = read(root / "plan.json")
    assert len(rows) == len(plan)
    expected = {(r["cell"], r["repetition"]): r for r in plan}
    seen = set()
    fits = complete = failed = 0
    selections = {}
    for row in rows:
        identity = row["cell"], row["repetition"]
        assert identity not in seen
        seen.add(identity)
        assert all(row[k] == v for k, v in expected[identity].items())
        for field, purpose in (
            ("data_seed", "training_data"),
            ("test_seed", "external_data"),
            ("split_seed", "fold_split"),
        ):
            payload = json.dumps(
                [config["random_seed"], *identity, purpose], separators=(",", ":")
            ).encode()
            assert row[field] == int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        with verified_archive(root, row) as z:

            def data(name):
                return z.extractfile(name).read()

            def load(name):
                return json.loads(data(name))

            inner = load("record.json")
            assert (
                row["selection"] == inner["selection"]
                and row["performance"] == inner["performance"]
            )
            result = load("workflow/summary.json")
            assert result["selection"] == row["selection"]
            records = result["searches"]
            fits += len(records)
            complete += sum(r["status"] == "complete" for r in records)
            failed += sum(r["status"] != "complete" for r in records)
            for record in records:
                if record["status"] != "complete":
                    assert record["error"]
                    continue
                prefix = record.get("archive_prefix", record.get("prefix"))
                candidates = load(f"workflow/{prefix}/run/candidates.json")
                with ZipFile(io.BytesIO(data(f"workflow/{prefix}/run/gmm_candidates.npz"))) as p:
                    h, _ = npy(p.read("scores.npy"))
                n, d = h["shape"]
                assert d == config["model"]["n_pcs"]
                for c in candidates:
                    k = c["k"]
                    if c["status"] == "insufficient_samples":
                        assert n < k * config["model"]["min_cluster_size"]
                        continue
                    covariance_count = (
                        1 if config["model"].get("covariance_type", "full") == "tied" else k
                    )
                    q = k - 1 + k * d + covariance_count * d * (d + 1) // 2
                    assert math.isclose(c["aic"], -2 * c["log_likelihood"] + 2 * q, abs_tol=1e-7)
                    assert math.isclose(
                        c["bic"], -2 * c["log_likelihood"] + math.log(n) * q, abs_tol=1e-7
                    )
                    sizes = c["distinct_unit_sizes"]
                    assert sum(sizes) == n
                    assert (c["status"] == "eligible") == (
                        c["converged"] and min(sizes) >= config["model"]["min_cluster_size"]
                    )
            selection = row["selection"]
            if selection["status"] == "complete":
                full = load("workflow/full/run/candidates.json")
                units = load("workflow/full/run/summary.json")["unit_ids"]
                assignments = load("workflow/fold_assignments.json")
                assert sorted(j for f in assignments for j in f["validation_indices"]) == list(
                    range(len(units))
                )
                decisions = []
                for c in full:
                    k = c["k"]
                    usable = c["status"] == "eligible"
                    gains = []
                    for f in assignments:
                        i = f["fold"]
                        assert set(f["training_indices"]).isdisjoint(f["validation_indices"])
                        assert sorted(f["training_indices"] + f["validation_indices"]) == list(
                            range(len(units))
                        )
                        assert f["validation_unit_ids"] == [
                            units[j] for j in f["validation_indices"]
                        ]
                        assert f["training_unit_ids"] == [units[j] for j in f["training_indices"]]
                        fold = load(f"workflow/fold_{i:03d}/run/candidates.json")
                        eligible = next(x for x in fold if x["k"] == k)["status"] == "eligible"
                        usable = usable and eligible
                        if eligible:
                            raw = data(f"workflow/fold_{i:03d}/validation.npz")
                            a, b = floats(raw, "k" + str(k)), floats(raw, "k1")
                            assert len(a) == len(f["validation_indices"])
                            gains.extend(x - y for x, y in zip(a, b, strict=True))
                    gain = math.fsum(gains) / len(units) if usable else None
                    expected_pass = k == 1 or (gain is not None and gain > 0)
                    actual = next(x for x in selection["candidates"] if x["k"] == k)
                    assert actual["passes"] == expected_pass
                    if gain is None:
                        assert actual["pooled_validation_gain"] is None
                    else:
                        assert math.isclose(actual["pooled_validation_gain"], gain, abs_tol=1e-11)
                    if expected_pass:
                        decisions.append(c)
                chosen = min(decisions, key=lambda c: (c["aic"], c["k"]))["k"]
                assert selection["screened_k"] == chosen
                raw = min(
                    (c for c in full if c["status"] == "eligible"), key=lambda c: (c["aic"], c["k"])
                )["k"]
                assert selection["ordinary_aic_k"] == raw
            else:
                assert selection["screened_k"] is None
                assert "workflow/selected_model.npz" not in z.getnames()
            selections[identity] = row["performance"]
    assert seen == set(expected)
    result = read(Path(audit) / "summary.json")
    assert (result["attempted_fits"], result["successful_fits"], result["failed_fits"]) == (
        fits,
        complete,
        failed,
    )
    assert len(result["primary_rates"]) == result["primary_family_size"]
    for rate in result["primary_rates"]:
        records = [v[rate["method"]] for (cell, _), v in selections.items() if cell == rate["cell"]]
        available = [x for x in records if x is not None]
        events = sum((x["k"] > 1 if rate["event"] == "K>1" else x["k"] == 4) for x in available)
        assert (rate["events"], rate["available"], rate["requested"]) == (
            events,
            len(available),
            len(records),
        )
    return {
        "standard_library_check_passed": True,
        "observations": len(rows),
        "searches": fits,
        "successful_searches": complete,
        "failed_searches": failed,
        "primary_rates": len(result["primary_rates"]),
        "no_refits": True,
        "fold_gains_and_selected_totals_checked": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.study_root, args.audit_root)
    with args.output.open("x") as h:
        json.dump(result, h, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
