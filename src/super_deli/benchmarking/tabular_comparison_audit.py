"""Independent numerical/density, pair-count and inclusion audits of wrapper artifacts."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def _bootstrap_audit(n, records, arrays):
    import numpy as np

    counts = np.zeros((n, n), dtype=int)
    oob_attempts, oob_success, oob_same = counts.copy(), counts.copy(), counts.copy()
    done = 0
    for row in records:
        ids = row["training_indices"]
        assert len(ids) == n and all(0 <= i < n for i in ids)
        oob = set(range(n)) - set(ids)
        assert sorted(oob) == row["oob_indices"]
        mask = np.zeros((n, n), dtype=bool)
        for i in oob:
            for j in oob:
                mask[i, j] = True
        oob_attempts += mask
        if row["status"] != "complete":
            continue
        done += 1
        labels = row["labels"]
        same = np.equal.outer(labels, labels)
        counts += same
        oob_success += mask
        oob_same += mask & same
    for name, expected in (
        ("coassignment_counts", counts),
        ("oob_requested", oob_attempts),
        ("oob_successful", oob_success),
        ("oob_coassignment_counts", oob_same),
    ):
        np.testing.assert_array_equal(arrays[name], expected)
    low, high = counts / len(records), (counts + len(records) - done) / len(records)
    np.fill_diagonal(low, 1)
    np.fill_diagonal(high, 1)
    np.testing.assert_allclose(arrays["coassignment_lower"], low)
    np.testing.assert_allclose(arrays["coassignment_upper"], high)
    if done == len(records):
        np.testing.assert_allclose(arrays["coassignment"], low)


def audit_analysis(directory):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.benchmarking.density_audit import _audit_fit, _audit_support

    out = Path(directory)
    summary = read_json(out / "summary.json")
    for name, digest in summary["artifacts"].items():
        assert fingerprint_file(out / name).sha256 == digest, name
    manifest = read_json(out / "manifest.json")
    assert (
        fingerprint_file(out / manifest["frozen_dataset"]["modalities"][0]["path"]).sha256
        == summary["source_sha256"]
    )
    table = read_json(out / "cohort/modality_000/table.json")
    x = np.asarray(table["values"], dtype=float)
    assert table["unit_ids"] == summary["unit_ids"]
    points = {}
    draws = 0
    for model in summary["models"]:
        if model["status"] == "failed":
            continue
        name, family = model["name"], model["family"]
        d = out / "arms" / name
        p = read_json(d / "comparison-point.json")
        points[name] = p
        native = read_json(d / "summary.json")
        assert native["unit_ids"] == summary["unit_ids"]
        assert native["observation_ids"] == summary["observation_ids"]
        assert native["provenance"]["ingestion"]["inputs"][0]["sha256"] == summary["source_sha256"]
        arrays = {}
        with np.load(d / ("preprocessing.npz" if family == "gmm" else "point.npz")) as data:
            arrays.update({k: data[k] for k in data.files})
        if family == "gmm":
            with np.load(d / "mixture.npz") as data:
                arrays.update({"mixture_means" if k == "means" else k: data[k] for k in data.files})
            arrays["strength"] = arrays["membership"].max(axis=1)
        record = {"labels": p["labels"]}
        if family == "gmm":
            record["candidates"] = read_json(d / "candidates.json")
        _audit_fit(x, {"a_" + k: v for k, v in arrays.items()}, "a", record, family)
        threshold = model["configuration"].get("assignment_threshold", 0)
        resolved = (arrays["labels"] >= 0) & (arrays["strength"] >= threshold)
        np.testing.assert_array_equal(p["resolved"], resolved)
        assert model["resolved_count"] == int(resolved.sum())
        assert model["selected_k"] == len(set(p["labels"]) - {-1})
        assert model["noise_count"] == sum(v < 0 for v in p["labels"])
        records = read_json(
            d / ("bootstrap_replicates.json" if family == "gmm" else "subsamples.json")
        )
        draws += len(records)
        if records:
            with np.load(d / "stability.npz") as data:
                if family == "gmm":
                    _bootstrap_audit(len(x), records, data)
                else:
                    _audit_support(len(x), records, data)
    for row in summary["comparisons"]:
        if row["status"] == "unavailable":
            continue
        a, b = points[row["left"]], points[row["right"]]
        ids = [
            i
            for i in range(len(x))
            if a["resolved"][i] and b["resolved"][i] and a["labels"][i] >= 0 and b["labels"][i] >= 0
        ]
        pairs = list(combinations(ids, 2))
        assert row["joint_resolved"] == len(ids) and row["joint_pairs"] == len(pairs)
        disagree = sum(
            (a["labels"][i] == a["labels"][j]) != (b["labels"][i] == b["labels"][j])
            for i, j in pairs
        )
        assert row["discordant_pairs"] == disagree
        if pairs:
            np.testing.assert_allclose(
                row["adjusted_rand"],
                adjusted_rand_score([a["labels"][i] for i in ids], [b["labels"][i] for i in ids]),
                atol=1e-12,
            )
            np.testing.assert_allclose(row["coassignment_agreement"], 1 - disagree / len(pairs))
        else:
            assert row["adjusted_rand"] is None
    return {"models": len(points), "draws": draws, "comparisons": len(summary["comparisons"])}


def audit_validation(directory, output_file):
    from super_deli.benchmarking.tabular_comparison import tasks

    root = Path(directory)
    plan = read_json(root / "plan.json")
    checks = []
    pipeline_runs = 0
    failures = []
    metrics = []
    for task in tasks(plan["phase"]):
        d = root / task["key"]
        validation = read_json(d / "validation.json")
        if validation["status"] != "complete":
            failures.append(validation)
        check = audit_analysis(d / "analysis")
        checks.append(check)
        pipeline_runs += check["models"] + check["draws"] + len(validation.get("replays", []))
        summary = read_json(d / "analysis/summary.json")
        for model in summary["models"]:
            metrics.append(
                {
                    "case": task["case"],
                    "n": task["n"],
                    "dataset": task["dataset"],
                    **{
                        k: model.get(k)
                        for k in ("name", "status", "selected_k", "noise_count", "resolved_count")
                    },
                }
            )
    expected = 88 if plan["phase"] == "preflight" else 672
    result = {
        "status": "passed" if not failures and pipeline_runs == expected else "failed",
        "phase": plan["phase"],
        "pipeline_runs": pipeline_runs,
        "expected_pipeline_runs": expected,
        "observations": len(checks),
        "comparisons": sum(r["comparisons"] for r in checks),
        "failures": failures,
        "metrics": metrics,
        "independent_checks": [
            "source/unit order",
            "PCA from raw means/scales and covariance",
            "GMM density/membership/BIC",
            "HDBSCAN bounds/noise/size",
            "native hashes",
            "explicit pair counts and sklearn ARI",
            "separate bootstrap/OOB/subsample denominators",
        ],
    }
    write_json(result, output_file)
    return result
