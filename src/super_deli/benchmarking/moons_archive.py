"""Read-only source binding and reconstruction of the prior density study."""

from __future__ import annotations

import csv
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json

BASE_SHA = "85c5db3686d451673bb2a5a852ba86a6024e2e3d"


def simulate_latents(case, n, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    if case == "gaussian":
        x = rng.normal(size=(n, 2))
        truth = np.zeros(n, dtype=int)
        latent = np.empty((0, 2))
        noise = x.copy()
        angles = np.empty(0)
    elif case in {"moons", "moons_008", "moons_002"}:
        a, b = rng.uniform(0, np.pi, (2, n // 2))
        latent = np.concatenate(
            [
                np.column_stack([np.cos(a), np.sin(a)]),
                np.column_stack([1 - np.cos(b), 0.5 - np.sin(b)]),
            ]
        )
        sigma = 0.02 if case == "moons_002" else 0.08
        noise = rng.normal(0, sigma, latent.shape)
        x = latent + noise
        angles = np.concatenate([a, b])
        truth = np.repeat([0, 1], n // 2)
    else:
        raise ValueError("Unsupported frozen simulation law")
    order = rng.permutation(n)
    return {
        "x": x[order],
        "truth": truth[order],
        "latent": latent[order] if len(latent) else latent,
        "noise": noise[order],
        "angles": angles[order] if len(angles) else angles,
    }


def source_matrix(path):
    import numpy as np

    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return np.array([[float(r["trait_1"]), float(r["trait_2"])] for r in rows]), tuple(
        r["observation_id"] for r in rows
    )


def archive_plan(archive_root, sample_sizes, *, draws_per_size=0):
    root = Path(archive_root).resolve()
    if read_json(root / "canonical-commit.json")["sha"] != BASE_SHA:
        raise ValueError("Archive must bind the completed density checkpoint")
    frozen = read_json(root / "study/frozen-plan.json")
    if frozen["phase"] != "main" or read_json(root / "audit/summary.json")["status"] != "passed":
        raise ValueError("Completed independently audited source study required")
    rows = []
    for item in frozen["plan"]:
        if item["case"] != "moons" or item["n"] not in sample_sizes:
            continue
        rows.append(
            {
                "key": "archived_" + item["key"],
                "source_key": item["key"],
                "origin": "archived",
                "case": "moons_008",
                "n": item["n"],
                "dataset": item["dataset"],
                "seed": item["data_seed"],
            }
        )
    expected = frozen["configuration"]["n_datasets"] * len(sample_sizes)
    if len(rows) != expected:
        raise ValueError("Archived moons coverage is incomplete")
    replays = []
    for n in sample_sizes:
        selected = sorted((x for x in rows if x["n"] == n), key=lambda x: x["dataset"])[
            :draws_per_size
        ]
        if len(selected) != draws_per_size:
            raise ValueError("Insufficient archived draw replay observations")
        for row in selected:
            replays.append(
                {**row, "key": "replay_" + row["source_key"], "origin": "replay", "draw": 0}
            )
    return rows, replays


def audit_archive(archive_root, *, output_dir, sample_sizes=(64, 256)):
    import numpy as np

    from super_deli.benchmarking.density_audit import _audit_fit, _audit_support
    from super_deli.benchmarking.moons_geometry import graph_diagnostics

    if not __debug__:
        raise RuntimeError("Numerical evidence audit requires assertions enabled")
    root, out = Path(archive_root).resolve(), Path(output_dir).resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Audit destination must be outside immutable archive")
    # Bind the complete 300-observation source by its previously audited inventory.
    inventory = read_json(root / "audit/files.json")
    for name, digest in inventory.items():
        if fingerprint_file(root / "study" / name).sha256 != digest:
            raise ValueError(f"Source archive checksum differs: {name}")
    rows, _ = archive_plan(root, sample_sizes)
    findings = []
    draws = 0
    for row in rows:
        p = root / "study" / row["source_key"]
        x, ids = source_matrix(p / "raw/traits.tsv")
        data = simulate_latents("moons", len(x), row["seed"])
        np.testing.assert_array_equal(x, data["x"])
        with np.load(p / "truth.npz", allow_pickle=False) as truth:
            np.testing.assert_array_equal(truth["labels"], data["truth"])
        metadata = read_json(p / "hdbscan.json")
        with np.load(p / "hdbscan.npz", allow_pickle=False) as a:
            arrays = dict(a)
        _audit_fit(x, arrays, "point", metadata["point"], "hdbscan")
        config = read_json(p / "density-config.json")
        expected = [
            np.sort(np.random.default_rng(s).choice(len(x), int(0.8 * len(x)), replace=False))
            for s in np.random.SeedSequence(config["random_seed"]).spawn(config["n_subsamples"])
        ]
        graphs = []
        for i, record in enumerate(metadata["draws"]):
            np.testing.assert_array_equal(record["training_indices"], expected[i])
            assert record["status"] == "complete"
            _audit_fit(x[expected[i]], arrays, f"draw_{i}", record, "hdbscan")
            graph, _, _ = graph_diagnostics(
                arrays[f"draw_{i}_scores"], data["truth"][expected[i]], 5
            )
            graphs.append(graph)
            draws += 1
        if metadata["draws"]:
            _audit_support(len(x), metadata["draws"], arrays)
        graph, _, _ = graph_diagnostics(arrays["point_scores"], data["truth"], 5)
        findings.append(
            {
                "source_key": row["source_key"],
                "n": len(x),
                "ids": ids,
                "selected_k": len(set(arrays["point_labels"]) - {-1}),
                "point_graph": graph,
                "draw_graphs": graphs,
                "prior_gmm_metrics": read_json(p / "summary.json")["families"]["gmm"]["metrics"],
            }
        )
    bindings = {
        n: fingerprint_file(root / n).sha256
        for n in (
            "canonical-commit.json",
            "study/frozen-plan.json",
            "study/summary.json",
            "audit/summary.json",
            "audit/files.json",
            "frozen-runtime-sha256.json",
        )
    }
    result = {
        "status": "passed",
        "source_commit": BASE_SHA,
        "source_binding": bindings,
        "verified_source_files": len(inventory),
        "observations": len(rows),
        "retained_draws": draws,
        "raw_features": 2,
        "retained_pcs": 2,
        "refits": 0,
        "archived_hierarchy": "not retained; available only through explicit new fits",
    }
    out.mkdir(exist_ok=False)
    write_json(findings, out / "findings.json")
    write_json(result, out / "summary.json")
    return result


def write_analytical(x, ids, directory):
    from super_deli.core.schema import config_from_dict

    out = Path(directory)
    out.mkdir()
    with (out / "traits.tsv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["observation_id", "trait_1", "trait_2"])
        writer.writerows([unit, *row] for unit, row in zip(ids, x, strict=True))
    value = {
        "schema_version": 1,
        "dataset_id": "invented_moons_diagnostic",
        "specimens": [{"specimen_id": u} for u in ids],
        "modalities": [
            {
                "name": "traits",
                "kind": "morphological",
                "level": "specimen",
                "format": "tsv",
                "path": "traits.tsv",
                "features": [{"name": f"trait_{j}", "dtype": "continuous"} for j in (1, 2)],
                "observations": [{"observation_id": u, "unit_id": u} for u in ids],
            }
        ],
    }
    dataset = config_from_dict(value)
    write_json(dataset.to_dict(), out / "dataset.json")
    return dataset
