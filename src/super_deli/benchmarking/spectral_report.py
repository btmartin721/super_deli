"""Retrospective supplied-K membership summaries, never cluster-count selection."""

from collections import Counter

import numpy as np

from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_fit import recovery
from super_deli.benchmarking.spectral_archive import load_coordinates
from super_deli.benchmarking.spectral_config import SPACES, graph_jobs
from super_deli.benchmarking.spectral_study import check_frozen, job_path
from super_deli.io.config import read_json, write_json


def cut_metrics(a, labels):
    from scipy.sparse import csr_matrix

    n = len(labels)
    affinity = csr_matrix((a["data"], a["indices"], a["indptr"]), shape=(n, n)).toarray()
    degree = affinity.sum(1)
    cuts = []
    conductance = []
    for k in set(labels):
        ix = labels == k
        cut = float(affinity[np.ix_(ix, ~ix)].sum())
        volume = float(degree[ix].sum())
        cuts.append(cut / volume)
        conductance.append(cut / min(volume, float(degree[~ix].sum())))
    return {"normalized_cut": float(sum(cuts)), "max_conductance": max(conductance)}


def report(root):
    frozen, config = check_frozen(root)
    require(read_json(root / "main-audit.json")["passed"], "Passed main audit required")
    binding = read_json(root / "binding.json")
    lookup = {e["key"]: e for e in binding["observations"]}
    rows, baselines, constant_reference = [], [], []
    for job in graph_jobs(binding, "main"):
        if job["kind"] != "point":
            continue
        entry = lookup[job["key"]]
        coords = load_coordinates(root, entry)
        out = job_path(root, "main", job)
        result = read_json(out / "result.json")
        a = arrays(out / "arrays.npz")
        for p in result["partitions"]:
            row = {
                **job,
                "case": entry["case"],
                "n": entry["n"],
                **p,
                "graph_components": result.get("components"),
                "matching_generating_k": p["target_k"] == len(set(coords["truth"])),
            }
            if p["status"] == "available":
                labels = a[f"K{p['target_k']}_labels"]
                row.update(recovery(labels, coords["truth"]))
                row["label_count"] = row.pop("selected_k")
                row.update(cut_metrics(a, labels))
                from scipy.sparse import csr_matrix

                graph = csr_matrix(
                    (a["data"], a["indices"], a["indptr"]), shape=(entry["n"], entry["n"])
                ).tocoo()
                row["oracle_edge_mixing"] = float(
                    graph.data[coords["truth"][graph.row] != coords["truth"][graph.col]].sum()
                    / graph.data.sum()
                )
            rows.append(row)
    for entry in binding["observations"]:
        coords = load_coordinates(root, entry)
        reference = recovery(np.zeros(entry["n"], dtype=int), coords["truth"])
        reference.pop("selected_k")
        constant_reference.append({"key": entry["key"], "fixed_k": 1, **reference})
        for space in SPACES:
            row = {"key": entry["key"], "case": entry["case"], "n": entry["n"], "space": space}
            if space.endswith("_traits"):
                job = {"key": entry["key"], "space": space, "kind": "point"}
                out = job_path(root, "main", job, True)
                r = read_json(out / "result.json")
                row["status"] = r["status"]
                labels = arrays(out / "arrays.npz").get("labels")
            else:
                row["status"] = "complete"
                labels = coords[space + "_labels"]
            if row["status"] == "complete":
                row.update(recovery(labels, coords["truth"]))
            baselines.append(row)
    groups = []
    for case in ("gaussian", "four_spherical", "moons"):
        for n in binding["sizes"]:
            for space in SPACES:
                base = [r for r in baselines if (r["case"], r["n"], r["space"]) == (case, n, space)]
                for neighbors in (5, 15):
                    for target in (2, 4):
                        selected = [
                            r
                            for r in rows
                            if (r["case"], r["n"], r["space"], r["neighbors"], r["target_k"])
                            == (case, n, space, neighbors, target)
                        ]
                        done = [r for r in selected if r["status"] == "available"]
                        nreq = len(selected)
                        vals = [r["ari"] for r in done]
                        unavailable = nreq - len(done)
                        common = [(r, next(b for b in base if b["key"] == r["key"])) for r in done]
                        common = [(r, b) for r, b in common if b["status"] == "complete"]
                        groups.append(
                            {
                                "case": case,
                                "n": n,
                                "space": space,
                                "neighbors": neighbors,
                                "target_k": target,
                                "requested": nreq,
                                "available": len(done),
                                "statuses": dict(Counter(r["status"] for r in selected)),
                                "reasons": dict(
                                    Counter(r.get("reason", "available") for r in selected)
                                ),
                                "available_mean_ari": float(np.mean(vals)) if vals else None,
                                "available_sd_ari": float(np.std(vals, ddof=1))
                                if len(vals) > 1
                                else None,
                                "all_request_ari_bounds": [
                                    (sum(vals) - unavailable) / nreq,
                                    (sum(vals) + unavailable) / nreq,
                                ],
                                "mean_density_ari": float(
                                    np.mean([b["ari"] for b in base if b["status"] == "complete"])
                                )
                                if any(b["status"] == "complete" for b in base)
                                else None,
                                "common_available_pairs": len(common),
                                "common_mean_ari_difference": float(
                                    np.mean([r["ari"] - b["ari"] for r, b in common])
                                )
                                if common
                                else None,
                                "mean_coverage_available": 1.0 if done else None,
                                "mean_small_clusters": float(
                                    np.mean([r["small_clusters"] for r in done])
                                )
                                if done
                                else None,
                            }
                        )
    summary = {
        "schema_version": 1,
        "fixture": config.fixture,
        "source_observations": len(binding["observations"]),
        "supplied_k": [2, 4],
        "selected_k": None,
        "neural_training_fits": 0,
        "constant_reference": constant_reference,
        "main_evaluations": frozen["counts"]["main"],
        "groups": groups,
        "partitions": dict(Counter(r["status"] for r in rows)),
        "interpretation": [
            "retrospective supplied-K diagnosis; not automatic species discovery",
            "Gaussian forced partitions are not false-positive tests",
            "available-only summaries are conditional; bounds retain every request",
            "ten archived datasets/cell; seeds/conditions/settings are repeated measurements",
            "no new p-values, resampling stability or default promotion",
        ],
    }
    write_json(rows, root / "point-assessments.json")
    write_json(baselines, root / "density-assessments.json")
    write_json(summary, root / "aggregate.json")
    return {
        "groups": len(groups),
        "point_partition_requests": len(rows),
        "density_baselines": len(baselines),
        "fixture": config.fixture,
    }
