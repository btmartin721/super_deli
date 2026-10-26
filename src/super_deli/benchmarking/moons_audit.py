"""Read-only checks using dense graph calculations and Python tree condensation."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json


def independent_condensation(linkage, n, minimum):
    """Prune by leaf-set sizes; independent of the library's condensation call."""
    import numpy as np

    members = {i: tuple([i]) for i in range(n)}
    children = {}
    for i, row in enumerate(linkage):
        a, b = int(row["left_node"]), int(row["right_node"])
        if a not in members or b not in members:
            raise ValueError("Invalid hierarchy ordering")
        members[n + i] = members[a] + members[b]
        children[n + i] = (a, b, float(row["value"]))
    pending = deque([(2 * n - 2, n)])
    next_id = n + 1
    edges = []
    while pending:
        node, branch = pending.popleft()
        if node < n:
            continue
        a, b, distance = children[node]
        large = [c for c in (a, b) if len(members[c]) >= minimum]
        if len(large) == 2:
            for c in large:
                label = next_id
                next_id += 1
                edges.append((branch, label, 1 / distance, len(members[c])))
                pending.append((c, label))
        else:
            for c in (a, b):
                if c in large:
                    pending.append((c, branch))
                else:
                    edges.extend((branch, leaf, 1 / distance, 1) for leaf in members[c])
    return np.array(
        edges,
        dtype=[
            ("parent", np.intp),
            ("child", np.intp),
            ("value", np.float64),
            ("cluster_size", np.intp),
        ],
    )


def _same_condensation(expected, actual, n):
    import numpy as np

    # IDs/row order can differ for equivalent trees; compare leaf-set branches.
    def canonical(tree):
        outgoing = {}
        for row in tree:
            outgoing.setdefault(int(row["parent"]), []).append(int(row["child"]))
        sets = {i: (i,) for i in range(n)}
        for node in sorted(outgoing, reverse=True):
            sets[node] = tuple(sorted(v for c in outgoing[node] for v in sets[c]))
        return sorted(
            [
                (
                    sets[int(row["parent"])],
                    sets[int(row["child"])],
                    int(row["cluster_size"]),
                    float(row["value"]),
                )
                for row in tree
            ]
        )

    a, b = canonical(expected), canonical(actual)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x[:3] == y[:3]
        np.testing.assert_allclose(x[3], y[3], atol=1e-11, rtol=1e-11)


def audit_arm(x, truth, metadata, arrays):
    import math

    import numpy as np
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import cdist
    from sklearn.cluster._hdbscan._tree import tree_to_labels

    from super_deli.benchmarking.moons_hierarchy import validate_linkage

    kept = arrays["kept_indices"]
    mean = x[:, kept].mean(0)
    scale = x[:, kept].std(0) if metadata["scaling"] == "standard" else np.ones(len(kept))
    np.testing.assert_allclose(arrays["means"], mean, atol=1e-12)
    np.testing.assert_allclose(arrays["scales"], scale, atol=1e-12)
    z = (x[:, kept] - mean) / scale
    scores = arrays["scores"]
    components = arrays["components"]
    np.testing.assert_allclose(scores, (z - arrays["pca_mean"]) @ components.T, atol=1e-11)
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-11)
    np.testing.assert_allclose(
        arrays["explained_variance"], np.linalg.eigvalsh(np.cov(z, rowvar=False))[::-1], atol=1e-11
    )
    d = cdist(scores, scores)
    np.testing.assert_allclose(d, cdist(z, z), atol=1e-11)
    params = metadata["parameters"]
    ms = params["min_samples"]
    # Partition selection differs algorithmically from a full sort used in reporting.
    core = np.partition(d, ms - 1, axis=1)[:, ms - 1]
    np.testing.assert_allclose(arrays["core_distances"], core, atol=1e-11)
    graph = np.maximum(d, np.maximum.outer(core, core))
    np.fill_diagonal(graph, 0)
    scipy_tree = minimum_spanning_tree(graph)
    np.testing.assert_allclose(
        np.sort(arrays["mst_edges"][:, 2]), np.sort(scipy_tree.data), atol=1e-11
    )
    groups = np.unique(truth)
    if len(groups) > 1:
        between = truth[:, None] != truth[None, :]
        neighbor = np.argsort(d, axis=1)[:, 1:ms]
        np.testing.assert_allclose(
            metadata["graph"]["cross_neighbor_fraction"], np.mean(truth[neighbor] != truth[:, None])
        )
        for name, matrix in (("euclidean", d), ("mutual", graph)):
            bridge = float(np.min(matrix[between]))
            bottleneck = max(
                float(minimum_spanning_tree(matrix[np.ix_(truth == k, truth == k)]).data.max())
                for k in groups
            )
            np.testing.assert_allclose(metadata["graph"][name + "_between_bridge"], bridge)
            np.testing.assert_allclose(metadata["graph"][name + "_within_bottleneck"], bottleneck)
            np.testing.assert_allclose(
                metadata["graph"][name + "_separation_ratio"], bridge / bottleneck
            )
    labels = arrays["labels"]
    assert labels.shape == (len(x),) and np.all(labels >= -1)
    assert np.isfinite(arrays["strength"]).all() and np.all(
        (arrays["strength"] >= 0) & (arrays["strength"] <= 1)
    )
    assert np.all(arrays["strength"][labels < 0] == 0)
    assert metadata["selected_k"] == len(set(labels) - {-1})
    assert metadata["noise_count"] == int(np.sum(labels < 0))
    if metadata["hierarchy_status"] != "available":
        return {"hierarchy_status": "unavailable"}
    linkage, condensed = arrays["linkage"], arrays["condensed"]
    members = validate_linkage(linkage, len(x))
    np.testing.assert_allclose(
        np.sort(linkage["value"]), np.sort(scipy_tree.data), rtol=1e-10, atol=1e-11
    )
    assert np.all(np.diff(linkage["value"]) >= -1e-11)
    for row in linkage:
        a, b = [sorted(members[int(row[k])]) for k in ("left_node", "right_node")]
        np.testing.assert_allclose(graph[np.ix_(a, b)].min(), row["value"], atol=1e-11)
    expected = independent_condensation(linkage, len(x), params["min_cluster_size"])
    _same_condensation(expected, condensed, len(x))
    # Replay only tree selection: no data, PCA, HDBSCAN fit or new pipeline run.
    result, strength = tree_to_labels(
        linkage,
        min_cluster_size=params["min_cluster_size"],
        cluster_selection_method=params["cluster_selection_method"],
        allow_single_cluster=params["allow_single_cluster"],
        cluster_selection_epsilon=0.0,
    )
    np.testing.assert_array_equal(result, labels)
    np.testing.assert_allclose(strength, arrays["strength"], atol=1e-12)
    if "branch_reference" in metadata and len(groups) == 2:
        parent = {int(row["child"]): int(row["parent"]) for row in condensed}
        root = int(condensed["parent"].min())
        members = {int(k): set() for k in np.unique(condensed["parent"])}
        for i in range(len(x)):
            node = parent[i]
            while True:
                members[node].add(i)
                if node == root:
                    break
                node = parent[node]
        candidates = [k for k in members if k != root]
        ref = [set(np.flatnonzero(truth == k)) for k in groups]
        score = {k: [len(members[k] & g) / len(members[k] | g) for g in ref] for k in candidates}
        best = [max((score[k][j] for k in candidates), default=0.0) for j in (0, 1)]
        disjoint = max(
            (
                min(score[a][0], score[b][1])
                for a in candidates
                for b in candidates
                if not members[a] & members[b]
            ),
            default=0.0,
        )
        np.testing.assert_allclose(best, metadata["branch_reference"]["best_branch_jaccard"])
        np.testing.assert_allclose(
            disjoint, metadata["branch_reference"]["best_disjoint_min_jaccard"]
        )
    selection = metadata["selection"]
    nodes = {r["node"]: r for r in selection["nodes"]}
    for node, row in nodes.items():
        edges = condensed[condensed["parent"] == node]
        value = float(np.dot(edges["value"] - row["birth"], edges["cluster_size"]))
        np.testing.assert_allclose(value, row["own_stability"], atol=1e-9, rtol=1e-11)
        total = math.fsum(
            max(nodes[c]["own_stability"], nodes[c]["best_child_sum"]) for c in row["children"]
        )
        np.testing.assert_allclose(total, row["best_child_sum"], atol=1e-9, rtol=1e-11)
        if params["cluster_selection_method"] == "eom":
            expected_choice = (
                node != selection["root"] or params["allow_single_cluster"]
            ) and value >= total
            if not math.isclose(value, total, abs_tol=1e-9, rel_tol=1e-11):
                assert bool(row["locally_selected"]) == bool(expected_choice)
    return {
        "hierarchy_status": "available",
        "condensed_edges": len(condensed),
        "selected_nodes": selection["selected_nodes"],
        "root_margin": selection["root_margin"],
    }


def audit_moons_diagnostic(study_dir, *, archive_root, output_dir):
    import numpy as np
    from scipy.stats import beta

    from super_deli.benchmarking.density_study import recovery_metrics
    from super_deli.benchmarking.moons_archive import archive_plan, simulate_latents, source_matrix
    from super_deli.benchmarking.moons_config import ARMS, CASES, MoonsDiagnosticConfig
    from super_deli.benchmarking.moons_study import diagnostic_plan

    if not __debug__:
        raise RuntimeError("Numerical evidence audit requires assertions enabled")
    source, out = Path(study_dir).resolve(), Path(output_dir).resolve()
    if out == source or out.is_relative_to(source):
        raise ValueError("Audit must be outside immutable study evidence")
    frozen = read_json(source / "frozen-plan.json")
    c = MoonsDiagnosticConfig.from_dict(frozen["configuration"])
    assert frozen["plan"] == diagnostic_plan(c, archive_root, frozen["phase"])
    rows = []
    numerical = []
    files = {}
    replays = 0
    hierarchy_missing = 0
    failed = 0
    for item in frozen["plan"]:
        p = source / item["key"]
        for name, digest in read_json(p / "checksums.json").items():
            assert fingerprint_file(p / name).sha256 == digest
            files[item["key"] + "/" + name] = digest
        row = read_json(p / "summary.json")
        assert read_json(p / "plan.json") == item
        if row.get("setup_status") == "failed":
            rows.append(row)
            continue
        data = simulate_latents(item["case"], item["n"], item["seed"])
        original_ids = tuple(f"unit_{i:04d}" for i in range(item["n"]))
        if item["origin"] == "replay":
            old = Path(archive_root) / "study" / item["source_key"]
            ids = np.array(
                read_json(old / "hdbscan.json")["draws"][item["draw"]]["training_indices"]
            )
            data = {k: v[ids] if len(v) else v for k, v in data.items()}
            original_ids = tuple(original_ids[i] for i in ids)
        raw, obs = source_matrix(p / "raw/traits.tsv")
        np.testing.assert_array_equal(raw, data["x"])
        assert obs == original_ids
        with np.load(p / "reference.npz", allow_pickle=False) as a:
            for k, v in data.items():
                if k != "x":
                    np.testing.assert_array_equal(a[k], v)
        for arm in ("baseline",) if item["origin"] == "replay" else ARMS:
            metadata = read_json(p / f"{arm}.json")
            if metadata["status"] == "complete":
                declared = frozen["arms"][arm]
                assert metadata["scaling"] == declared["scaling"]
                for key in (
                    "min_cluster_size",
                    "min_samples",
                    "alpha",
                    "cluster_selection_epsilon",
                    "cluster_selection_method",
                    "allow_single_cluster",
                ):
                    assert metadata["parameters"][key] == declared["model"][key]
                assert metadata["parameters"]["metric"] == "euclidean"
                assert metadata["parameters"]["n_jobs"] == 1
            if metadata["status"] == "failed":
                failed += 1
                continue
            with np.load(p / f"{arm}.npz", allow_pickle=False) as a:
                arrays = dict(a)
            result = audit_arm(raw, data["truth"], metadata, arrays)
            numerical.append({"key": item["key"], "arm": arm, **result})
            hierarchy_missing += result["hierarchy_status"] != "available"
            assert row["arms"][arm]["metrics"] == recovery_metrics(arrays["labels"], data["truth"])
            if arm == "baseline" and item["origin"] != "fresh":
                old = Path(archive_root) / "study" / item["source_key"]
                prefix = f"draw_{item['draw']}" if item["origin"] == "replay" else "point"
                with np.load(old / "hdbscan.npz", allow_pickle=False) as a:
                    for key in ("labels", "strength", "scores", "means", "scales", "components"):
                        np.testing.assert_allclose(arrays[key], a[f"{prefix}_{key}"], atol=1e-11)
                replays += 1
        rows.append(row)
    recorded = read_json(source / "summary.json")
    ledger = [
        __import__("json").loads(line)
        for line in (source / "ledger.jsonl").read_text().splitlines()
    ]
    assert ledger == rows
    assert recorded["successful_fits"] == len(numerical)
    assert recorded["failed_fits"] == failed
    assert recorded["unavailable_hierarchies"] == hierarchy_missing
    expected_count = sum(1 if item["origin"] == "replay" else len(ARMS) for item in frozen["plan"])
    unattempted = expected_count - sum(len(row["arms"]) for row in rows)
    assert recorded["unattempted_fits"] == unattempted
    rates = []
    descriptive = []
    fresh = [r for r in rows if r["origin"] == "fresh"]
    multiplicity = len(ARMS) * len(CASES) * len(c.sample_sizes)
    tail = 0.05 / multiplicity / 2
    for case in CASES:
        for n in c.sample_sizes:
            cell = [r for r in fresh if r["case"] == case and r["n"] == n]
            for arm in ARMS:
                valid = [
                    r["arms"][arm]
                    for r in cell
                    if r["arms"].get(arm, {}).get("status") == "complete"
                ]
                passed = sum(a["metrics"]["correct_k"] for a in valid)
                missing = len(cell) - len(valid)
                total = len(cell)
                upper_count = passed + missing
                rates.append(
                    {
                        "case": case,
                        "n": n,
                        "arm": arm,
                        "datasets": total,
                        "correct_k": passed,
                        "failed_or_unattempted": missing,
                        "rate_lower": passed / total,
                        "rate_upper": upper_count / total,
                        "simultaneous_95_lower": float(beta.ppf(tail, passed, total - passed + 1))
                        if passed
                        else 0.0,
                        "simultaneous_95_upper": float(
                            beta.ppf(1 - tail, upper_count + 1, total - upper_count)
                        )
                        if upper_count < total
                        else 1.0,
                    }
                )
    for origin in ("fresh", "archived"):
        for case in CASES:
            for n in c.sample_sizes:
                cell = [
                    r for r in rows if r["origin"] == origin and r["case"] == case and r["n"] == n
                ]
                if not cell:
                    continue
                for arm in ARMS:
                    good = [
                        r["arms"][arm]
                        for r in cell
                        if r["arms"].get(arm, {}).get("status") == "complete"
                    ]
                    d = {
                        "origin": origin,
                        "case": case,
                        "n": n,
                        "arm": arm,
                        "datasets": len(cell),
                        "successful": len(good),
                        "K_counts": {
                            str(k): sum(a["selected_k"] == k for a in good)
                            for k in sorted({a["selected_k"] for a in good})
                        },
                    }
                    for field in (
                        "ari",
                        "assigned_ari",
                        "assigned_fraction",
                        "split_generating_groups",
                        "merged_predicted_groups",
                    ):
                        values = [
                            a["metrics"][field] for a in good if a["metrics"].get(field) is not None
                        ]
                        d["mean_" + field] = float(np.mean(values)) if values else None
                    descriptive.append(d)
    archived, _ = archive_plan(archive_root, c.sample_sizes)
    result = {
        "status": "passed",
        "phase": frozen["phase"],
        "configuration_sha256": configuration_hash(c.to_dict()),
        "observations": len(rows),
        "archived_observations": len(archived),
        "audited_successful_fits": len(numerical),
        "failed_fits": failed,
        "unattempted_fits": unattempted,
        "unavailable_hierarchies": hierarchy_missing,
        "verified_replays": replays,
        "primary_rates": rates,
        "descriptive_metrics": descriptive,
        "audit_limit": "Graph/condensation/stability checks; not a full second HDBSCAN",
    }
    out.mkdir(exist_ok=False)
    write_json(result, out / "summary.json")
    write_json(numerical, out / "numerical.json")
    write_json(files, out / "files.json")
    return result
