"""Read-only numerical audits independent of diagnostic fitting/reporting."""

import math

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.cluster._hdbscan._tree import tree_to_labels
from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.moons_audit import _same_condensation, independent_condensation
from super_deli.benchmarking.moons_hierarchy import validate_linkage
from super_deli.benchmarking.neural_analysis import neural_forward
from super_deli.benchmarking.neural_moons_config import POLICIES


def audit_context(context, entry, source_root, diagnostics=None):
    """Check coordinate construction using separate forward/projection formulas."""
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.benchmarking.neural_moons_coordinates import safe_path

    ref = arrays(safe_path(source_root, entry["reference"]))
    ids = np.array(entry["indices"])
    assert entry["unit_ids"] == [f"invented_{i:04d}" for i in ids]
    x, z = ref["x"][ids], ref["z"][ids]
    np.testing.assert_array_equal(context["x"], x)
    np.testing.assert_array_equal(context["z"], z)
    np.testing.assert_array_equal(context["truth"], ref["truth"][ids])
    np.testing.assert_allclose(context["source_standard"], (z - z.mean(0)) / z.std(0), atol=1e-12)
    standardized = (x - x.mean(0)) / x.std(0)
    np.testing.assert_allclose(context["traits_standard"], standardized, atol=1e-12)
    for arm, name in (
        ("density", "pca_native"),
        ("neural_0", "neural_0_standard"),
        ("neural_101", "neural_101_standard"),
        ("neural_202", "neural_202_standard"),
    ):
        a = arrays(safe_path(source_root, entry["points"][arm]))
        if arm == "density":
            projection = np.einsum("ij,kj->ik", standardized - a["pca_mean"], a["components"])
        else:
            raw = neural_forward(standardized, a, "encoder")
            np.testing.assert_allclose(context[arm + "_raw"], raw, atol=1e-12)
            projection = (raw - a["latent_means"]) / a["latent_scales"]
            if diagnostics is not None:
                error = (neural_forward(raw, a, "decoder") - standardized) ** 2
                diag = diagnostics[arm]
                np.testing.assert_allclose(diag["mse_by_feature"], error.mean(0), atol=1e-12)
                for k in set(context["truth"]):
                    np.testing.assert_allclose(
                        diag["mse_by_generating_group"][str(int(k))],
                        error[context["truth"] == k].mean(),
                        atol=1e-12,
                    )
                np.testing.assert_array_equal(diag["training_losses"], a["losses"])
                np.testing.assert_allclose(diag["raw_latent_scales"], raw.std(0), atol=1e-12)
                np.testing.assert_allclose(
                    diag["raw_latent_singular_values"],
                    np.linalg.svd(raw - raw.mean(0), compute_uv=False),
                    atol=1e-12,
                )
        np.testing.assert_allclose(context[name], projection, atol=1e-11)
        np.testing.assert_array_equal(context[name], a["scores"])


def _neighbors(values, neighbors):
    d = cdist(values, values)
    np.fill_diagonal(d, np.inf)
    assert neighbors.shape == (len(values), 4) and neighbors.dtype.kind in "iu"
    assert all(len(set(row)) == 4 and i not in row for i, row in enumerate(neighbors))
    assert np.all(neighbors >= 0) and np.all(neighbors < len(values))
    selected = np.take_along_axis(d, neighbors, axis=1)
    cutoff = np.partition(d, 3, axis=1)[:, 3]
    assert np.all(selected <= cutoff[:, None] + 1e-11)
    return d


def audit_fit(context, representation, record, data):
    scores, truth = context[representation], context["truth"]
    np.testing.assert_array_equal(data["scores"], scores)
    d = cdist(scores, scores)
    core = np.partition(d, 4, axis=1)[:, 4]
    np.testing.assert_allclose(core, data["core_distances"], atol=1e-11)
    graph = np.maximum(d, np.maximum.outer(core, core))
    np.fill_diagonal(graph, 0)
    tree = minimum_spanning_tree(graph)
    np.testing.assert_allclose(np.sort(tree.data), np.sort(data["mst_edges"][:, 2]), atol=1e-11)
    edges = data["mst_edges"]
    assert edges.shape == (len(scores) - 1, 3)
    np.testing.assert_allclose(
        graph[edges[:, 0].astype(int), edges[:, 1].astype(int)], edges[:, 2], atol=1e-11
    )
    # Compare graph summaries using a distinct dense implementation.
    g = record["graph"]
    for key, expected in (
        ("core_min", core.min()),
        ("core_median", np.median(core)),
        ("core_max", core.max()),
    ):
        np.testing.assert_allclose(g[key], expected, atol=1e-11)
    if len(set(truth)) > 1:
        between = truth[:, None] != truth[None, :]
        for name, matrix in (("euclidean", d), ("mutual", graph)):
            bridge = matrix[between].min()
            bottle = max(
                minimum_spanning_tree(matrix[np.ix_(truth == k, truth == k)]).data.max()
                for k in set(truth)
            )
            np.testing.assert_allclose(g[name + "_between_bridge"], bridge)
            np.testing.assert_allclose(g[name + "_within_bottleneck"], bottle)
            np.testing.assert_allclose(g[name + "_separation_ratio"], bridge / bottle)
            assert bool(g[name + "_threshold_separable"]) == bool(bridge > bottle)
    nearest = data["neighbors"]
    _neighbors(scores, nearest)
    metrics = record["neighborhood"]
    np.testing.assert_allclose(
        metrics["cross_neighbor_fraction"], np.mean(truth[nearest] != truth[:, None])
    )
    from scipy.stats import spearmanr

    upper = np.triu_indices(len(scores), 1)
    np.testing.assert_allclose(
        metrics["distance_tie_excess"], len(d[upper]) - len(np.unique(d[upper]))
    )
    distance_without_self = d.copy()
    np.fill_diagonal(distance_without_self, np.inf)
    cutoff = np.partition(distance_without_self, 3, axis=1)[:, 3:4]
    assert metrics["cutoff_tie_rows"] == int(
        np.sum(
            np.sum(np.isclose(distance_without_self, cutoff, rtol=1e-12, atol=1e-12), axis=1) > 1
        )
    )
    for name in ("source_standard", "traits_standard"):
        reference = data[name + "_neighbors"]
        other = _neighbors(context[name], reference)
        overlaps = [
            sum(i in set(b) for i in a) / 4 for a, b in zip(nearest, reference, strict=True)
        ]
        np.testing.assert_allclose(metrics[name + "_neighbor_overlap"], np.mean(overlaps))
        np.testing.assert_allclose(
            metrics[name + "_distance_rank_correlation"],
            spearmanr(d[upper], other[upper]).statistic,
            atol=1e-10,
        )
    assert record["parameters"] == {
        "min_cluster_size": 5,
        "min_samples": 5,
        "alpha": 1,
        "epsilon": 0,
        "metric": "euclidean",
        "self_in_min_samples": True,
    }
    policies = record["policies"]
    for policy, result in policies.items():
        if result["status"] != "complete":
            assert record["hierarchy_status"] == "unavailable"
            continue
        labels, strength = data[policy + "_labels"], data[policy + "_strength"]
        assert labels.shape == strength.shape == truth.shape
        assert np.all(labels >= -1) and np.isfinite(strength).all()
        assert np.all((strength >= 0) & (strength <= 1)) and np.all(strength[labels < 0] == 0)
        assert result["selected_k"] == len(set(labels) - {-1})
        assert all(sum(labels == k) >= 5 for k in set(labels) - {-1})
        assert result["noise_count"] == sum(labels < 0)
        np.testing.assert_allclose(result["coverage"], np.mean(labels >= 0))
        predicted = labels.copy()
        predicted[labels < 0] = np.arange(sum(labels < 0)) + max(int(labels.max()), 0) + 1
        np.testing.assert_allclose(result["ari"], adjusted_rand_score(truth, predicted))
        if sum(labels >= 0) >= 2:
            np.testing.assert_allclose(
                result["assigned_ari"], adjusted_rand_score(truth[labels >= 0], labels[labels >= 0])
            )
        else:
            assert result["assigned_ari"] is None
        assert result["splits"] == sum(
            len(set(labels[(truth == k) & (labels >= 0)])) > 1 for k in set(truth)
        )
        assert result["merges"] == sum(len(set(truth[labels == k])) > 1 for k in set(labels) - {-1})
    if record["hierarchy_status"] != "available":
        return
    linkage, condensed = data["linkage"], data["condensed"]
    members = validate_linkage(linkage, len(scores))
    np.testing.assert_allclose(np.sort(linkage["value"]), np.sort(tree.data), atol=1e-11)
    assert np.all(np.diff(linkage["value"]) >= -1e-11)
    for row in linkage:
        a, b = [sorted(members[int(row[k])]) for k in ("left_node", "right_node")]
        np.testing.assert_allclose(graph[np.ix_(a, b)].min(), row["value"], atol=1e-11)
    _same_condensation(independent_condensation(linkage, len(scores), 5), condensed, len(scores))
    for policy, result in policies.items():
        method, root = POLICIES[policy]
        labels, strength = tree_to_labels(
            linkage,
            min_cluster_size=5,
            cluster_selection_method=method,
            allow_single_cluster=root,
            cluster_selection_epsilon=0.0,
        )
        np.testing.assert_array_equal(labels, data[policy + "_labels"])
        np.testing.assert_allclose(strength, data[policy + "_strength"], atol=1e-12)
        trace = result["selection"]
        nodes = {n["node"]: n for n in trace["nodes"]}
        actual_root = int(condensed["parent"].min())
        assert trace["root"] == actual_root
        assert set(nodes) == set(condensed["parent"])
        expected_frontier = {}
        for node in sorted(nodes, reverse=True):
            row = nodes[node]
            edges_from_node = condensed[condensed["parent"] == node]
            children = [int(e["child"]) for e in edges_from_node if e["cluster_size"] > 1]
            assert row["children"] == children
            incoming = condensed[condensed["child"] == node]
            if node == actual_root:
                assert row["birth"] == 0 and row["parent"] is None
            else:
                assert len(incoming) == 1
                assert row["parent"] == int(incoming[0]["parent"])
                np.testing.assert_allclose(row["birth"], incoming[0]["value"])
            chosen = (
                (node != actual_root or root) and row["own_stability"] >= row["best_child_sum"]
                if method == "eom"
                else node != actual_root and not children
            )
            expected_frontier[node] = (
                [node] if chosen else [c for child in children for c in expected_frontier[child]]
            )
        assert trace["selected_nodes"] == sorted(expected_frontier[actual_root])
        assert trace["root_selected"] == (actual_root in trace["selected_nodes"])
        np.testing.assert_allclose(trace["root_stability"], nodes[actual_root]["own_stability"])
        np.testing.assert_allclose(
            trace["root_best_child_sum"], nodes[actual_root]["best_child_sum"]
        )
        np.testing.assert_allclose(
            trace["root_margin"], trace["root_stability"] - trace["root_best_child_sum"]
        )
        for node, n in nodes.items():
            rows = condensed[condensed["parent"] == node]
            own = math.fsum((float(r["value"]) - n["birth"]) * int(r["cluster_size"]) for r in rows)
            np.testing.assert_allclose(n["own_stability"], own, atol=1e-9)
            child_sum = math.fsum(
                max(nodes[c]["own_stability"], nodes[c]["best_child_sum"]) for c in n["children"]
            )
            np.testing.assert_allclose(n["best_child_sum"], child_sum, atol=1e-9)
            if method == "eom":
                assert n["locally_selected"] == (
                    (node != trace["root"] or root) and own >= child_sum
                )
        if len(set(truth)) == 2:
            # Independent leaf-set correspondence, without selecting by truth.
            parent = {int(r["child"]): int(r["parent"]) for r in condensed}
            leaves = {int(k): set() for k in condensed["parent"]}
            for i in range(len(scores)):
                node = parent[i]
                while True:
                    leaves[node].add(i)
                    if node == trace["root"]:
                        break
                    node = parent[node]
            candidates = [k for k in leaves if k != trace["root"]]
            refs = [set(np.flatnonzero(truth == k)) for k in sorted(set(truth))]
            score = {
                k: [len(leaves[k] & ref) / len(leaves[k] | ref) for ref in refs] for k in candidates
            }
            best = max(
                (
                    min(score[a][0], score[b][1])
                    for a in candidates
                    for b in candidates
                    if not leaves[a] & leaves[b]
                ),
                default=0.0,
            )
            np.testing.assert_allclose(
                best, record["branch_reference"]["best_disjoint_min_jaccard"]
            )
            np.testing.assert_allclose(
                record["branch_reference"]["best_branch_jaccard"],
                [max((score[k][j] for k in candidates), default=0.0) for j in (0, 1)],
            )
