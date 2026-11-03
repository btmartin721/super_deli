"""Independent graph, moment, projection, likelihood and archival replay checks."""

import itertools
import math

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from super_deli.benchmarking.matched_reference import audit_integrative, audit_pca
from super_deli.benchmarking.moons_audit import _same_condensation, independent_condensation
from super_deli.benchmarking.moons_hierarchy import validate_linkage
from super_deli.benchmarking.neural_analysis import arrays, neural_forward, require
from super_deli.benchmarking.view_signal_coordinates import native_arm
from super_deli.io.config import read_json


def reference_ari(first, second):
    """Contingency-table Rand adjustment, independent of sklearn's metric."""
    table = np.zeros((len(np.unique(first)), len(np.unique(second))), dtype=np.int64)
    a, b = np.unique(first, return_inverse=True)[1], np.unique(second, return_inverse=True)[1]
    np.add.at(table, (a, b), 1)
    choose = lambda v: int(np.sum(v * (v - 1) // 2))  # noqa: E731
    pairs = len(first) * (len(first) - 1) // 2
    if pairs == 0:
        return 1.0
    observed = choose(table)
    rows, columns = choose(table.sum(1)), choose(table.sum(0))
    expected = rows * columns / pairs
    denominator = (rows + columns) / 2 - expected
    return (observed - expected) / denominator if denominator else 1.0


def audit_context(context, diagnostics, source, entry):
    from super_deli.benchmarking.matched_audit import audit_point, audit_sources
    from super_deli.benchmarking.neural_moons_coordinates import safe_path

    folder = safe_path(source, entry["path"])
    if entry.get("fixture"):
        original = arrays(folder / "context.npz")
        require(context.keys() == original.keys(), "Fixture array coverage")
        for key in original:
            np.testing.assert_array_equal(context[key], original[key])
        return []
    ref = audit_sources(folder, entry["task"])
    for key in ("morphology", "ecology", "latent_morphology", "latent_ecology", "truth"):
        np.testing.assert_array_equal(context[key], ref[key])
    n = entry["task"]["n"]
    require(context["unit_ids"].tolist() == [f"unit_{i:04d}" for i in range(n)], "Source IDs")
    for view in ("morphology", "ecology"):
        x = ref[view]
        np.testing.assert_allclose(
            context[view + "_standard"], (x - x.mean(0)) / x.std(0), atol=1e-12
        )
    for view in ("paired", "morphology", "ecology"):
        x = np.column_stack((ref["morphology"], ref["ecology"])) if view == "paired" else ref[view]
        for seed in (0, 101, 202):
            arm = f"{view}_{seed}"
            a = arrays(folder / arm / "point.npz")
            np.testing.assert_array_equal(a["kept_indices"], np.arange(x.shape[1]))
            z = (x - x.mean(0)) / x.std(0)
            weight = np.ones(x.shape[1])
            if view == "paired":
                weight[:4] = np.sqrt(12 / 8)
                weight[4:] = np.sqrt(12 / 16)
            z *= weight
            raw = neural_forward(z, a, "encoder")
            if view == "paired":
                np.testing.assert_allclose(context["balanced_traits"], z, atol=1e-11)
                np.testing.assert_allclose(context[arm + "_raw"], raw, atol=1e-11)
            np.testing.assert_allclose(
                context[arm + "_standard"], (raw - raw.mean(0)) / raw.std(0), atol=1e-10
            )
            recon = neural_forward(raw, a, "decoder")
            residual = (recon - z) / weight
            slices = (slice(0, 4), slice(4, 12)) if view == "paired" else (slice(None),)
            expected = [float(np.mean(residual[:, s] ** 2)) for s in slices]
            np.testing.assert_allclose(
                diagnostics[arm]["standardized_mse_by_view"], expected, atol=1e-11
            )
            centroid = [
                [np.mean((residual[ref["truth"] == g, s].mean(0)) ** 2) for s in slices]
                for g in range(4)
            ]
            np.testing.assert_allclose(
                diagnostics[arm]["centroid_mse_by_group_view"], centroid, atol=1e-11
            )
            np.testing.assert_allclose(
                diagnostics[arm]["final_weighted_mse"], np.mean((recon - z) ** 2), atol=1e-11
            )
    root = folder / "integrative_gmm"
    np.testing.assert_allclose(
        context["joint_pcs"], audit_integrative(ref["morphology"], ref["ecology"], root), atol=1e-10
    )
    fusion = arrays(root / "fusion.npz")
    np.testing.assert_array_equal(context["fused_profiles"], fusion["fused"])
    for i, view in enumerate(("morphology", "ecology")):
        profiles = arrays(root / f"view_{i:03d}" / "parameters" / "profiles.npz")["profiles"]
        np.testing.assert_array_equal(context[view + "_view_pcs"], profiles)
        a = arrays(folder / (view + "_density") / "point.npz")
        audit_pca(ref[view], a, context[view + "_density_pcs"])
    gmm = []
    for row in read_json(folder / "fits.json"):
        if row["kind"] != "point" or not row["arm"].endswith("gmm"):
            continue
        fit = folder / row["arm"]
        audit_point(entry["task"], ref, fit, row, {})  # No optimizer or estimator fitting.
        trace = arrays(fit / "candidate-trace.npz")
        candidates = read_json(fit / "candidates.json")
        baseline = candidates[0]["bic"]
        for candidate in candidates:
            k = candidate["k"]
            d = trace["scores"].shape[1]
            p = k * (d + d * (d + 1) // 2) + k - 1
            bic = candidate.get("bic")
            gmm.append(
                {
                    "arm": row["arm"],
                    **candidate,
                    "parameter_count": p,
                    "penalty": p * math.log(n),
                    "log_likelihood": (p * math.log(n) - bic) / 2 if bic is not None else None,
                    "bic_improvement_over_one": baseline - bic if bic is not None else None,
                    "selected": k == row["selected_k"],
                }
            )
    require(len(gmm) == 12, "Three unchanged four-candidate archival searches")
    return gmm


def audit_geometry(scores, context, metadata, data):
    """Alternative moment/projector arithmetic and explicit neighbor denominators."""
    truth = context["truth"]
    centered = scores - scores.mean(0)
    total = np.trace(centered.T @ centered)
    for key, labels in (
        ("scatter_joint", truth),
        ("scatter_axis_m", truth // 2),
        ("scatter_axis_e", truth % 2),
    ):
        within = sum(
            np.sum((scores[labels == g] - scores[labels == g].mean(0)) ** 2)
            for g in np.unique(labels)
        )
        np.testing.assert_allclose(metadata[key], 1 - within / total, atol=1e-11)
    u, values, _ = np.linalg.svd(centered, full_matrices=False)
    rank = int(sum(values > values.max() * max(centered.shape) * np.finfo(float).eps))
    for key, target in (
        ("linear_morphology", context["latent_morphology"]),
        ("linear_ecology", context["latent_ecology"]),
    ):
        y = target - target.mean(0)
        r2 = np.sum((u[:, :rank].T @ y) ** 2, axis=0) / np.sum(y**2, axis=0)
        require(metadata[key]["rank"] == rank, "Linear projection rank")
        np.testing.assert_allclose(metadata[key]["r2"], r2, atol=1e-10)
    distance = cdist(scores, scores)
    reference = cdist(context["morphology_standard"], context["morphology_standard"])
    np.fill_diagonal(distance, np.inf)
    np.fill_diagonal(reference, np.inf)
    for k in (5, 15):
        index = np.argsort(distance, axis=1)[:, :k]
        ref = np.argsort(reference, axis=1)[:, :k]
        purity, overlap = [], []
        ties = []
        for i, (ids, other) in enumerate(zip(index, ref, strict=True)):
            cutoff = distance[i, ids[-1]]
            tie = (
                sum(math.isclose(v, cutoff, rel_tol=1e-12, abs_tol=1e-12) for v in distance[i]) > 1
            )
            ref_cutoff = reference[i, other[-1]]
            ref_tie = (
                sum(math.isclose(v, ref_cutoff, rel_tol=1e-12, abs_tol=1e-12) for v in reference[i])
                > 1
            )
            ties.append(tie)
            if not tie:
                purity.extend(bool(truth[j] == truth[i]) for j in ids)
                if not ref_tie:
                    overlap.append(len(set(ids) & set(other)) / k)
        np.testing.assert_array_equal(data[f"neighbors_{k}"], index)
        np.testing.assert_array_equal(data[f"ties_{k}"], ties)
        m = metadata[f"neighbors_{k}"]
        require(
            m["available_rows"] == len(purity) // k and m["overlap_available_rows"] == len(overlap),
            "Neighbor denominators",
        )
        for key, values in (("purity", purity), ("morphology_overlap", overlap)):
            if values:
                np.testing.assert_allclose(m[key], np.mean(values), atol=1e-12)
            else:
                require(m[key] is None, "Tied neighborhoods remain unavailable")


def audit_branches(condensed, truth, record):
    # Walk ancestors from each individual, independently of bottom-up reporting.
    parent = {int(r["child"]): int(r["parent"]) for r in condensed}
    root = int(condensed["parent"].min())
    members = {int(i): set() for i in condensed["parent"]}
    for i in range(len(truth)):
        node = parent[i]
        while True:
            members[node].add(i)
            if node == root:
                break
            node = parent[node]
    members.pop(root)
    scores = {}
    for node, ids in members.items():
        scores[node] = []
        for g in range(4):
            reference = set(np.flatnonzero(truth == g))
            scores[node].append(len(ids & reference) / len(ids | reference))
    best = [max((s[g] for s in scores.values()), default=0) for g in range(4)]
    np.testing.assert_allclose(record["best_jaccard"], best, atol=1e-12)
    passing = [[node for node in scores if scores[node][g] >= 0.8] for g in range(4)]
    disjoint = any(
        len(set.union(*(members[node] for node in nodes)))
        == sum(len(members[node]) for node in nodes)
        for nodes in itertools.product(*passing)
    )
    require(record["four_disjoint_at_0_8"] == disjoint, "Four disjoint reference branches")


def audit_fit(context, name, record, data):
    require(__debug__, "Independent auditing requires Python assertions enabled")
    scores = context[name]
    np.testing.assert_array_equal(data["scores"], scores)
    if native_arm(name):
        for key in ("labels", "strength"):
            np.testing.assert_array_equal(data[key], context[name + "_" + key])
    d = cdist(scores, scores)
    core = np.partition(d, 4, axis=1)[:, 4]
    np.testing.assert_allclose(data["core_distances"], core, atol=1e-11)
    graph = np.maximum(d, np.maximum.outer(core, core))
    np.fill_diagonal(graph, 0)
    mst = minimum_spanning_tree(graph)
    np.testing.assert_allclose(np.sort(data["mst_edges"][:, 2]), np.sort(mst.data), atol=1e-11)
    require(record["recovery"]["selected_k"] == len(set(data["labels"]) - {-1}), "Cluster count")
    require(record["recovery"]["noise_count"] == int(sum(data["labels"] < 0)), "Noise accounting")
    labels, truth = data["labels"], context["truth"]
    selected = labels >= 0
    full = labels.copy()
    full[~selected] = np.arange(sum(~selected)) + max(int(labels.max()), 0) + 1
    np.testing.assert_allclose(record["recovery"]["ari"], reference_ari(truth, full), atol=1e-12)
    if sum(selected) >= 2:
        np.testing.assert_allclose(
            record["recovery"]["assigned_ari"],
            reference_ari(truth[selected], labels[selected]),
            atol=1e-12,
        )
    else:
        require(record["recovery"]["assigned_ari"] is None, "Assigned agreement denominator")
    np.testing.assert_allclose(record["recovery"]["coverage"], np.mean(data["labels"] >= 0))
    require(
        np.all(data["labels"] >= -1)
        and np.isfinite(data["strength"]).all()
        and np.all((data["strength"] >= 0) & (data["strength"] <= 1)),
        "Density ranges",
    )
    require(np.all(data["strength"][data["labels"] < 0] == 0), "Noise strength")
    require(
        record["parameters"]
        == {
            "min_cluster_size": 5,
            "min_samples": 5,
            "self_in_min_samples": True,
            "metric": "euclidean",
            "alpha": 1,
            "epsilon": 0,
            "method": "eom",
            "root": True,
        },
        "Unchanged density rule",
    )
    audit_geometry(scores, context, record["geometry"], data)
    if record["hierarchy_status"] != "available":
        require(bool(record.get("hierarchy_error")), "Unavailable hierarchy reason required")
        return
    linkage, condensed = data["linkage"], data["condensed"]
    members = validate_linkage(linkage, len(scores))
    np.testing.assert_allclose(np.sort(linkage["value"]), np.sort(mst.data), atol=1e-10)
    for row in linkage:
        left, right = [sorted(members[int(row[k])]) for k in ("left_node", "right_node")]
        np.testing.assert_allclose(graph[np.ix_(left, right)].min(), row["value"], atol=1e-11)
    _same_condensation(independent_condensation(linkage, len(scores), 5), condensed, len(scores))
    from sklearn.cluster._hdbscan._tree import tree_to_labels

    labels, strength = tree_to_labels(
        linkage,
        min_cluster_size=5,
        cluster_selection_method="eom",
        allow_single_cluster=True,
        cluster_selection_epsilon=0,
    )
    np.testing.assert_array_equal(labels, data["labels"])
    np.testing.assert_allclose(strength, data["strength"], atol=1e-12)
    nodes = {r["node"]: r for r in record["selection"]["nodes"]}
    for node, info in nodes.items():
        edges = condensed[condensed["parent"] == node]
        own = float(np.dot(edges["value"] - info["birth"], edges["cluster_size"]))
        np.testing.assert_allclose(own, info["own_stability"], rtol=1e-11, atol=1e-9)
        child_sum = math.fsum(
            max(nodes[c]["own_stability"], nodes[c]["best_child_sum"]) for c in info["children"]
        )
        np.testing.assert_allclose(child_sum, info["best_child_sum"], rtol=1e-11, atol=1e-9)
    audit_branches(condensed, context["truth"], record["branch_reference"])
