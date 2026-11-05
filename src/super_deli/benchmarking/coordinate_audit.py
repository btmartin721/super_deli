"""Independent geometry, frozen-transform, mixture density and criterion audits."""

import math
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logsumexp
from sklearn.cluster._hdbscan._tree import tree_to_labels

from super_deli.benchmarking.coordinate_config import CLUSTERERS, SPACES, MixtureSettings
from super_deli.benchmarking.coordinate_fit import frozen_only
from super_deli.benchmarking.moons_hierarchy import validate_linkage
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_coordinates import safe_path
from super_deli.benchmarking.view_weight_audit import audit_density
from super_deli.io.config import read_json
from super_deli.models.tabular_neural import forward_numpy

PROBABILITY_ATOL = 1e-8
CRITERION_ATOL = 1e-5


def audit_selection(condensed, trace, method, allow_root):
    root = int(condensed["parent"].min())
    expected, frontier = {}, {}
    recorded = {row["node"]: row for row in trace["nodes"]}
    require(set(recorded) == set(condensed["parent"]), "Every hierarchy branch audited")
    for node in sorted(recorded, reverse=True):
        outgoing = condensed[condensed["parent"] == node]
        incoming = condensed[condensed["child"] == node]
        birth = 0.0 if node == root else float(incoming[0]["value"])
        children = [int(row["child"]) for row in outgoing if row["cluster_size"] > 1]
        own = math.fsum(
            (float(row["value"]) - birth) * int(row["cluster_size"]) for row in outgoing
        )
        child_sum = math.fsum(expected[c] for c in children)
        allowed = node != root or allow_root
        choose = (
            (allowed and own >= child_sum) if method == "eom" else node != root and not children
        )
        expected[node] = max(own, child_sum) if allowed else child_sum
        frontier[node] = [node] if choose else [c for child in children for c in frontier[child]]
        actual = recorded[node]
        require(
            actual["children"] == children and actual["locally_selected"] == choose,
            "Hierarchy selection decision",
        )
        require(
            actual["parent"] == (None if node == root else int(incoming[0]["parent"])),
            "Hierarchy branch ancestry",
        )
        np.testing.assert_allclose(
            [actual["birth"], actual["own_stability"], actual["best_child_sum"], actual["margin"]],
            [birth, own, child_sum, own - child_sum],
            rtol=1e-10,
            atol=1e-9,
        )
    require(
        trace["root"] == root and trace["selected_nodes"] == sorted(frontier[root]),
        "Independent hierarchy frontier",
    )
    require(trace["root_selected"] == (root in frontier[root]), "Root selection record")
    np.testing.assert_allclose(
        [trace["root_stability"], trace["root_best_child_sum"], trace["root_margin"]],
        [
            recorded[root]["own_stability"],
            recorded[root]["best_child_sum"],
            recorded[root]["margin"],
        ],
        atol=1e-9,
    )


def reference_scores(raw, a, space):
    """Separate forward implementation and einsum projections from saved parameters."""
    if space.startswith("joint"):
        x = np.concatenate([raw[role] for role in ("morphology", "ecology")], axis=1)
        z = ((x - a["means"]) / a["scales"])[:, a["kept_indices"]] * a["column_weights"]
        raw_scores = forward_numpy(z, a, "encoder")
        return (raw_scores - a["latent_means"]) / a["latent_scales"]
    result = []
    for role in ("morphology", "ecology"):
        b = {k.split("__", 1)[1]: v for k, v in a.items() if k.startswith(role + "__")}
        z = ((raw[role] - b["means"]) / b["scales"])[:, b["kept_indices"]]
        value = (
            np.einsum("ij,kj->ik", z - b["pca_mean"], b["components"])
            if space == "pca4"
            else forward_numpy(z, b, "encoder")
        )
        result.append((value - b["latent_means"]) / b["latent_scales"])
    return np.concatenate(result, axis=1)


def gaussian_reference(scores, a, k, kind):
    means, weights, cov = a[f"k{k}_means"], a[f"k{k}_weights"], a[f"k{k}_covariances"]
    n, d = scores.shape
    require(
        means.shape == (k, d) and weights.shape == (k,) and np.all(weights > 0),
        "Candidate mean/weights",
    )
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
    require(
        cov.shape == ((k, d, d) if kind == "full" else (d, d)), "Prespecified covariance family"
    )
    densities = []
    for j in range(k):
        sigma = cov[j] if kind == "full" else cov
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)
        require(
            np.isfinite(sigma).all() and np.linalg.eigvalsh(sigma).min() > 0,
            "Positive definite fitted covariance",
        )
        delta = scores - means[j]
        sign, logdet = np.linalg.slogdet(sigma)
        require(sign == 1, "Gaussian determinant")
        quadratic = np.einsum("ij,ji->i", delta, np.linalg.solve(sigma, delta.T))
        densities.append(np.log(weights[j]) - 0.5 * (d * np.log(2 * np.pi) + logdet + quadratic))
    log_components = np.column_stack(densities)
    marginal = logsumexp(log_components, axis=1)
    probabilities = np.exp(log_components - marginal[:, None])
    ll = float(marginal.sum())
    parameters = (k - 1) + k * d + (k if kind == "full" else 1) * d * (d + 1) // 2
    return probabilities, ll, parameters * np.log(n) - 2 * ll, parameters


def audit_mixture(scores, row, a):
    kind = "full" if row["clusterer"] == "gmm_full" else "tied"
    require(row["configuration"] == MixtureSettings(kind).to_dict(), "Unchanged GMM settings")
    attempts = row["candidate_attempts"]
    require(
        [r["k"] for r in attempts] == list(range(1, len(attempts) + 1)) and len(attempts) <= 4,
        "Each candidate attempted once; none replaced",
    )
    complete = {}
    verified = 0
    for attempt in attempts:
        k = attempt["k"]
        require(
            attempt["covariance_type"] == kind
            and attempt["n_init"] == 10
            and attempt["max_iter"] == 500
            and attempt["tol"] == 0.001
            and attempt["reg_covar"] == 1e-6,
            "Prespecified numerical optimization settings",
        )
        expected = int(np.random.SeedSequence([71323, k]).generate_state(1)[0])
        require(attempt["random_state"] == expected, "Unchanged per-K initialization seed")
        require(
            attempt["status"] in {"complete", "failed"}, "Complete or explicitly failed candidate"
        )
        for key, value in a.items():
            if key.startswith(f"k{k}_"):
                require(np.isfinite(value).all(), "Retained candidate finite values")
        if attempt["status"] == "failed":
            require(
                row["status"] == "failed" and bool(attempt.get("error")),
                "Failed candidate withholds search",
            )
            continue
        require(
            attempt["converged"] and 1 <= attempt["iterations"] <= 500,
            "Candidate convergence metadata",
        )
        probability, ll, bic, p = gaussian_reference(scores, a, k, kind)
        np.testing.assert_allclose(
            probability, a[f"k{k}_membership"], rtol=1e-8, atol=PROBABILITY_ATOL
        )
        labels = a[f"k{k}_membership"].argmax(1)
        np.testing.assert_array_equal(labels, a[f"k{k}_labels"])
        chosen = probability[np.arange(len(scores)), labels]
        require(
            np.all(probability.max(1) - chosen <= PROBABILITY_ATOL),
            "Independent membership maximizers within fixed tolerance",
        )
        np.testing.assert_allclose(
            [ll, bic], [attempt["log_likelihood"], attempt["bic"]], rtol=0, atol=CRITERION_ATOL
        )
        sizes = np.bincount(labels, minlength=k)
        complete[k] = dict(
            bic=attempt["bic"],
            hard_cluster_sizes=sizes.tolist(),
            eligible=sizes.min() >= 5,
            dense_log_likelihood=ll,
            parameter_count=p,
        )
        verified += 1
    if row["status"] == "available":
        require(len(complete) == 4, "Complete configured K search")
        for candidate in row["candidates"]:
            actual = complete[candidate["k"]]
            require(
                candidate["status"] == ("eligible" if actual["eligible"] else "small_component"),
                "Minimum hard size eligibility",
            )
            require(
                candidate["hard_cluster_sizes"] == actual["hard_cluster_sizes"],
                "Hard component accounting",
            )
            np.testing.assert_allclose(candidate["bic"], actual["bic"], atol=1e-10)
        winner = min((r["bic"], k) for k, r in complete.items() if r["eligible"])[1]
        require(winner == row["selected_k"], "BIC, not reference labels, selects K")
        labels = a[f"k{winner}_labels"]
        order = sorted(range(winner), key=lambda k: int(np.flatnonzero(labels == k)[0]))
        require(order == row["component_order"], "Canonical selected label order")
        np.testing.assert_array_equal(a["membership"], a[f"k{winner}_membership"][:, order])
        np.testing.assert_array_equal(a["labels"], a["membership"].argmax(1))
    else:
        require(
            row["status"] == "failed" and bool(row.get("error")) and row["selected_k"] is None,
            "Withheld incomplete searches do not become K=1",
        )
        require(
            "labels" not in a and "membership" not in a, "No completed output after search failure"
        )
    return verified, complete


def audit_geometry(scores, reference, data, record):
    distance = np.linalg.norm(scores[:, None, :] - scores[None, :, :], axis=2)
    np.testing.assert_allclose(data["distances"], distance, atol=1e-11)
    n = len(scores)
    neighbors = data["neighbors_15"]
    require(
        neighbors.shape == (n, 15) and np.all(neighbors != np.arange(n)[:, None]),
        "Other-unit neighbors",
    )
    for i in range(n):
        expected = sorted((j for j in range(n) if j != i), key=lambda j: (distance[i, j], j))[:15]
        np.testing.assert_array_equal(neighbors[i], expected)
    core = np.sort(distance, axis=1)[:, 4]
    np.testing.assert_allclose(core, data["core_distances"], atol=1e-11)
    graph = np.maximum(np.maximum(core[:, None], core[None, :]), distance)
    np.fill_diagonal(graph, 0)
    edge = data["mst_edges"]
    np.testing.assert_allclose(
        np.sort(edge[:, 2]), np.sort(minimum_spanning_tree(graph).data), atol=1e-11
    )
    partitions = (
        {"common": reference["truth"]}
        if "truth" in reference
        else {role: reference[role + "_truth"] for role in ("morphology", "ecology")}
    )
    nodes = validate_linkage(data["linkage"], n)
    centered = scores - scores.mean(0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    rank = int(sum(singular > np.finfo(float).eps * max(centered.shape) * singular[0]))
    for role in ("morphology", "ecology"):
        target = reference["latent_" + role] - reference["latent_" + role].mean(0)
        residual = target - u[:, :rank] @ (u[:, :rank].T @ target)
        expected_r2 = 1 - (residual * residual).sum(0) / (target * target).sum(0)
        require(record["linear_" + role]["rank"] == rank, "Independent linear-recovery rank")
        np.testing.assert_allclose(record["linear_" + role]["r2"], expected_r2, atol=1e-9)
    for name, truth in partitions.items():
        r = record["references"][name]
        groups = np.unique(truth)
        for k in (5, 15):
            expected = (
                float(np.mean(truth[neighbors[:, :k]] != truth[:, None]))
                if len(groups) > 1
                else None
            )
            require(
                r["graph"]["neighbor_impurity"][str(k)] == expected,
                "Reference-only neighborhood mixing",
            )
        if len(groups) < 2:
            continue
        between = truth[:, None] != truth[None, :]
        for title, matrix in (("euclidean", distance), ("mutual", graph)):
            bridge = float(matrix[between].min())
            within = []
            for group in groups:
                ids = np.flatnonzero(truth == group)
                tree = minimum_spanning_tree(matrix[np.ix_(ids, ids)])
                within.append(float(tree.data.max()) if tree.nnz else 0.0)
            bottleneck = max(within)
            np.testing.assert_allclose(
                [r["graph"][title + "_between_bridge"], r["graph"][title + "_within_bottleneck"]],
                [bridge, bottleneck],
                atol=1e-11,
            )
            require(
                r["graph"][title + "_threshold_separable"] == (bridge > bottleneck),
                "Single-threshold separation diagnostic",
            )
        pure = []
        for group in groups:
            ids = set(np.flatnonzero(truth == group))
            pure.append(max(len(v) for v in nodes.values() if v <= ids) / len(ids))
        np.testing.assert_allclose(
            pure, r["branches"]["largest_pure_linkage_fraction"], atol=0, rtol=0
        )
        parent = {int(row["child"]): int(row["parent"]) for row in data["condensed"]}
        root = int(data["condensed"]["parent"].min())
        leaves = {int(node): set() for node in data["condensed"]["parent"]}
        for i in range(n):
            node = parent[i]
            while True:
                leaves[node].add(i)
                if node == root:
                    break
                node = parent[node]
        expected = []
        for group in groups:
            ids = set(np.flatnonzero(truth == group))
            expected.append(
                max(
                    (len(v & ids) / len(v | ids) for node, v in leaves.items() if node != root),
                    default=0.0,
                )
            )
        np.testing.assert_allclose(expected, r["branches"]["best_branch_jaccard"], atol=1e-12)


def audit_dataset(folder, task, *, cache, source):
    folder = Path(folder)
    base = safe_path(source, "main/" + task["key"])
    rows = read_json(folder / "fits.json")
    require(
        [r["arm"] for r in rows] == [s + "__" + c for s in SPACES for c in CLUSTERERS],
        "Complete frozen arm grid",
    )
    results = []
    with frozen_only():
        raw = arrays(base / "generating-reference.npz")
        for space in SPACES:
            original = arrays(base / space / "point.npz")
            scores = arrays(folder / space / "coordinates.npz")["scores"]
            np.testing.assert_array_equal(scores, original["scores"])
            np.testing.assert_allclose(
                scores, reference_scores(raw, original, space), rtol=1e-10, atol=1e-10
            )
            for clusterer in CLUSTERERS:
                row = next(r for r in rows if r["arm"] == space + "__" + clusterer)
                stored = read_json(folder / space / clusterer / "execution.json")
                require(
                    row == stored and not row["numerical_guard_failed"],
                    "Bound complete model attempt",
                )
                a = arrays(folder / space / clusterer / "fit.npz")
                repeated = folder.parent.parent / "preflight" / task["key"] / space / clusterer
                if folder.parent.name == "main" and repeated.is_dir():
                    previous = arrays(repeated / "fit.npz")
                    require(a.keys() == previous.keys(), "Repeated frozen evaluation coverage")
                    for key in a:
                        np.testing.assert_array_equal(a[key], previous[key])
                    before = read_json(repeated / "execution.json")
                    require(
                        (row["status"], row["selected_k"])
                        == (before["status"], before["selected_k"]),
                        "Repeated selection/failure parity",
                    )
                verified = 0
                if clusterer == "hdbscan":
                    require(
                        row["status"] == "available" and row["source_replay"],
                        "Unchanged density replay",
                    )
                    np.testing.assert_array_equal(a["labels"], original["labels"])
                    np.testing.assert_array_equal(a["strength"], original["strength"])
                    audit_density(
                        dict(scores=scores, labels=a["labels"], strength=a["strength"]),
                        {"fit_0_linkage": a["linkage"], "fit_0_condensed": a["condensed"]},
                        0,
                    )
                    for name, method, root in (
                        ("baseline", "eom", True),
                        ("no_root", "eom", False),
                        ("leaf", "leaf", True),
                    ):
                        labels, strength = tree_to_labels(
                            a["linkage"],
                            min_cluster_size=5,
                            cluster_selection_method=method,
                            allow_single_cluster=root,
                            cluster_selection_epsilon=0.0,
                        )
                        np.testing.assert_array_equal(a[name + "_labels"], labels)
                        np.testing.assert_allclose(a[name + "_strength"], strength, atol=1e-12)
                        audit_selection(a["condensed"], row["policies"][name], method, root)
                    audit_geometry(scores, raw, a, read_json(folder / space / "geometry.json"))
                else:
                    verified, _ = audit_mixture(scores, row, a)
                results.append(
                    dict(
                        neural_fits=0,
                        verified_updates=0,
                        draws=0,
                        candidate_densities_checked=verified,
                        density_replays=int(clusterer == "hdbscan"),
                        failed=int(row["status"] != "available"),
                    )
                )
    return results
