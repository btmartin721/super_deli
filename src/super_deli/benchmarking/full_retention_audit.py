"""Independent raw-transform, density, eligibility, projection and denominator audits."""

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import logsumexp

from super_deli.benchmarking.multimodal_validation import array_hash
from super_deli.benchmarking.signal_loss_fit import audit_candidates
from super_deli.io.config import read_json


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)


def _shape(raw, consensus, reflection):
    centered = raw - raw.mean(axis=1, keepdims=True)
    unit = centered / np.sqrt(np.sum(centered**2, axis=(1, 2)))[:, None, None]
    aligned = []
    for value in unit:
        u, _, vt = np.linalg.svd(value.T @ consensus)
        if reflection == "forbid" and np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        aligned.append(value @ u @ vt)
    aligned = np.asarray(aligned)
    cosine = np.einsum("nld,ld->n", aligned, consensus)
    return (aligned - cosine[:, None, None] * consensus).reshape(len(raw), -1)


def _view(source, model, arrays, prefix, layout, unit_ids, draw):
    targets = layout["target_ids"]
    assert len(targets) == len(set(targets)) and set(targets) == set(unit_ids)
    lookup = {u: i for i, u in enumerate(targets)}
    expected = [lookup[unit_ids[i]] for i in draw]
    indices = arrays[prefix + "indices"]
    np.testing.assert_array_equal(indices, expected)
    raw = source[indices]
    assert array_hash(raw) == model["raw_training_sha256"]
    c = model["configuration"]
    if model["kind"] == "morphometric":
        consensus = arrays[prefix + "consensus"]
        close(consensus.mean(0), 0)
        close(np.linalg.norm(consensus), 1)
        full = _shape(source, consensus, c["shape"]["reflection"])
        matrix = full[indices]
    else:
        mean, variance = raw.mean(0), raw.var(0)
        kept = np.flatnonzero(variance > 0)
        np.testing.assert_array_equal(kept, arrays[prefix + "kept"])
        close(mean[kept], arrays[prefix + "feature_means"])
        scaling = c["scaling"] or ("patterson" if model["kind"] == "genomic" else "standard")
        frequency = mean[kept] / 2
        scale = (
            np.sqrt(2 * frequency * (1 - frequency))
            if scaling == "patterson"
            else (np.sqrt(variance[kept]) if scaling == "standard" else np.ones(len(kept)))
        )
        close(scale, arrays[prefix + "feature_scales"])
        full = (source[:, kept] - mean[kept]) / scale
        matrix = full[indices]
    mean, components = arrays[prefix + "pca_mean"], arrays[prefix + "components"]
    close(matrix.mean(0), mean)
    close(components @ components.T, np.eye(len(components)))
    all_scores = (full - mean) @ components.T
    train = all_scores[indices]
    eigenvalues = np.linalg.eigvalsh(np.cov(matrix, rowvar=False))[::-1]
    close(train.var(axis=0, ddof=1), eigenvalues[: len(components)])
    close(arrays[prefix + "variance"], eigenvalues[: len(components)])
    close(train, arrays[prefix + "profiles"])
    m = train.mean(0)
    scale = np.sqrt(np.sum((train - m) ** 2) / (len(train) - 1))
    close(m, arrays[prefix + "mean"])
    close(scale, arrays[prefix + "scale"])
    weight = float(arrays[prefix + "weight"])
    return (train - m) * np.sqrt(weight) / scale, (all_scores - m) * np.sqrt(weight) / scale


def _search(event, candidates, directory, draw):
    c = event["configuration"]
    observed = {r["k"]: r for r in candidates[event["candidate_start"] : event["candidate_stop"]]}
    eligible = []
    for row in event["candidates"]:
        k = row["k"]
        if len(set(draw)) < k * c["min_cluster_size"]:
            assert row["status"] == "insufficient_samples" and k not in observed
            continue
        fit = observed[k]
        assert fit["fit_status"] == "complete" and fit["converged"]
        assert fit["n_init"] == c["n_init"]
        assert fit["seed"] == int(np.random.SeedSequence([event["seed"], k]).generate_state(1)[0])
        with np.load(directory / fit["arrays"], allow_pickle=False) as a:
            labels = a["labels"]
            np.testing.assert_array_equal(
                np.bincount(labels, minlength=k), row["hard_cluster_sizes"]
            )
            distinct = [len(set(np.asarray(draw)[labels == j])) for j in range(k)]
            assert distinct == row["distinct_unit_sizes"]
        status = "eligible" if min(distinct) >= c["min_cluster_size"] else "small_component"
        assert row["status"] == status
        close(row["bic"], fit["bic"])
        if status == "eligible":
            eligible.append((row["bic"], k))
    assert min(eligible)[1] == event["selected_k"]
    return observed[event["selected_k"]]


def audit_run(directory, *, search_auditor=None, candidate_auditor=None):
    directory = Path(directory)
    root = directory / "observed-pipeline"
    if not (root / "raw-layout.json").exists():
        candidates = read_json(directory / "observed-candidates/index.json")
        assert not candidates
        return dict(
            passed=True,
            point_withheld=True,
            component_checks=0,
            events=0,
            stage="before_numerical_fit",
        )
    layout = read_json(root / "raw-layout.json")
    events = read_json(root / "events.json")
    models = read_json(directory / "observed-candidates/index.json")
    checked = (candidate_auditor or audit_candidates)(directory / "observed-candidates")
    density_checks = sum(r["fit_status"] == "complete" for r in models)
    failed_components = len(models) - density_checks
    unit_ids = layout["unit_ids"]
    n = len(unit_ids)
    raw = [np.load(root / f"raw_{i}.npy", allow_pickle=False) for i in range(len(layout["views"]))]
    for i, view in enumerate(layout["views"]):
        np.testing.assert_array_equal(
            raw[i], np.load(directory / f"view_{i:03d}/source_values.npy", allow_pickle=False)
        )
        binding = read_json(directory / f"view_{i:03d}/source_rows.json")
        assert binding["unit_ids"] == view["unit_ids"]
        assert binding["target_ids"] == view["target_ids"]
        assert binding["observation_ids"] == view["observation_ids"]
    projections, shape_checks = {}, []
    retained_checks = 0
    for event in events:
        draw = event["draw"]
        with np.load(root / event["arrays"], allow_pickle=False) as loaded:
            a = {k: loaded[k].copy() for k in loaded.files}
        for i, attempted in enumerate(event["attempted_views"]):
            lookup = {u: j for j, u in enumerate(layout["views"][i]["target_ids"])}
            expected_hash = array_hash(raw[i][[lookup[unit_ids[j]] for j in draw]])
            assert attempted["raw_training_sha256"] == expected_hash
            if attempted["kind"] == "morphometric":
                shape_checks.append(expected_hash)
        training, projected = [], []
        weights = np.asarray([v["weight"] for v in event["configuration"]["views"]], dtype=float)
        weights /= weights.sum()
        for i, view in enumerate(event["views"]):
            close(a[f"v{i}_weight"], weights[i])
            x, y = _view(raw[i], view, a, f"v{i}_", layout["views"][i], unit_ids, draw)
            lookup = {u: j for j, u in enumerate(layout["views"][i]["target_ids"])}
            training.append(x)
            projected.append(y[[lookup[u] for u in unit_ids]])
        if event["status"] != "complete":
            continue
        fused = np.concatenate(training, axis=1)
        close(fused, a["fused"])
        close(fused.mean(0), a["pca_mean"])
        pc = a["components"]
        close(pc @ pc.T, np.eye(len(pc)))
        close((fused - a["pca_mean"]) @ pc.T, a["scores"])
        close(
            a["scores"].var(axis=0, ddof=1),
            np.linalg.eigvalsh(np.cov(fused, rowvar=False))[::-1][: len(pc)],
        )
        close((np.concatenate(projected, axis=1) - a["pca_mean"]) @ pc.T, a["projected"])
        if event["configuration"]["n_pcs"] == fused.shape[1]:
            close(pdist(fused), pdist(a["scores"]))
            retained_checks += 1
        selected = (search_auditor or _search)(
            event, models, directory / "observed-candidates", draw
        )
        with np.load(
            directory / "observed-candidates" / selected["arrays"], allow_pickle=False
        ) as b:
            terms = []
            for weight, mean in zip(b["weights"], b["means"], strict=True):
                residual = a["projected"] - mean
                cov = b["covariances"]
                terms.append(
                    np.log(weight)
                    - 0.5
                    * (
                        len(mean) * np.log(2 * np.pi)
                        + np.linalg.slogdet(cov)[1]
                        + np.einsum("ij,ji->i", residual, np.linalg.solve(cov, residual.T))
                    )
                )
            terms = np.asarray(terms).T
            close(np.exp(terms - logsumexp(terms, axis=1)[:, None]), a["projected_membership"])
            np.testing.assert_array_equal(terms.argmax(1), a["projected_labels"])
        projections[event["index"]] = a["projected_labels"]
    shapes = read_json(directory / "observed-shape.json")
    assert [r["raw_sha256"] for r in shapes] == shape_checks
    if not events or events[0]["status"] != "complete":
        return dict(passed=True, point_withheld=True, component_checks=checked, events=len(events))
    summary = read_json(directory / "summary.json")
    replicates = read_json(directory / "bootstrap_replicates.json")
    c = events[0]["configuration"]
    children = np.random.SeedSequence(c["random_seed"]).spawn(c["n_bootstrap"] + 1)
    assert len(events) == len(replicates) + 1 == c["n_bootstrap"] + 1
    assert events[0]["seed"] == int(children[0].generate_state(1)[0])
    np.testing.assert_array_equal(events[0]["draw"], np.arange(n))
    co, oob_count, requested, successful = [np.zeros((n, n), dtype=np.uint64) for _ in range(4)]
    ks = Counter()
    with np.load(directory / "mixture.npz", allow_pickle=False) as a:
        point_labels = a["labels"].copy()
    groups = [frozenset(np.flatnonzero(point_labels == j)) for j in range(summary["selected_k"])]
    recovery_counts = np.zeros(len(groups), dtype=int)
    for i, (event, rep, child) in enumerate(zip(events[1:], replicates, children[1:], strict=True)):
        draw_seed, seed = map(int, child.generate_state(2))
        draw = np.random.default_rng(draw_seed).integers(n, size=n)
        assert (
            rep["replicate"] == i
            and rep["draw_seed"] == draw_seed
            and rep["fit_seed"] == seed == event["seed"]
        )
        np.testing.assert_array_equal(rep["training_indices"], draw)
        np.testing.assert_array_equal(event["draw"], draw)
        absent = np.bincount(draw, minlength=n) == 0
        assert rep["oob_indices"] == np.flatnonzero(absent).tolist()
        assert rep["unique_training_units"] == len(set(draw))
        pairs = np.logical_and.outer(absent, absent)
        requested += pairs
        assert rep["status"] == event["status"]
        if event["status"] == "complete":
            np.testing.assert_array_equal(rep["labels"], projections[i + 1])
            assert rep["selected_k"] == event["selected_k"]
            same = np.equal.outer(rep["labels"], rep["labels"])
            co += same
            oob_count += same & pairs
            successful += pairs
            ks[rep["selected_k"]] += 1
            partitions = {
                frozenset(np.flatnonzero(np.asarray(rep["labels"]) == j))
                for j in range(rep["selected_k"])
            }
            recovery_counts += [g in partitions for g in groups]
    assert summary["bootstrap"]["successful"] == sum(ks.values())
    assert summary["bootstrap"]["requested"] == len(replicates)
    assert summary["bootstrap"]["failed"] == len(replicates) - sum(ks.values())
    assert summary["bootstrap"]["selected_k_counts"] == {str(k): v for k, v in sorted(ks.items())}
    clusters = read_json(directory / "clusters.json")
    for i, cluster in enumerate(clusters):
        assert cluster["exact_recovery_count"] == (int(recovery_counts[i]) if replicates else None)
    if replicates:
        with np.load(directory / "stability.npz", allow_pickle=False) as a:
            for key, value in (
                ("coassignment_counts", co),
                ("oob_coassignment_counts", oob_count),
                ("oob_requested", requested),
                ("oob_successful", successful),
            ):
                np.testing.assert_array_equal(a[key], value)
            failures = len(replicates) - sum(ks.values())
            low, high = co / len(replicates), (co + failures) / len(replicates)
            np.fill_diagonal(low, 1)
            np.fill_diagonal(high, 1)
            close(low, a["coassignment_lower"])
            close(high, a["coassignment_upper"])
            close(low if failures == 0 else np.full((n, n), np.nan), a["coassignment"])
            for key, numerator in (
                ("oob_coassignment_lower", oob_count),
                ("oob_coassignment_upper", oob_count + requested - successful),
            ):
                expected = np.full((n, n), np.nan)
                np.divide(numerator, requested, out=expected, where=requested > 0)
                close(expected, a[key])
            lower = np.full((n, n), np.nan)
            np.divide(oob_count, requested, out=lower, where=requested > 0)
            close(np.where(requested == successful, lower, np.nan), a["oob_coassignment"])
    return dict(
        passed=True,
        point_withheld=False,
        component_checks=checked,
        events=len(events),
        draws=len(replicates),
        failed_draws=len(replicates) - sum(ks.values()),
        shape_checks=len(shapes),
        shape_complete_checks=sum(r["status"] == "complete" for r in shapes),
        density_checks=density_checks,
        failed_component_attempts=failed_components,
        retained_space_checks=retained_checks,
    )
