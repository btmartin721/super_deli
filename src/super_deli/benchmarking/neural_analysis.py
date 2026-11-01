"""Read-only numerical, source, replay and denominator checks for the neural study."""

from __future__ import annotations

import csv
import json

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arrays(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def reference(task):
    """Reconstruct the declared sampling laws without the production generator."""
    import numpy as np

    rng = np.random.default_rng(task["data_seed"])
    n, case = task["n"], task["case"]
    if case == "gaussian":
        z, truth = rng.normal(size=(n, 2)), np.zeros(n, dtype=int)
    elif case == "four_spherical":
        centers = np.array([[-3, -3], [-3, 3], [3, -3], [3, 3]])
        z = np.concatenate([rng.multivariate_normal(c, np.eye(2), size=n // 4) for c in centers])
        truth = np.repeat(np.arange(4), n // 4)
    elif case == "moons":
        a, b = rng.uniform(0, np.pi, (2, n // 2))
        z = np.concatenate(
            [
                np.column_stack([np.cos(a), np.sin(a)]),
                np.column_stack([1 - np.cos(b), 0.5 - np.sin(b)]),
            ]
        )
        z += rng.normal(0, 0.08, z.shape)
        truth = np.repeat(np.arange(2), n // 2)
    elif case == "cline":
        a = rng.uniform(-3, 3, n)
        z, truth = np.column_stack([a, 0.5 * a + rng.normal(0, 0.3, n)]), None
    else:
        raise ValueError("Unknown reference law")
    order = rng.permutation(n)
    z = z[order]
    truth = truth[order] if truth is not None else None
    if case == "moons":
        a, b = z[:, 0], z[:, 1]
        mapped = np.array([a, b, a * b, a * a, b * b, np.sin(a), np.cos(b), np.sin(a + b)]).T
    else:
        transform = np.array(
            [[1, 0], [0, 1], [1, 1], [1, -1], [0.5, 1], [-1, 0.5], [2, 0.5], [0.5, -2]]
        )
        mapped = np.einsum("ij,kj->ik", z, transform)
    noise = np.random.default_rng(task["noise_seed"]).normal(0, 0.15, mapped.shape)
    return mapped + noise, z, truth


def read_raw(directory):
    import numpy as np

    with (directory / "traits.tsv").open(newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    require(rows[0] == ["observation_id", *[f"trait_{i}" for i in range(1, 9)]], "Trait header")
    return [r[0] for r in rows[1:]], np.array([[float(v) for v in r[1:]] for r in rows[1:]])


def neural_forward(x, a, side):
    """Separate row-vector evaluation of each stored linear/tanh layer."""
    import numpy as np

    result = np.asarray(x)
    for layer in range(3):
        result = np.einsum("ij,kj->ik", result, a[f"{side}_{layer}_weight"])
        result = result + a[f"{side}_{layer}_bias"][None, :]
        if layer != 2:
            result = np.tanh(result)
    return result


def audit_point(x, directory, arm, record):
    import numpy as np
    from scipy.special import logsumexp

    summary = read_json(directory / "summary.json")
    a = arrays(directory / ("mixture.npz" if arm == "gmm" else "point.npz"))
    if arm == "gmm":
        a["mixture_means"] = a["means"].copy()
        a.update(arrays(directory / "preprocessing.npz"))
    ids = a["kept_indices"]
    np.testing.assert_array_equal(ids, np.arange(8))
    np.testing.assert_allclose(a["means"], x.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(a["scales"], x.std(axis=0), atol=1e-12)
    standardized = (x[:, ids] - x[:, ids].mean(axis=0)) / x[:, ids].std(axis=0)
    if arm.startswith("neural_"):
        raw = neural_forward(standardized, a, "encoder")
        np.testing.assert_allclose(a["latent_means"], raw.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(a["latent_scales"], raw.std(axis=0), atol=1e-12)
        scores = (raw - raw.mean(axis=0)) / raw.std(axis=0)
        require(np.linalg.matrix_rank(scores) == 2, "Neural rank")
        reconstructed = neural_forward(raw, a, "decoder")
        require(a["losses"].shape == (501,) and np.isfinite(a["losses"]).all(), "Loss history")
        np.testing.assert_allclose(
            a["losses"][-1], np.mean((reconstructed - standardized) ** 2), atol=1e-12
        )
        require(
            summary["configuration"]["model"]["random_seed"] == int(arm.split("_")[1]),
            "Neural seed",
        )
        require(summary["configuration"]["training"]["updates"] == 500, "Neural update budget")
        require(summary["species_significance"]["status"] == "not_computed", "Neural significance")
        for name, digest in summary["artifact_sha256"].items():
            require(fingerprint_file(directory / name).sha256 == digest, "Neural artifact hash")
    else:
        scores = (standardized - a["pca_mean"]) @ a["components"].T
        np.testing.assert_allclose(a["components"] @ a["components"].T, np.eye(2), atol=1e-12)
        spectrum = np.linalg.eigvalsh(np.cov(standardized, rowvar=False))[::-1]
        np.testing.assert_allclose(a["explained_variance"], spectrum[:2], atol=1e-11)
    np.testing.assert_allclose(a["scores"], scores, atol=1e-11)
    np.testing.assert_array_equal(a["labels"], record["labels"])
    k = len(set(a["labels"]) - {-1})
    require(k == summary["selected_k"] == record["selected_k"], "Selected K")
    if arm != "gmm":
        require(np.isfinite(a["strength"]).all(), "Density strengths finite")
        require(np.all((a["strength"] >= 0) & (a["strength"] <= 1)), "Density strength range")
        require(np.all(a["strength"][a["labels"] == -1] == 0), "Noise strength")
        require(all(np.sum(a["labels"] == j) >= 5 for j in set(a["labels"]) - {-1}), "Cluster size")
    else:
        joint = []
        mixture = arrays(directory / "mixture.npz")
        for w, mu, cov in zip(
            mixture["weights"], mixture["means"], mixture["covariances"], strict=True
        ):
            require(np.linalg.eigvalsh(cov).min() > 0, "GMM covariance")
            delta = scores - mu
            joint.append(
                np.log(w)
                - 0.5
                * (
                    2 * np.log(2 * np.pi)
                    + np.linalg.slogdet(cov)[1]
                    + np.einsum("ij,jk,ik->i", delta, np.linalg.inv(cov), delta)
                )
            )
        joint = np.column_stack(joint)
        density = logsumexp(joint, axis=1)
        membership = np.exp(joint - density[:, None])
        np.testing.assert_allclose(mixture["membership"], membership, atol=1e-10)
        np.testing.assert_array_equal(membership.argmax(axis=1), mixture["labels"])
        candidates = read_json(directory / "candidates.json")
        winner = min(
            (c for c in candidates if c["status"] == "eligible"), key=lambda c: (c["bic"], c["k"])
        )
        require(winner["k"] == k, "GMM BIC selection")
        np.testing.assert_allclose(
            winner["bic"], (6 * k - 1) * np.log(len(x)) - 2 * density.sum(), atol=1e-7
        )
    with (directory / "assignments.csv").open(newline="") as f:
        assignments = list(csv.DictReader(f))
    require(len(assignments) == len(x), "Assignment row count")
    key = "cluster" if arm != "gmm" else "candidate_cluster"
    exported = [
        int(r[key]) if arm != "gmm" else int(r[key].split("_")[-1]) - 1 for r in assignments
    ]
    np.testing.assert_array_equal(exported, a["labels"])
    return a


def audit_phase(root, phase):
    import numpy as np

    from super_deli.benchmarking.density_audit import _audit_support
    from super_deli.benchmarking.neural_study import ARMS, SETTINGS, assert_frozen, tasks

    assert_frozen(root)
    require(__debug__, "Audits must run without Python optimization")
    total, failed, replays = 0, 0, 0
    for task in tasks(phase):
        directory = root / phase / task["key"]
        require(read_json(directory / "task.json") == task, "Task freeze")
        checksums = read_json(directory / "checksums.json")
        actual = {
            str(p.relative_to(directory))
            for p in directory.rglob("*")
            if p.is_file() and p.name != "checksums.json"
        }
        require(actual == set(checksums), "Complete observation artifact accounting")
        for name, digest in checksums.items():
            require(
                fingerprint_file(directory / name).sha256 == digest, "Observation artifact hash"
            )
        x, z, truth = reference(task)
        saved = arrays(directory / "generating-reference.npz")
        np.testing.assert_allclose(saved["x"], x, atol=1e-14)
        np.testing.assert_array_equal(saved["z"], z)
        np.testing.assert_array_equal(saved["truth"], [] if truth is None else truth)
        ids, source = read_raw(directory / "raw")
        np.testing.assert_array_equal(source, saved["x"])
        require(ids == [f"invented_{i:04d}" for i in range(len(x))], "Source identities")
        children = np.random.SeedSequence(task["subsampling_seed"]).spawn(task["draws"])
        draws = [
            np.sort(np.random.default_rng(s).choice(len(x), int(0.8 * len(x)), replace=False))
            for s in children
        ]
        records = read_json(directory / "fits.json")
        require(
            len(records) == 5 * (1 + len(draws) + int(task["replay"])), "Observation attempt count"
        )
        keys = [(r["kind"], r.get("draw"), r["arm"]) for r in records]
        expected_keys = (
            [("point", None, a) for a in ARMS]
            + [("draw", i, a) for i in range(len(draws)) for a in ARMS]
            + ([("replay", None, a) for a in ARMS] if task["replay"] else [])
        )
        require(keys == expected_keys, "Exact frozen workflow grid")
        points = {}
        for r in records:
            total += 1
            indices = draws[r["draw"]] if r["kind"] == "draw" else np.arange(len(x))
            np.testing.assert_array_equal(r["training_indices"], indices)
            parent = (
                directory / "draws" / f"draw_{r['draw']:03d}"
                if r["kind"] == "draw"
                else directory / r["kind"]
            )
            raw_dir = parent / "raw" if r["kind"] == "draw" else directory / "raw"
            sub_ids, sub_raw = read_raw(raw_dir)
            require(sub_ids == [ids[i] for i in indices], "Subset identity order")
            np.testing.assert_allclose(sub_raw, x[indices], atol=1e-14)
            fit_dir = parent / r["arm"]
            if r["kind"] == "replay":
                original = next(p for p in records if p["kind"] == "point" and p["arm"] == r["arm"])
                require(r["status"] == original["status"], "Replay status differs")
            if r["status"] == "failed":
                failed += 1
                require(
                    not (fit_dir / "summary.json").exists() and bool(r["error"]), "Retained failure"
                )
                continue
            fitted = audit_point(sub_raw, fit_dir, r["arm"], r)
            s = read_json(fit_dir / "summary.json")
            require(s["unit_ids"] == sub_ids, "Fit identity alignment")
            if r["kind"] == "point":
                points[r["arm"]] = fitted
            elif r["kind"] == "replay":
                require(r["arm"] in points, "Replay without successful original")
                for key, value in fitted.items():
                    np.testing.assert_array_equal(value, points[r["arm"]][key])
                replays += 1
        for arm in ARMS:
            records_for_arm = [r for r in records if r["kind"] == "draw" and r["arm"] == arm]
            support = arrays(directory / "support" / f"{arm}.npz")
            if records_for_arm:
                _audit_support(len(x), records_for_arm, support)
            else:
                require(not support, "No invented support without draws")
            summary = read_json(directory / "support" / f"{arm}.json")
            require(summary["requested"] == len(draws), "Support denominator")
            require(
                summary["failed"] == sum(r["status"] == "failed" for r in records_for_arm),
                "Support failures",
            )
    require(total == SETTINGS[f"{phase}_fits"], "Frozen total workflow count")
    result = {
        "passed": True,
        "phase": phase,
        "observations": len(tasks(phase)),
        "attempted_workflows": total,
        "failed_workflows": failed,
        "exact_replays": replays,
        "checks": [
            "independent generating law",
            "source/identity/hash accounting",
            "normalization and PCA",
            "independent NumPy neural forward/final MSE",
            "density invariants and exports",
            "GMM likelihood/membership/BIC",
            "exact saved-fit replay",
            "independent subsampling denominators",
        ],
        "limit": "HDBSCAN and optimizer are not independently reimplemented",
    }
    write_json(result, root / f"{phase}-audit.json")
    print(json.dumps(result), flush=True)
    return result


def ari(reference_labels, labels):
    """Compute adjusted Rand agreement directly from contingency pair counts."""
    from collections import Counter
    from math import comb

    pairs = sum(comb(n, 2) for n in Counter(zip(reference_labels, labels, strict=True)).values())
    a = sum(comb(n, 2) for n in Counter(reference_labels).values())
    b = sum(comb(n, 2) for n in Counter(labels).values())
    total = comb(len(labels), 2)
    expected = a * b / total if total else 0
    denom = (a + b) / 2 - expected
    return (pairs - expected) / denom if denom else 1.0


def measurements(record, truth):
    import numpy as np

    if record["status"] != "complete":
        return {"status": "failed", "error": record["error"]}
    labels = np.asarray(record["labels"])
    resolved = labels >= 0
    result = {
        "status": "complete",
        "selected_k": record["selected_k"],
        "coverage": float(np.mean(resolved)),
        "seconds": record["seconds"],
    }
    if truth is not None:
        predicted = labels.copy()
        predicted[~resolved] = np.arange(np.sum(~resolved)) + max(int(labels.max()), 0) + 1
        result.update(
            correct_k=record["selected_k"] == len(set(truth)),
            ari=ari(truth.tolist(), predicted.tolist()),
            assigned_ari=ari(truth[resolved].tolist(), labels[resolved].tolist())
            if np.sum(resolved) >= 2
            else None,
            split_generating_groups=sum(
                len(set(labels[(truth == k) & resolved])) > 1 for k in set(truth)
            ),
            merged_predicted_groups=sum(
                len(set(truth[labels == k])) > 1 for k in set(labels[resolved])
            ),
        )
    else:
        result.update(correct_k=None, ari=None, assigned_ari=None)
    return result


def interval(c, n, alpha=0.05):
    from scipy.stats import beta

    if not n:
        return [0.0, 1.0]
    return [
        float(beta.ppf(alpha / 2, c, n - c + 1)) if c else 0.0,
        float(beta.ppf(1 - alpha / 2, c + 1, n - c)) if c < n else 1.0,
    ]


def report_study(root):
    import numpy as np

    from super_deli.benchmarking.neural_study import ARMS, CASES, tasks

    require(read_json(root / "main-audit.json")["passed"], "Main audit required")
    rows, groups, initializations, stability = [], [], [], []
    for task in tasks("main"):
        directory = root / "main" / task["key"]
        reference_arrays = arrays(directory / "generating-reference.npz")
        truth = reference_arrays["truth"] if len(reference_arrays["truth"]) else None
        records = read_json(directory / "fits.json")
        point = {r["arm"]: r for r in records if r["kind"] == "point"}
        for arm in ARMS:
            row = {
                "case": task["case"],
                "n": task["n"],
                "replicate": task["replicate"],
                "arm": arm,
                **measurements(point[arm], truth),
            }
            if arm.startswith("neural_") and row["status"] == "complete":
                diag = read_json(directory / "point" / arm / "training.json")
                row.update(initial_loss=diag["initial_loss"], final_loss=diag["final_loss"])
            rows.append(row)
            if task["draws"]:
                s = read_json(directory / "support" / f"{arm}.json")
                a = arrays(directory / "support" / f"{arm}.npz")
                upper = np.triu_indices(task["n"], 1)
                support_row = {
                    "key": task["key"],
                    "arm": arm,
                    **s,
                    "median_pair_inclusions": float(np.median(a["pair_included"][upper])),
                    "mean_pair_coassignment": float(np.nanmean(a["coassignment"][upper])),
                    "coassignment_scope": (
                        "included-unit density agreement, not point-cluster recovery"
                    ),
                }
                if point[arm]["status"] == "complete":
                    labels = np.asarray(point[arm]["labels"])
                    pair = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
                    mask = np.triu(pair, 1)
                    denominator = int(a["pair_successful"][mask].sum())
                    support_row.update(
                        point_k=point[arm]["selected_k"],
                        within_point_pair_successful_inclusions=denominator,
                        within_point_coassignment=float(a["pair_coassigned"][mask].sum())
                        / denominator
                        if denominator
                        else None,
                    )
                stability.append(support_row)
        for i, left in enumerate(ARMS[2:]):
            for right in ARMS[3 + i :]:
                a, b = point[left], point[right]
                if a["status"] == b["status"] == "complete":
                    la, lb = np.array(a["labels"]), np.array(b["labels"])
                    jointly = (la >= 0) & (lb >= 0)
                    initializations.append(
                        {
                            "key": task["key"],
                            "left": left,
                            "right": right,
                            "jointly_assigned": int(np.sum(jointly)),
                            "n": task["n"],
                            "status": "complete",
                            "left_k": a["selected_k"],
                            "right_k": b["selected_k"],
                            "assigned_ari": ari(la[jointly].tolist(), lb[jointly].tolist())
                            if np.sum(jointly) >= 2
                            else None,
                        }
                    )
                else:
                    initializations.append(
                        {
                            "key": task["key"],
                            "left": left,
                            "right": right,
                            "status": "unavailable",
                            "n": task["n"],
                        }
                    )
    for case in CASES:
        for n in (64, 256):
            for arm in ARMS:
                cell = [r for r in rows if r["case"] == case and r["n"] == n and r["arm"] == arm]
                done = [r for r in cell if r["status"] == "complete"]
                c = sum(r["correct_k"] is True for r in done)
                failed = len(cell) - len(done)
                primary = case != "cline" and arm in ARMS[:3]
                group = {
                    "case": case,
                    "n": n,
                    "arm": arm,
                    "requested": len(cell),
                    "successful": len(done),
                    "failed": failed,
                    "selected_more_than_one": sum(r["selected_k"] > 1 for r in done),
                    "selected_k_counts": {
                        str(k): sum(r["selected_k"] == k for r in done)
                        for k in sorted({r["selected_k"] for r in done})
                    },
                    "primary": primary,
                    "correct_k_count": c if case != "cline" else None,
                    "correct_k_rate_resolved": c / len(done) if done and case != "cline" else None,
                    "failure_bounds": [c / len(cell), (c + failed) / len(cell)]
                    if case != "cline"
                    else None,
                }
                if primary:
                    group.update(
                        interval95_resolved=interval(c, len(done)),
                        simultaneous95_resolved=interval(c, len(done), 0.05 / 18),
                        simultaneous95_failure_envelope=[
                            interval(c, len(cell), 0.05 / 18)[0],
                            interval(c + failed, len(cell), 0.05 / 18)[1],
                        ],
                    )
                for name in (
                    "coverage",
                    "ari",
                    "assigned_ari",
                    "seconds",
                    "final_loss",
                    "split_generating_groups",
                    "merged_predicted_groups",
                ):
                    vals = [r[name] for r in done if r.get(name) is not None]
                    group[f"median_{name}"] = float(np.median(vals)) if vals else None
                groups.append(group)
    require(sum(g["primary"] for g in groups) == 18, "Primary denominator accounting")
    summary = {
        "schema_version": 1,
        "observations": 80,
        "main_workflows": 1210,
        "preflight_workflows": 120,
        "primary_rates": 18,
        "groups": groups,
        "failed_point_workflows": sum(r["status"] != "complete" for r in rows),
        "initialization_agreement": initializations,
        "subsampling": stability,
        "interpretation": [
            "Ten independent datasets per cell; descriptive pilot validation",
            "seeds/draws are repeated measurements, not extra datasets",
            "fixed transductive architecture; no best-seed or best-model selection",
            "training reconstruction is not held-out prediction",
            "no new p-values or species-boundary claim",
        ],
    }
    write_json(rows, root / "point-assessments.json")
    write_json(summary, root / "aggregate.json")
    print(
        json.dumps(
            {
                "observations": 80,
                "primary_rates": 18,
                "groups": len(groups),
                "failed_points": summary["failed_point_workflows"],
            }
        ),
        flush=True,
    )
    return summary
