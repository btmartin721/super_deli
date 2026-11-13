"""Independent raw-law, exclusion, dense-likelihood and three-reference audits."""

import csv
import time

import numpy as np
from scipy.stats import multivariate_normal

from super_deli.benchmarking.abstention_audit import (
    compare,
    decisions,
    prediction,
    project,
    reference,
)
from super_deli.benchmarking.known_reference_config import ARMS, POLICIES, roles
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.residual_audit import preprocessing
from super_deli.io.config import read_json
from super_deli.models.reference_abstention_config import DECISIONS


def independent_metrics(y, p, compatible, accepted, status, known):
    chosen = p.argmax(1)
    n = len(y)
    total = sum(bool(x) for x in accepted)
    result = dict(
        n=n,
        accepted=total,
        accepted_fraction=total / n,
        reasons={s: sum(v == s for v in status) for s in DECISIONS},
    )
    if not known:
        return result
    correct = [int(a) == int(b) for a, b in zip(y, chosen, strict=True)]
    good = sum(bool(a) and bool(b) for a, b in zip(correct, accepted, strict=True))
    result.update(
        accepted_accuracy=good / total if total else None,
        correct_accepted_fraction=good / n,
        incorrect_accepted_fraction=(total - good) / n,
        known_withheld_fraction=(n - total) / n,
        raw_accuracy=sum(correct) / n,
        per_class=[],
    )
    for c in range(p.shape[1]):
        members = np.flatnonzero(y == c)
        selected = [i for i in members if accepted[i]]
        result["per_class"].append(
            dict(
                reference=c,
                n=len(members),
                accepted=len(selected),
                accepted_accuracy=sum(correct[i] for i in selected) / len(selected)
                if selected
                else None,
                true_reference_included=sum(bool(compatible[i, c]) for i in members) / len(members)
                if compatible is not None
                else None,
            )
        )
    accuracies = [v["accepted_accuracy"] for v in result["per_class"]]
    result["accepted_balanced_accuracy"] = (
        sum(accuracies) / p.shape[1] if all(v is not None for v in accuracies) else None
    )
    if compatible is not None:
        result.update(
            true_reference_included=sum(bool(compatible[i, int(y[i])]) for i in range(n)) / n,
            mean_compatible_references=sum(sum(row) for row in compatible) / n,
        )
    return result


def quantiles(scores, y):
    counts = np.asarray([np.count_nonzero(y == c) for c in range(scores.shape[1])])
    require(np.all(counts >= 20), "Known-only calibration counts")
    ranks = np.asarray([(19 * (int(m) + 1) + 19) // 20 for m in counts])
    return (
        np.asarray([sorted(scores[y == c, c])[ranks[c] - 1] for c in range(len(counts))]),
        counts,
        ranks,
    )


def raw_matrix(folder, expected):
    ds = read_json(folder / "dataset.json")
    ids = [v["specimen_id"] for v in ds["specimens"]]
    require(ids == sorted(set(ids)), "Complete unique sorted raw identities")
    for m in ds["modalities"]:
        with (folder / m["path"]).open(newline="") as f:
            rows = list(csv.reader(f, delimiter="\t"))
        values = {r[0]: [float(v) for v in r[1:]] for r in rows[1:]}
        require(len(values) == len(ids), "One measurement per retained identity")
        np.testing.assert_array_equal(
            np.asarray([values[r["observation_id"]] for r in m["observations"]]),
            expected[m["name"]],
        )
        require([r["unit_id"] for r in m["observations"]] == ids, "Explicit raw joins")
    return ids


def quadratic(z, centers, covariances):
    return np.column_stack(
        [
            np.einsum("ni,ij,nj->n", z - center, np.linalg.inv(cov), z - center)
            for center, cov in zip(centers, covariances, strict=True)
        ]
    )


def neighbors(z, train, y, covariance):
    precision = np.linalg.inv(covariance)
    out = np.empty((len(z), 3))
    for c in range(3):
        library = train[y == c]
        for start in range(0, len(z), 256):
            delta = z[start : start + 256, None, :] - library[None, :, :]
            distances = np.einsum("nki,ij,nkj->nk", delta, precision, delta)
            out[start : start + 256, c] = np.sort(distances, axis=1)[:, 4]
    return out


def audit_fold(parent, h, folder):
    known = [c for c in range(4) if c != h]
    selected = {
        role: np.flatnonzero(parent[role]["truth"] != h) for role in ("train", "calibration")
    }
    x = {
        role: np.column_stack((parent[role]["morphology"], parent[role]["ecology"]))[idx]
        for role, idx in selected.items()
    }
    y = {
        role: np.asarray([known.index(int(v)) for v in parent[role]["truth"][idx]])
        for role, idx in selected.items()
    }
    manifest = read_json(folder / "exclusion-manifest.json")
    require(
        manifest["heldout"] == h and manifest["known_original_groups"] == known,
        "Held-out class map",
    )
    require(
        manifest["original_to_reference"] == {str(c): j for j, c in enumerate(known)},
        "Retained reference labels",
    )
    require(
        manifest["selected_training_rows"] == {k: v.tolist() for k, v in selected.items()},
        "Exact fitting exclusions",
    )
    ids = {}
    for role in ("train", "calibration"):
        raw_role = "training" if role == "train" else role
        data = {name: parent[role][name][selected[role]] for name in ("morphology", "ecology")}
        ids[role] = raw_matrix(folder / raw_role, data)
        with (folder / (raw_role + "-labels.csv")).open(newline="") as f:
            labels = list(csv.DictReader(f))
        require([r["unit_id"] for r in labels] == ids[role], "Known-only label identities")
        require(
            [r["reference_label"] for r in labels] == [f"reference_{v}" for v in y[role]],
            "Known-only labels",
        )
    require(not set(ids["train"]) & set(ids["calibration"]), "Independent role identities")
    pooled, local = arrays(folder / "pooled-support.npz"), arrays(folder / "local-support.npz")
    view_counts = np.bincount((pooled["kept_indices"] >= 4).astype(int), minlength=2)
    z = preprocessing(
        x["train"],
        {
            **pooled,
            "retained_view_counts": view_counts,
            "view_weights": np.sqrt(len(pooled["kept_indices"]) / (2 * view_counts)),
        },
    )
    zc = project(x["calibration"], pooled)
    centers = np.stack([z[y["train"] == c].mean(0) for c in range(3)])
    raw = np.stack(
        [
            (z[y["train"] == c] - centers[c]).T
            @ (z[y["train"] == c] - centers[c])
            / sum(y["train"] == c)
            for c in range(3)
        ]
    )
    shrunk = np.stack([0.9 * s + 0.1 * np.trace(s) / z.shape[1] * np.eye(z.shape[1]) for s in raw])
    for k, v in [("centers", centers)]:
        np.testing.assert_allclose(pooled[k], v, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pooled["unshrunk_covariance"], sum(raw) / 3, rtol=1e-12, atol=1e-12)
    density = (
        arrays(folder / "density-support.npz")
        if (folder / "density-support.npz").exists()
        else None
    )
    if density is not None:
        for k, v in [("centers", centers), ("unshrunk_covariances", raw), ("covariances", shrunk)]:
            np.testing.assert_allclose(density[k], v, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            density["cholesky"] @ np.transpose(density["cholesky"], (0, 2, 1)),
            shrunk,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            density["log_determinants"], np.linalg.slogdet(shrunk)[1], rtol=1e-12, atol=1e-12
        )
    else:
        require(read_json(folder / "density-failure.json")["error"], "Retained candidate failure")
    covariance = pooled["covariance"]
    score_cal = dict(
        distance=quadratic(zc, centers, np.repeat(covariance[None], 3, axis=0)),
        local=neighbors(zc, z, y["train"], covariance),
    )
    if density is not None:
        score_cal["density"] = quadratic(zc, centers, shrunk)
    thresholds = {}
    for policy, score in score_cal.items():
        q, counts, ranks = quantiles(score, y["calibration"])
        thresholds[policy] = q
        a = pooled if policy == "distance" else local if policy == "local" else density
        np.testing.assert_allclose(
            a["distance_thresholds" if policy == "distance" else "thresholds"],
            q,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_array_equal(a["calibration_counts"], counts)
        np.testing.assert_array_equal(a["calibration_ranks"], ranks)
    geometry = arrays(folder / "geometry.npz")
    rows = read_json(folder / "assessments.json")
    evaluations = {}
    for name in (
        "iid",
        "heldout",
        "near",
        "far",
        "overlap",
        *(["shifted"] if "shifted" in parent else []),
    ):
        source = "iid" if name == "heldout" else "overlap_pool" if name == "overlap" else name
        d = parent[source]
        truth = d["truth"]
        idx = (
            np.flatnonzero(truth == h)
            if name == "heldout"
            else np.flatnonzero(truth == known[0])
            if name == "overlap"
            else np.flatnonzero(truth != h)
            if name in ("iid", "shifted")
            else np.arange(len(truth))
        )
        is_known = name in ("iid", "shifted")
        yy = (
            np.asarray([known.index(int(c)) for c in truth[idx]])
            if is_known
            else np.full(len(idx), 3)
        )
        xx = np.column_stack((d["morphology"], d["ecology"]))[idx]
        zz = project(xx, pooled)
        require(
            manifest["evaluations"][name]
            == dict(parent_role=source, parent_rows=idx.tolist(), known=is_known),
            "Exact evaluation-only routing",
        )
        scores = dict(
            distance=quadratic(zz, centers, np.repeat(covariance[None], 3, axis=0)),
            local=neighbors(zz, z, y["train"], covariance),
        )
        if density is not None:
            scores["density"] = quadratic(zz, centers, shrunk)
            for c in range(3):
                nll = -multivariate_normal.logpdf(zz, mean=centers[c], cov=shrunk[c])
                reconstructed = 0.5 * (
                    scores["density"][:, c]
                    + np.linalg.slogdet(shrunk[c])[1]
                    + zz.shape[1] * np.log(2 * np.pi)
                )
                np.testing.assert_allclose(nll, reconstructed, rtol=1e-10, atol=1e-10)
        for policy, score in scores.items():
            np.testing.assert_allclose(geometry[name + "_" + policy], score, rtol=1e-10, atol=1e-10)
        evaluations[name] = dict(x=xx, y=yy, known=is_known, scores=scores)
    checked = updates = 0
    for row in rows:
        arm = row["arm"]
        require(arm in ARMS and row["heldout"] == h, "Declared fold/head")
        if row["status"] != "available":
            require(row["failed_points"] > 0 and row.get("error"), "Retained unavailable fit")
            require(
                all(v == "unavailable" for v in row["policy_status"].values()),
                "No inference after unavailable classifier",
            )
            completed = row.get("completed_optimizer_updates", 0)
            if arm != "logistic" and completed:
                path = folder / arm / "optimizer-checks.npz"
                if not path.exists():
                    path = folder / arm / "failed-optimizer-checks.npz"
                checks = arrays(path)
                require(
                    checks["errors"].shape == (completed, 4)
                    and np.isfinite(checks["errors"]).all(),
                    "Retained partial update audits",
                )
                require(
                    len(checks["step_sha256"]) == completed, "Retained partial step fingerprints"
                )
                updates += completed
            continue
        a = arrays(folder / arm / ("model.npz" if arm == "logistic" else "run/model.npz"))
        preprocessing(x["train"], a)
        if arm != "logistic":
            checks = arrays(folder / arm / "optimizer-checks.npz")
            require(
                checks["errors"].shape == (500, 4) and np.isfinite(checks["errors"]).all(),
                "All actual optimizer audits retained",
            )
            require(
                len(checks["step_sha256"]) == 500 and len(set(checks["step_sha256"])) == 500,
                "Every optimizer step fingerprint",
            )
            require(
                read_json(folder / arm / "optimizer-checks.json")["updates_checked"] == 500,
                "Actual update budget",
            )
            updates += 500
        else:
            cfg = read_json(folder / arm / "model-config.json")
            require(
                (cfg["C"], cfg["solver"], cfg["max_iter"], cfg["tol"]) == (1, "lbfgs", 2000, 1e-9),
                "Unchanged logistic settings",
            )
            require(np.all(a["iterations"] < 2000), "Converged logistic fit")
        fitted = arrays(folder / arm / "calibration.npz")
        saved = arrays(folder / arm / "predictions.npz")
        pc = prediction(x["calibration"], a, arm != "logistic")
        np.testing.assert_allclose(saved["calibration_probabilities"], pc, rtol=1e-10, atol=1e-10)
        thresholds["probability"] = quantiles(1 - pc, y["calibration"])[0]
        np.testing.assert_allclose(
            fitted["probability_thresholds"], thresholds["probability"], rtol=1e-10, atol=1e-10
        )
        for name, ev in evaluations.items():
            p = prediction(ev["x"], a, arm != "logistic")
            np.testing.assert_allclose(saved[name + "_probabilities"], p, rtol=1e-10, atol=1e-10)
            for policy in POLICIES:
                if row["policy_status"][policy] != "available":
                    require(
                        policy == "density" and density is None, "Withheld failed candidate only"
                    )
                    continue
                if policy == "unscreened":
                    compatible = None
                    accepted = np.ones(len(p), dtype=bool)
                    status = np.full(len(p), "assigned")
                else:
                    score = 1 - p if policy == "probability" else ev["scores"][policy]
                    compatible, accepted, status = decisions(p, score, thresholds[policy])
                    np.testing.assert_array_equal(
                        saved[name + "_" + policy + "_compatible"], compatible
                    )
                for k, value in [
                    ("accepted", accepted),
                    ("decisions", status),
                    ("accepted_indices", np.where(accepted, p.argmax(1), -1)),
                ]:
                    np.testing.assert_array_equal(saved[name + "_" + policy + "_" + k], value)
                compare(
                    row["metrics"][policy][name],
                    independent_metrics(ev["y"], p, compatible, accepted, status, ev["known"]),
                )
                checked += 1
        for policy, m in row["metrics"].items():
            expected = (
                m["iid"]["accepted_fraction"] >= 0.8
                and m["iid"]["accepted_accuracy"] is not None
                and m["iid"]["accepted_accuracy"] >= 0.9
                and all(m[c]["accepted_fraction"] <= 0.1 for c in ("heldout", "near", "far"))
            )
            require(
                row["joint_targets"][policy] == expected,
                "Prespecified three-reference joint target",
            )
    return dict(
        verified_updates=updates,
        policy_cohort_evaluations=checked,
        folds=1,
        neural_fits=sum(r["neural_fits"] for r in rows),
    )


def audit_unit(folder, job, source, *, cache):
    started = time.monotonic()
    task = job["task"]
    require(read_json(folder / "task.json") == task, "Frozen work-unit identity")
    original = task if job["kind"] == "fresh" else job["parent_task"]
    base = folder if job["kind"] == "fresh" else folder.parent / original["key"]
    parent = {}
    for role, _ in roles(original):
        actual = arrays(base / "parent" / (role + ".npz"))
        names = [v for v, _ in roles(original)]
        t = {**original, "seed_path": [*original["seed_path"], names.index(role)]}
        expected = reference(t, "iid" if role == "overlap_pool" else role)
        require(actual.keys() == expected.keys(), "Complete parent generative inventory")
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        parent[role] = actual
    heldouts = range(4) if job["kind"] == "fresh" else [job["heldout"]]
    results = [audit_fold(parent, h, folder / f"holdout-{h}") for h in heldouts]
    if job["kind"] != "fresh":
        from super_deli.benchmarking.known_reference_run import same_arrays

        count = same_arrays(
            base / f"holdout-{job['heldout']}", folder / f"holdout-{job['heldout']}"
        )
        require(
            read_json(folder / "replay-parity.json")["scientific_arrays"] == count,
            "Exact replay audit count",
        )
    return [
        dict(
            key=task["key"],
            seconds=time.monotonic() - started,
            neural_fits=sum(r["neural_fits"] for r in results),
            verified_updates=sum(r["verified_updates"] for r in results),
            policy_cohort_evaluations=sum(r["policy_cohort_evaluations"] for r in results),
            folds=len(heldouts),
        )
    ]
