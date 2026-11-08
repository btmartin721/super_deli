"""Independent raw-Gaussian, actual optimizer, candidate-density and tail audits."""

import json
import time
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from scipy.stats import beta

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.coordinate_audit import audit_mixture, gaussian_reference
from super_deli.benchmarking.matched_steps import digest_arrays
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_calibration_config import ARMS
from super_deli.benchmarking.neural_calibration_data import reference
from super_deli.benchmarking.neural_calibration_run import replay_equal
from super_deli.benchmarking.separate_audit import audit_pca
from super_deli.benchmarking.separate_gmm_observer import evaluation_path
from super_deli.benchmarking.view_weight_audit import audit_trajectory
from super_deli.io.config import read_json
from super_deli.models.separate_neural import unpack_branch


def zip_arrays(z, name):
    with np.load(BytesIO(z.read(name)), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def audit_layout(archive, prefix, layout, values):
    """Check original per-view row order, IDs, traits, units and raw selected cells."""
    positions = {unit: i for i, unit in enumerate(layout["unit_ids"])}
    offset = 0
    for index, view in enumerate(layout["views"]):
        table = json.loads(archive.read(prefix + f"ingestion/modality_{index:03d}/table.json"))
        require(table["observation_ids"] == view["observation_ids"], "Observation row order")
        require(table["unit_ids"] == view["unit_ids"], "Observation-to-specimen links")
        features = {f["name"]: (i, f) for i, f in enumerate(table["features"])}
        names = view["feature_names"]
        indices = [features[name][0] for name in names]
        require(
            [features[name][1]["unit"] for name in names] == view["feature_units"], "Trait units"
        )
        actual = np.asarray(table["values"], dtype=float)[:, indices]
        expected = values[[positions[u] for u in view["unit_ids"]], offset : offset + len(names)]
        np.testing.assert_array_equal(actual, expected)
        offset += len(names)


def audit_head(scores, meta, a):
    if meta["status"] == "available":
        return audit_mixture(scores, {**meta, "clusterer": "gmm_shared"}, a)[0]
    count = 0
    require(meta["status"] == "failed", "Retained failed search")
    for r in meta["candidate_attempts"]:
        if r["status"] == "complete":
            p, ll, bic, _ = gaussian_reference(scores, a, r["k"], "tied")
            np.testing.assert_allclose(p, a[f"k{r['k']}_membership"], atol=1e-8)
            np.testing.assert_allclose(
                [ll, bic], [r["log_likelihood"], r["bic"]], atol=1e-5, rtol=0
            )
            count += 1
    return count


def audit_point(
    root, arm, index, x, a, meta, *, events, evidence, used, config, cache, replay=False
):
    scores = []
    verified = 0
    for role, sl in (("morphology", slice(0, 4)), ("ecology", slice(4, 12))):
        branch = unpack_branch(a, role)
        raw = x[:, sl]
        if "scores" not in branch:
            continue
        if arm == "pca4":
            audit_pca(raw, branch)
        else:
            kept = branch["kept_indices"]
            z = (raw[:, kept] - branch["means"][kept]) / branch["scales"][kept]
            matches = [
                e
                for e in events
                if e["fit"] not in used
                and e["status"] == "complete"
                and e["input_sha256"] == digest_arrays({"input": z})
            ]
            require(bool(matches), "Actual branch input linked to raw data")
            event = matches[0]
            used.add(event["fit"])
            audit_trajectory(
                raw, branch, config["model"], event, evidence, event["fit"], cache, paired=False
            )
            verified += 500
        scores.append(branch["scores"])
    if len(scores) != 2:
        require(meta["status"] != "complete", "Complete point has both views")
        return verified, 0
    score = np.column_stack(scores)
    if "scores" in a:
        np.testing.assert_array_equal(score, a["scores"])
    path = (
        evaluation_path(root, index, "gmm_shared")
        if arm == "neural"
        else root / ("head-evidence" if replay else "calibration/head-evidence") / str(index)
    )
    if not (path / "execution.json").exists():
        require(meta["status"] != "complete", "A complete point has captured candidates")
        return verified, 0
    row = read_json(path / "execution.json")
    head = arrays(path / "fit.npz")
    checked = audit_head(score, row, head)
    if meta["status"] == "complete":
        np.testing.assert_array_equal(a["labels"], head["labels"])
        require(meta["selected_k"] == row["selected_k"], "Selected K bound to head")
        candidates = [c for c in row["candidates"] if c["status"] == "eligible"]
        baseline = next(c["bic"] for c in candidates if c["k"] == 1)
        statistic = max(0, baseline - min(c["bic"] for c in candidates))
        if "statistic" in meta:
            np.testing.assert_allclose(meta["statistic"], statistic, atol=1e-10)
    return verified, checked


def audit_tail(summary, records):
    require(summary["attempted_fits"] == 1 + len(records), "Observed/draw denominator")
    inf = summary["inference"]
    point = summary["point"]
    if point["status"] != "complete":
        require(not records and inf["p_value"] is None, "Failed observed fit withholds inference")
        return
    t = point["statistic"]
    b = sum(r["status"] == "complete" and r["statistic"] >= t for r in records)
    failed = sum(r["status"] == "failed" for r in records)
    total = summary["configuration_n_simulations"]
    require(
        len(records) == total and inf["exceedances"] == b and inf["failed"] == failed,
        "Exact tail accounting",
    )
    if failed:
        require(
            inf["p_value"] is None and inf["reject_joint_raw_gaussian"] is None,
            "Incomplete p-value withheld",
        )
    else:
        p = (b + 1) / (total + 1)
        require(
            inf["p_value"] == p and inf["reject_joint_raw_gaussian"] == (p <= inf["alpha"]),
            "Plus-one upper tail",
        )
        expected = [
            0 if b == 0 else float(beta.ppf(0.025, b, total - b + 1)),
            1 if b == total else float(beta.ppf(0.975, b + 1, total - b)),
        ]
        np.testing.assert_allclose(inf["conditional_null_tail_interval_95"], expected, atol=1e-14)
    np.testing.assert_array_equal(
        inf["failed_outcome_bounds"], [(b + 1) / (total + 1), (b + failed + 1) / (total + 1)]
    )


def audit_arm(folder, task, arm, *, cache, replay=False):
    started = time.monotonic()
    root = folder / arm
    row = read_json(root / "execution.json")
    require(not row["numerical_guard_failed"], "No unresolved optimizer guard")
    events = read_json(root / "optimizer-checks.json") if arm == "neural" else []
    evidence = arrays(root / "optimizer-checks.npz") if arm == "neural" else {}
    used = set()
    updates = 0
    densities = 0
    inputs = {}
    if replay:
        x = arrays(root / "run/input.npz")["values"]
        a = arrays(root / "run/point.npz")
        meta = read_json(root / "run/summary.json")
        meta["status"] = "complete"
        updates, densities = audit_point(
            root,
            arm,
            0,
            x,
            a,
            meta,
            events=events,
            evidence=evidence,
            used=used,
            config=row["configuration"],
            cache=cache,
            replay=True,
        )
    else:
        cal = root / "calibration"
        if not (cal / "null_model.npz").exists():
            require(
                row["status"] == "failed" and (cal / "failure.json").exists(),
                "Retained invalid raw null",
            )
        else:
            null = arrays(cal / "null_model.npz")
            layout = read_json(cal / "layout.json")
            x = null["values"]
            n, p = x.shape
            d = arrays(folder / "generating-reference.npz")
            expected = np.column_stack((d["morphology"], d["ecology"]))
            np.testing.assert_array_equal(x, expected)
            mean = x.mean(0)
            sigma = (x - mean).T @ (x - mean) / n
            np.testing.assert_allclose(null["mean"], mean, atol=1e-13)
            np.testing.assert_allclose(null["covariance"], sigma, atol=1e-13)
            sign, logdet = np.linalg.slogdet(sigma)
            require(sign == 1, "Positive-definite raw MLE")
            q = np.einsum("ij,ji->", x - mean, np.linalg.solve(sigma, (x - mean).T))
            np.testing.assert_allclose(q, n * p, rtol=1e-10, atol=1e-8)
            require(
                np.isfinite(-0.5 * (n * p * np.log(2 * np.pi) + n * logdet + q)),
                "Finite independent raw likelihood",
            )
            summary = read_json(cal / "summary.json") if (cal / "summary.json").exists() else None
            records = (
                read_json(cal / "null_replicates.json")
                if summary
                else [read_json(v) for v in sorted(cal.glob("draw_*.json"))]
            )
            point = read_json(cal / "observed.json") if (cal / "observed.json").exists() else None
            index = 0
            with ZipFile(cal / "pipeline_artifacts.zip") as z:
                inventory = read_json(cal / "archive_inventory.json") if summary else None
                if inventory:
                    from hashlib import sha256

                    require(
                        set(z.namelist()) == set(inventory), "Complete pipeline archive inventory"
                    )
                    for name, item in inventory.items():
                        b = z.read(name)
                        require(
                            len(b) == item["size_bytes"]
                            and sha256(b).hexdigest() == item["sha256"],
                            "Archived bytes",
                        )
                for i, r in enumerate(([point] if point else []) + records):
                    prefix = r["archive_prefix"] + "/run/"
                    if i:
                        seed = derived_seed(
                            row["configuration"]["random_seed"],
                            "separate_raw_gaussian_v1",
                            i - 1,
                            "null_data",
                        )
                        require(
                            r["simulation_seed"] == seed and r["replicate"] == i - 1,
                            "Fresh independent draw stream",
                        )
                        expected = (
                            mean
                            + np.random.default_rng(seed).standard_normal((n, p))
                            @ np.linalg.cholesky(sigma).T
                        )
                    else:
                        expected = x
                    if prefix + "input.npz" not in z.namelist():
                        require(
                            r["status"] == "failed", "Missing point inputs imply failed ingestion"
                        )
                        continue
                    actual = zip_arrays(z, prefix + "input.npz")["values"]
                    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-12)
                    alignment = json.loads(z.read(prefix + "alignment.json"))
                    require(
                        alignment["unit_ids"] == [f"unit_{j:04d}" for j in range(n)],
                        "All matched identities preserved",
                    )
                    inputs[r["archive_prefix"]] = actual
                    audit_layout(z, prefix, layout, actual)
                    file = "point.npz" if r["status"] == "complete" else "failed_point.npz"
                    a = zip_arrays(z, prefix + file) if prefix + file in z.namelist() else {}
                    u, c = audit_point(
                        root,
                        arm,
                        index,
                        actual,
                        a,
                        r,
                        events=events,
                        evidence=evidence,
                        used=used,
                        config=row["configuration"],
                        cache=cache,
                    )
                    updates += u
                    densities += c
                    index += 1
            if summary:
                audit_tail(
                    {**summary, "configuration_n_simulations": task["n_simulations"]}, records
                )
    for e in events:
        if e["fit"] in used:
            continue
        errors = evidence[f"fit_{e['fit']}_errors"]
        require(errors.ndim == 2 and len(errors) <= 500, "Retained partial training")
        if e["status"] == "complete":
            # A completed training can precede a failed latent transform. Its live
            # update checks remain evidence; no valid projection is claimed.
            require(len(errors) == 500, "All completed updates retained")
            possible = set()
            for raw in inputs.values():
                for sl in (slice(0, 4), slice(4, 12)):
                    view = raw[:, sl]
                    scale = view.std(0)
                    kept = scale > 0
                    z = (view[:, kept] - view.mean(0)[kept]) / scale[kept]
                    possible.add(digest_arrays({"input": z}))
            require(e["input_sha256"] in possible, "Failed projection training linked to raw input")
            normalized = evidence[f"fit_{e['fit']}_normalized_errors"]
            require(np.isfinite(normalized).all() and np.all(normalized <= 1), "Live update checks")
        updates += len(errors)
    return dict(
        key=task["key"],
        arm=arm,
        draws=0,
        neural_fits=len(events),
        verified_updates=updates,
        candidate_densities_checked=densities,
        seconds=time.monotonic() - started,
        paired_raw_inputs=inputs,
    )


def audit_dataset(folder, task, *, cache):
    data = arrays(folder / "generating-reference.npz")
    expected = reference(task)
    require(set(data) == set(expected), "Complete independent generator reference")
    for key, value in expected.items():
        np.testing.assert_allclose(data[key], value, atol=1e-12, rtol=1e-12)
    records = [audit_arm(folder, task, a, cache=cache) for a in ARMS]
    left, right = (r.pop("paired_raw_inputs") for r in records)
    # An arm with a failed observed fit does not attempt null fits. Compare only
    # common attempted inputs; each arm's complete failure denominator is audited.
    for key in left.keys() & right.keys():
        np.testing.assert_array_equal(left[key], right[key])
    return records


def audit_replay(folder, source, entry, cache):
    replay_equal(folder, source, entry)
    rows = [audit_arm(folder, {"key": entry["key"]}, a, cache=cache, replay=True) for a in ARMS]
    for r in rows:
        r.pop("paired_raw_inputs")
    return rows
