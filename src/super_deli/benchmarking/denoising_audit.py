"""Fit-free independent generation, training, identity, replay and support audits."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

from super_deli.benchmarking.denoising_numerics import audit_training
from super_deli.benchmarking.denoising_study import ARMS, CONDITIONS, SETTINGS, assert_frozen, tasks
from super_deli.benchmarking.density_audit import _audit_support
from super_deli.benchmarking.neural_analysis import (
    arrays,
    audit_point,
    read_raw,
    reference,
    require,
)
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json


def audit_inventory(directory):
    inventory = read_json(directory / "checksums.json")
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }
    require(actual == set(inventory), "Complete evidence inventory")
    for name, digest in inventory.items():
        require(fingerprint_file(directory / name).sha256 == digest, "Evidence hash mismatch")


def audit_density(a, record):
    from scipy.spatial.distance import cdist

    labels = a["labels"]
    require(labels.dtype.kind in "iu", "Integer density labels")
    require(labels.shape == (len(a["scores"]),), "Label shape")
    np.testing.assert_array_equal(labels, record["labels"])
    require(len(set(labels) - {-1}) == record["selected_k"], "Selected K")
    require(np.all(labels >= -1), "Supported noise label")
    strength = a["strength"]
    require(
        np.isfinite(strength).all() and np.all((strength >= 0) & (strength <= 1)),
        "Density strengths",
    )
    require(np.all(strength[labels < 0] == 0), "Noise strength zero")
    require(
        all(np.sum(labels == k) >= 5 for k in set(labels) - {-1}), "Minimum density cluster size"
    )
    scores = a["scores"]
    independent = np.sqrt(np.maximum(0, ((scores[:, None, :] - scores[None, :, :]) ** 2).sum(2)))
    np.testing.assert_allclose(cdist(scores, scores), independent, atol=1e-12)


def audit_legacy(root, phase, lock):
    archive = Path(lock["archive_dir"])
    protected = Path(__file__).resolve().parents[1]
    old = read_json(archive / "frozen.json")
    with tarfile.open(archive / "frozen-sources.tar.gz", "r:gz") as bundle:
        for name, digest in old["source_sha256"].items():
            if name.startswith("models/"):
                require(
                    fingerprint_file(protected / name).sha256 == digest,
                    "Legacy numerical source changed",
                )
            member = bundle.extractfile(name)
            require(member is not None, "Archived source missing")
            import hashlib

            require(
                hashlib.sha256(member.read()).hexdigest() == digest, "Legacy source bundle hash"
            )
    records = read_json(root / phase / "legacy-replays" / "fits.json")
    require(
        [(r["key"], r["arm"]) for r in records]
        == [(k, a) for k in ("gaussian_n64_0000", "moons_n256_0000") for a in ARMS[:4]],
        "Legacy replay grid",
    )
    failures = 0
    for r in records:
        source = archive / "main" / r["key"]
        audit_inventory(source)
        if r["status"] != "complete":
            failures += 1
            require(bool(r["error"]), "Retained legacy failure")
            continue
        current = arrays(root / phase / r["path"] / "point.npz")
        original = arrays(source / "point" / r["arm"] / "point.npz")
        require(set(current) == set(original), "Legacy array inventory")
        for name, value in original.items():
            np.testing.assert_array_equal(current[name], value)
        ids, x = read_raw(source / "raw")
        audit_point(x, root / phase / r["path"], r["arm"], r)
        require(
            read_json(root / phase / r["path"] / "summary.json")["unit_ids"] == ids,
            "Legacy identities",
        )
    return failures


def audit_phase(root, phase):
    lock = assert_frozen(root)
    require(__debug__, "Audits require Python without optimization")
    total = failed = replays = neural = corruptions = 0
    for task in tasks(phase):
        out = root / phase / task["key"]
        require(read_json(out / "task.json") == task, "Task freeze")
        audit_inventory(out)
        noisy, z, truth = reference(task)
        noise = np.random.default_rng(task["noise_seed"]).normal(0, 0.15, noisy.shape)
        clean = noisy - noise
        source = arrays(out / "generating-reference.npz")
        np.testing.assert_array_equal(source["z"], z)
        np.testing.assert_array_equal(source["truth"], truth)
        np.testing.assert_array_equal(source["noise"], noise)
        np.testing.assert_allclose(source["clean"], clean, rtol=1e-13, atol=1e-13)
        draws = [
            np.sort(np.random.default_rng(s).choice(len(z), int(0.8 * len(z)), replace=False))
            for s in np.random.SeedSequence(task["subsampling_seed"]).spawn(task["draws"])
        ]
        records = read_json(out / "fits.json")
        expected = []
        for condition in CONDITIONS:
            for kind, draw in [
                ("point", None),
                *[("draw", i) for i in range(len(draws))],
                *([("replay", None)] if task["replay"] else []),
            ]:
                expected.extend((condition, kind, draw, arm) for arm in ARMS)
        expected.extend(
            ("source", kind, None, "source_reference")
            for kind in (("point", "replay") if task["replay"] else ("point",))
        )
        require(
            [(r["condition"], r["kind"], r["draw"], r["arm"]) for r in records] == expected,
            "Frozen fit grid",
        )
        points = {}
        for r in records:
            total += 1
            if r["arm"] != "density" and r["arm"] != "source_reference":
                neural += 1
            ix = draws[r["draw"]] if r["kind"] == "draw" else np.arange(len(z))
            np.testing.assert_array_equal(ix, r["training_indices"])
            directory = out / r["path"]
            key = (r["condition"], r["arm"])
            if r["status"] == "failed":
                failed += 1
                require(
                    bool(r["error"]) and not (directory / "summary.json").exists(),
                    "Retained failed outcome",
                )
                continue
            if r["arm"] == "source_reference":
                a = arrays(directory / "point.npz")
                np.testing.assert_allclose(a["scores"], (z - z.mean(0)) / z.std(0), atol=1e-13)
            else:
                x = source["clean"] if r["condition"] == "clean" else source["clean"] + noise
                raw = (
                    directory.parent / "raw"
                    if r["kind"] == "draw"
                    else out / r["condition"] / "raw"
                )
                ids, raw_values = read_raw(raw)
                require(
                    ids == [f"invented_{j:04d}" for j in ix], "Unit order and source identities"
                )
                np.testing.assert_array_equal(raw_values, x[ix])
                arm = (
                    r["arm"]
                    if not r["arm"].startswith("denoising_")
                    else r["arm"].replace("denoising_", "neural_")
                )
                a = audit_point(raw_values, directory, arm, r)
                summary = read_json(directory / "summary.json")
                require(summary["unit_ids"] == ids, "Fit/ingestion alignment")
                if r["arm"].startswith("denoising_"):
                    seed = int(r["arm"].split("_")[1])
                    expected_seed = int(
                        np.random.SeedSequence([*task["corruption_seed"], seed]).generate_state(1)[
                            0
                        ]
                    )
                    require(
                        r["corruption_seed"]
                        == summary["configuration"]["model"]["corruption_seed"]
                        == expected_seed,
                        "Frozen corruption seed",
                    )
                    require(
                        summary["model"] == "tabular_denoising_autoencoder_hdbscan",
                        "Denoising family",
                    )
                    require(
                        configuration_hash(summary["configuration"])
                        == summary["configuration_sha256"],
                        "Configuration hash",
                    )
                    audit_training((raw_values - a["means"]) / a["scales"], a, expected_seed)
                    corruptions += 500
            audit_density(a, r)
            if r["kind"] == "point":
                points[key] = a
            if r["kind"] == "replay":
                require(key in points, "Replay without successful point")
                require(set(a) == set(points[key]), "Replay inventory")
                for name, value in a.items():
                    np.testing.assert_array_equal(value, points[key][name])
                replays += 1
        for condition in CONDITIONS:
            for arm in ARMS:
                rows = [
                    r
                    for r in records
                    if r["condition"] == condition and r["arm"] == arm and r["kind"] == "draw"
                ]
                support = arrays(out / condition / f"{arm}-support.npz")
                if rows:
                    _audit_support(len(z), rows, support)
                else:
                    require(not support, "No invented stability")
                s = read_json(out / condition / f"{arm}-support.json")
                require(
                    s["requested"] == task["draws"]
                    and s["failed"] == sum(r["status"] == "failed" for r in rows),
                    "Stability denominators",
                )
        # Paired conditions share both initialization and corruption streams.
        for arm in ARMS[4:]:
            if ("clean", arm) in points and ("noisy", arm) in points:
                for name, value in points[("clean", arm)].items():
                    if name.startswith("checkpoint_0_") or name == "corruption_sha256":
                        np.testing.assert_array_equal(value, points[("noisy", arm)][name])
    failed += audit_legacy(root, phase, lock)
    total += 8
    neural += 6
    require(total == SETTINGS[f"{phase}_fits"], "Total frozen workflow count")
    require(neural == SETTINGS[f"{phase}_neural_fits"], "Total neural training count")
    result = {
        "passed": True,
        "phase": phase,
        "observations": len(tasks(phase)),
        "attempted_workflows": total,
        "neural_training_fits": neural,
        "failed_workflows": failed,
        "exact_fresh_replays": replays,
        "exact_legacy_replays": 8
        - (
            sum(
                r["status"] != "complete"
                for r in read_json(root / phase / "legacy-replays" / "fits.json")
            )
        ),
        "corruption_matrices_checked": corruptions,
        "checks": [
            "independent source laws and noise",
            "exact row/draw pairing and hashes",
            "independent forward and first Adam update",
            "all corruption digests",
            "PCA and density invariants",
            "legacy numeric source/weight parity",
            "independent subsampling denominators",
            "exports and frozen fit accounting",
        ],
        "density_audit_boundary": (
            "invariants and distances; HDBSCAN not independently reimplemented"
        ),
    }
    write_json(result, root / f"{phase}-audit.json")
    print(result, flush=True)
    return result
