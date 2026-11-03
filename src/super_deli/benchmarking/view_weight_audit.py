"""Independent raw-law, weighting, trajectory, density and denominator checks."""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from super_deli.benchmarking.density_audit import _audit_support
from super_deli.benchmarking.matched_audit import audit_point as audit_old_point
from super_deli.benchmarking.matched_audit import audit_sources, synthetic_tables
from super_deli.benchmarking.matched_reference import audit_latent
from super_deli.benchmarking.matched_steps import digest_arrays
from super_deli.benchmarking.moons_audit import _same_condensation, independent_condensation
from super_deli.benchmarking.moons_hierarchy import selection_trace, validate_linkage
from super_deli.benchmarking.neural_analysis import arrays, neural_forward, require
from super_deli.benchmarking.view_weight_config import ARMS, JOINT, model_for
from super_deli.benchmarking.view_weight_data import reference
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def audit_density(point, trace, index):
    scores, labels, strength = (point[k] for k in ("scores", "labels", "strength"))
    require(
        labels.shape == strength.shape == (len(scores),) and np.all(labels >= -1), "Density shapes"
    )
    require(
        np.isfinite(strength).all() and np.all((strength >= 0) & (strength <= 1)),
        "Density strength range",
    )
    require(np.all(strength[labels < 0] == 0), "Noise strength")
    require(all(sum(labels == k) >= 5 for k in set(labels) - {-1}), "Unchanged density size rule")
    distance = cdist(scores, scores)
    core = np.partition(distance, 4, axis=1)[:, 4]
    mutual = np.maximum(distance, np.maximum.outer(core, core))
    np.fill_diagonal(mutual, 0)
    mst = minimum_spanning_tree(mutual)
    linkage, condensed = trace[f"fit_{index}_linkage"], trace[f"fit_{index}_condensed"]
    members = validate_linkage(linkage, len(scores))
    np.testing.assert_allclose(np.sort(linkage["value"]), np.sort(mst.data), atol=1e-11)
    for row in linkage:
        left, right = [sorted(members[int(row[key])]) for key in ("left_node", "right_node")]
        np.testing.assert_allclose(mutual[np.ix_(left, right)].min(), row["value"], atol=1e-11)
    _same_condensation(independent_condensation(linkage, len(scores), 5), condensed, len(scores))
    _, assigned, supports = selection_trace(condensed, method="eom", allow_single_cluster=True)
    np.testing.assert_array_equal(labels, assigned)
    np.testing.assert_allclose(strength, supports, atol=1e-12)


def audit_trajectory(x, point, config, event, evidence, index, cache, *, paired):
    kept = point["kept_indices"]
    np.testing.assert_array_equal(kept, np.flatnonzero(x.std(0) > 0))
    np.testing.assert_allclose(point["means"], x.mean(0), atol=1e-12)
    scale = x.std(0)
    scale[scale == 0] = 1
    np.testing.assert_allclose(point["scales"], scale, atol=1e-12)
    z = (x[:, kept] - point["means"][kept]) / point["scales"][kept]
    if paired:
        from super_deli.models.matched_neural_config import MatchedNeuralConfig

        c = MatchedNeuralConfig.from_dict(config)
        split = len(c.morphology_features)
        count = np.array([sum(kept < split), sum(kept >= split)])
        fraction = c.loss_fractions()
        expected = (
            np.sqrt(len(kept) / (2 * count))
            if fraction == (0.5, 0.5)
            else np.sqrt(len(kept) * np.array(fraction) / count)
        )
        np.testing.assert_allclose(point["view_weights"], expected, atol=1e-15)
        np.testing.assert_allclose(
            point["column_weights"], expected[(kept >= split).astype(int)], atol=1e-15
        )
        np.testing.assert_array_equal(point["retained_view_counts"], count)
        if c.schema_version == 2:
            np.testing.assert_array_equal(point["declared_view_loss_weights"], c.view_loss_weights)
            np.testing.assert_array_equal(point["normalized_view_loss_weights"], fraction)
        z *= point["column_weights"]
    require(
        event["status"] == "complete" and event["updates_checked"] == 500,
        "Complete live optimizer checks",
    )
    require(
        event["seed"] == config["random_seed"] and event["n"] == len(x), "Fit seed/sample alignment"
    )
    initial = {
        k.removeprefix(f"fit_{index}_initial_"): v
        for k, v in evidence.items()
        if k.startswith(f"fit_{index}_initial_")
    }
    require(len(initial) == 12, "Complete initialization")
    key = (z.shape[1], event["seed"])
    if key not in cache:
        from super_deli.models.tabular_neural import _network, _torch, _weights

        torch = _torch()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(key[1])
            cache[key] = _weights(_network(key[0]))
    for name, value in initial.items():
        np.testing.assert_array_equal(value, cache[key][name])
    require(
        event["input_sha256"] == digest_arrays({"input": z}),
        "Actual weighted input bound to raw observations",
    )
    require(event["initial_sha256"] == digest_arrays(initial), "Recorded initialization hash")
    require(
        event["final_sha256"] == digest_arrays({k: point[k] for k in initial}),
        "Final weights bound to trajectory",
    )
    require(
        event["loss_sha256"] == digest_arrays({"losses": point["losses"]}),
        "All losses bound to trajectory",
    )
    errors = evidence[f"fit_{index}_errors"]
    normalized = evidence[f"fit_{index}_normalized_errors"]
    checks = evidence[f"fit_{index}_comparison_counts"]
    require(
        errors.shape == (500, 4) and np.isfinite(errors).all() and np.all(errors >= 0),
        "Every update retained",
    )
    require(
        normalized.shape == (500, 3)
        and np.isfinite(normalized).all()
        and np.all((normalized >= 0) & (normalized <= 1)),
        "Every update within unchanged tolerances",
    )
    np.testing.assert_array_equal(checks, np.tile([12, 24, 12], (500, 1)))
    hashes = evidence[f"fit_{index}_step_sha256"]
    require(
        hashes.shape == (500,) and all(len(h) == 64 for h in hashes),
        "All actual step hashes retained",
    )
    np.testing.assert_allclose(event["max_errors"], np.max(errors, axis=0), atol=0, rtol=0)
    reported, expected_losses = (
        evidence[f"fit_{index}_reported_losses"],
        evidence[f"fit_{index}_reference_losses"],
    )
    np.testing.assert_array_equal(point["losses"], reported)
    require(expected_losses.shape == (501,), "All independently evaluated losses")
    np.testing.assert_allclose(reported, expected_losses, rtol=1e-10, atol=1e-12)
    audit_latent(z, point)
    decoded = neural_forward(neural_forward(z, point, "encoder"), point, "decoder")
    np.testing.assert_allclose(np.mean((decoded - z) ** 2), point["losses"][-1], atol=1e-11)
    return z


def audit_arm(folder, task, data, row, cache):
    arm, paired = row["arm"], row["arm"] in JOINT
    require(
        not row.get("numerical_guard_failed"), "Unresolved numerical guard; main work must stop"
    )
    if row["status"] != "available":
        require(
            bool(row.get("error")) and (folder / "failure.json").exists(),
            "Explicit native failure evidence",
        )
        return {
            "neural_fits": row["neural_fits_attempted"],
            "verified_updates": 0,
            "failed": 1,
            "draws": 0,
        }
    seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
    expected_config = model_for(arm, seed, task["draws"] if paired else 0).to_dict()
    # JSON stores tuples as arrays; compare canonical JSON-compatible structures.
    import json

    require(
        row["configuration"] == json.loads(json.dumps(expected_config)),
        "Prespecified point configuration",
    )
    if arm == "integrative_gmm":
        audit_old_point(task, data, folder, row, {})
        return {"neural_fits": 0, "verified_updates": 0, "failed": 0, "draws": 0}
    summary = read_json(folder / "summary.json")
    require(
        summary["unit_ids"] == [f"unit_{i:04d}" for i in range(task["n"])], "Point unit alignment"
    )
    require(summary["configuration"]["model"] == row["configuration"], "Native model configuration")
    for name, digest in summary["artifact_sha256"].items():
        require(fingerprint_file(folder / name).sha256 == digest, "Native artifact fingerprint")
    point, stable = arrays(folder / "point.npz"), arrays(folder / "stability.npz")
    if paired:
        from super_deli.benchmarking.matched_config import ECO, MORPH

        cohort = read_json(folder / "cohort.json")
        require(cohort["unit_ids"] == summary["unit_ids"], "Paired unit ledger")
        for i, (name, features) in enumerate((("morphology", MORPH), ("ecology", ECO))):
            view = cohort["views"][i]
            require(
                view["modality"] == name and view["feature_names"] == list(features),
                "Paired feature order",
            )
            require(
                view["observation_ids"] == [f"{name}_{j:04d}" for j in range(task["n"])],
                "Paired observation ledger",
            )
            expected_rows = np.arange(task["n"]) if i == 0 else np.argsort(data["ecology_order"])
            np.testing.assert_array_equal(view["source_row_indices"], expected_rows)
    require(
        len(set(point["labels"]) - {-1}) == row["selected_k"] == summary["selected_k"],
        "Native selected count",
    )
    records = read_json(folder / "subsamples.json")
    require(
        len(records) == row["draws_requested"] == row["draws_attempted"],
        "Requested/attempted native draws",
    )
    if records:
        _audit_support(task["n"], records, stable)
    else:
        require(not stable, "Unrequested support is empty")
    seeds = np.random.SeedSequence(seed).spawn(len(records))
    for record, s in zip(records, seeds, strict=True):
        ids = np.sort(
            np.random.default_rng(s).choice(task["n"], int(0.8 * task["n"]), replace=False)
        )
        np.testing.assert_array_equal(record["training_indices"], ids)
        require(record["unit_ids"] == [f"unit_{i:04d}" for i in ids], "Paired subsample identities")
    events, evidence = (
        read_json(folder / "optimizer-checks.json"),
        arrays(folder / "optimizer-checks.npz"),
    )
    trace, trace_records = (
        arrays(folder / "density-trace.npz"),
        read_json(folder / "density-trace.json"),
    )
    successful = [(np.arange(task["n"]), point)] + [
        (
            np.array(r["training_indices"]),
            {
                k.removeprefix(f"draw_{i}_"): v
                for k, v in stable.items()
                if k.startswith(f"draw_{i}_")
            },
        )
        for i, r in enumerate(records)
        if r["status"] == "complete"
    ]
    require(
        len(events) == row["neural_fits_attempted"], "Every attempted primary training retained"
    )
    used = set()
    for ids, a in successful:
        x = (
            np.column_stack((data["morphology"][ids], data["ecology"][ids]))
            if paired
            else data[arm.split("_")[0]][ids]
        )
        # Unused successful events preserve actual fitting order even if a prior
        # draw failed before training. Never create a replacement event.
        z = (x[:, a["kept_indices"]] - a["means"][a["kept_indices"]]) / a["scales"][
            a["kept_indices"]
        ]
        if paired:
            z *= a["column_weights"]
        actual_input_hash = digest_arrays({"input": z})
        available = [
            e
            for e in events
            if e["fit"] not in used
            and e["status"] == "complete"
            and e["n"] == len(ids)
            and e["input_sha256"] == actual_input_hash
        ]
        require(bool(available), "Missing successful optimizer trajectory")
        event = available[0]
        index = event["fit"]
        audit_trajectory(x, a, row["configuration"], event, evidence, index, cache, paired=paired)
        used.add(index)
        require(
            any(r["fit"] == index and r["status"] == "complete" for r in trace_records),
            "Native density trace",
        )
        audit_density(a, trace, index)
    # Completed trajectories without a retained successful density result are
    # explicitly counted, never claimed as independently verified endpoints.
    unmatched = [e["fit"] for e in events if e["status"] == "complete" and e["fit"] not in used]
    require(
        len(unmatched) <= sum(r["status"] != "complete" for r in records),
        "Unmatched optimizer event accounting",
    )
    if paired:
        from super_deli.workflows.matched_neural import load_matched_neural_model

        model = load_matched_neural_model(folder)
        np.testing.assert_array_equal(
            model.encode(synthetic_tables(data))["scores"], point["scores"]
        )
        reconstructed = model.reconstruct(synthetic_tables(data))["views"]
        residual = [
            np.mean(((reconstructed[v] - data[v]) / point["scales"][s]) ** 2)
            for v, s in (("morphology", slice(0, 4)), ("ecology", slice(4, 12)))
        ]
        fraction = model.config.loss_fractions()
        np.testing.assert_allclose(np.dot(fraction, residual), point["losses"][-1], atol=1e-11)
    return {
        "neural_fits": len(events),
        "verified_updates": 500 * len(used),
        "failed": sum(e["status"] != "complete" for e in events),
        "draws": len(records),
    }


def audit_dataset(folder, task, *, cache):
    data = audit_sources(folder, task, reference_function=reference)
    records = read_json(folder / "fits.json")
    require([r["arm"] for r in records] == list(ARMS), "Complete ordered arm accounting")
    return [audit_arm(folder / r["arm"], task, data, r, cache) for r in records]
