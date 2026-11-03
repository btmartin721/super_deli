"""Independent branch/fusion, numerical, hierarchy and native-denominator audits."""

import json

import numpy as np

from super_deli.benchmarking.density_audit import _audit_support
from super_deli.benchmarking.matched_audit import audit_sources, synthetic_tables
from super_deli.benchmarking.matched_config import ECO, MORPH
from super_deli.benchmarking.matched_steps import digest_arrays
from super_deli.benchmarking.neural_analysis import arrays, neural_forward, require
from super_deli.benchmarking.neural_moons_coordinates import safe_path
from super_deli.benchmarking.separate_config import ARMS, NATIVE, model_for
from super_deli.benchmarking.separate_run import replay_equal
from super_deli.benchmarking.view_weight_audit import audit_arm as audit_reference
from super_deli.benchmarking.view_weight_audit import audit_density, audit_trajectory
from super_deli.benchmarking.view_weight_data import reference
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def four_initial(d):
    """Independent constructor from the frozen widths; never calls the fitted factory."""
    import torch

    result = {}
    for side, widths in (("encoder", (d, 16, 8, 4)), ("decoder", (4, 8, 16, d))):
        for i, (left, right) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            layer = torch.nn.Linear(left, right, dtype=torch.float64, device="cpu")
            for kind in ("weight", "bias"):
                result[f"{side}_{i}_{kind}"] = getattr(layer, kind).detach().numpy().copy()
    return result


def audit_four_latent(z, a):
    raw = neural_forward(z, a, "encoder")
    np.testing.assert_allclose(a["latent_means"], raw.mean(0), atol=1e-11)
    np.testing.assert_allclose(a["latent_scales"], raw.std(0), atol=1e-11)
    np.testing.assert_allclose(a["scores"], (raw - raw.mean(0)) / raw.std(0), atol=1e-10)
    require(a["scores"].shape == (len(z), 4), "Frozen four-coordinate network output")


def audit_fusion(a):
    score = a["scores"]
    require(
        score.ndim == 2 and score.shape[1] == 4 and np.isfinite(score).all(),
        "Four fused coordinates",
    )
    np.testing.assert_allclose(score.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(score.std(0), 1, atol=1e-10)
    singular = np.linalg.svd(score, compute_uv=False)
    rank = int(np.sum(singular > 1e-12 * singular[0]))
    require(rank >= 2 and int(a["fusion_rank"]) == rank, "Recorded intrinsic joint rank")
    np.testing.assert_allclose(a["fusion_singular_values"], singular, atol=1e-11)


def audit_cohort(folder, task, data):
    cohort = read_json(folder / "cohort.json")
    require(
        cohort["unit_ids"] == [f"unit_{i:04d}" for i in range(task["n"])],
        "Aligned independent unit ledger",
    )
    for i, (role, features) in enumerate((("morphology", MORPH), ("ecology", ECO))):
        view = cohort["views"][i]
        require(
            view["modality"] == role and view["feature_names"] == list(features),
            "Declared view/feature order",
        )
        require(
            view["observation_ids"] == [f"{role}_{j:04d}" for j in range(task["n"])],
            "Observation-to-unit identity",
        )
        np.testing.assert_array_equal(
            view["source_row_indices"],
            np.arange(task["n"]) if i == 0 else np.argsort(data["ecology_order"]),
        )


def audit_optimizer_tables(events, evidence):
    verified = 0
    for i, event in enumerate(events):
        require(
            event["fit"] == i and event["status"] in ("complete", "failed"),
            "Training call accounting",
        )
        k = event["updates_checked"]
        require(
            0 <= k <= 500
            and (k == 500 if event["status"] == "complete" else bool(event.get("error"))),
            "Complete or explicit partial optimizer",
        )
        errors = evidence[f"fit_{i}_errors"]
        require(
            errors.size == 4 * k and np.isfinite(errors).all() and np.all(errors >= 0),
            "Retained actual-step discrepancies",
        )
        ratios = evidence[f"fit_{i}_normalized_errors"][:k]
        require(
            ratios.shape == (k, 3)
            and np.isfinite(ratios).all()
            and np.all((ratios >= 0) & (ratios <= 1)),
            "Unchanged strict numerical tolerances",
        )
        np.testing.assert_array_equal(
            evidence[f"fit_{i}_comparison_counts"][:k], np.tile([12, 24, 12], (k, 1))
        )
        hashes = evidence[f"fit_{i}_step_sha256"]
        require(
            hashes.shape == (k,) and all(len(s) == 64 for s in hashes),
            "Every actual optimizer-step hash",
        )
        if event["status"] == "complete":
            actual, ref = (
                evidence[f"fit_{i}_reported_losses"],
                evidence[f"fit_{i}_reference_losses"],
            )
            require(actual.shape == ref.shape == (501,), "Every independent objective evaluation")
            np.testing.assert_allclose(actual, ref, rtol=1e-10, atol=1e-12)
            require(
                digest_arrays({"losses": actual}) == event["loss_sha256"],
                "Training losses bound to observer",
            )
        verified += k
    return verified


def audit_pca(view, a):
    kept = a["kept_indices"]
    scale = view.std(0)
    scale[scale == 0] = 1
    np.testing.assert_array_equal(kept, np.flatnonzero(view.std(0) > 0))
    np.testing.assert_allclose(a["means"], view.mean(0), atol=1e-12)
    np.testing.assert_allclose(a["scales"], scale, atol=1e-12)
    z = (view[:, kept] - a["means"][kept]) / a["scales"][kept]
    np.testing.assert_allclose(a["pca_mean"], z.mean(0), atol=1e-12)
    components = a["components"]
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-11)
    covariance = np.cov(z, rowvar=False)
    values = np.linalg.eigvalsh(covariance)[::-1][:2]
    np.testing.assert_allclose(components @ covariance @ components.T, np.diag(values), atol=1e-10)
    raw = (z - a["pca_mean"]) @ components.T
    np.testing.assert_allclose(a["latent_means"], raw.mean(0), atol=1e-12)
    np.testing.assert_allclose(a["latent_scales"], raw.std(0), atol=1e-12)
    np.testing.assert_allclose(a["scores"], (raw - raw.mean(0)) / raw.std(0), atol=1e-10)
    return float(np.mean((raw @ components + a["pca_mean"] - z) ** 2))


def audit_arm(folder, task, data, row, cache):
    arm = row["arm"]
    seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
    expected = model_for(arm, seed, task["draws"] if arm in NATIVE else 0).to_dict()
    require(
        row["configuration"] == json.loads(json.dumps(expected)),
        "Prespecified family/seed/configuration",
    )
    require(not row.get("numerical_guard_failed"), "Unresolved numerical guard")
    if arm.startswith("joint2_"):
        return audit_reference(
            folder, task, data, {**row, "arm": arm.replace("joint2_", "joint_equal_")}, cache
        )
    if arm not in NATIVE and arm != "pca4":
        return audit_reference(folder, task, data, row, cache)
    neural = arm != "pca4"
    events = read_json(folder / "optimizer-checks.json") if neural else []
    evidence = arrays(folder / "optimizer-checks.npz") if neural else {}
    require(len(events) == row["neural_fits_attempted"], "All attempted neural calls retained")
    verified = audit_optimizer_tables(events, evidence)
    if row["status"] != "available":
        require(
            bool(row.get("error")) and (folder / "failure.json").exists(), "Retained failed point"
        )
        return {
            "neural_fits": len(events),
            "verified_updates": verified,
            "failed": 1,
            "draws": row["draws_attempted"],
        }
    audit_cohort(folder, task, data)
    summary = read_json(folder / "summary.json")
    require(
        summary["configuration"]["model"] == row["configuration"]
        and summary["model"] == expected["model_family"],
        "Actual native family/configuration",
    )
    for name, digest in summary["artifact_sha256"].items():
        require(fingerprint_file(folder / name).sha256 == digest, "Native file fingerprint")
    point, stable = arrays(folder / "point.npz"), arrays(folder / "stability.npz")
    require(
        len(set(point["labels"]) - {-1}) == row["selected_k"] == summary["selected_k"],
        "Selected density count",
    )
    records = read_json(folder / "subsamples.json")
    require(len(records) == row["draws_requested"] == row["draws_attempted"], "Native refit budget")
    if records:
        _audit_support(task["n"], records, stable)
    else:
        require(not stable, "Unrequested support is empty")
    for record, s in zip(records, np.random.SeedSequence(seed).spawn(len(records)), strict=True):
        ids = np.sort(
            np.random.default_rng(s).choice(task["n"], int(0.8 * task["n"]), replace=False)
        )
        np.testing.assert_array_equal(record["training_indices"], ids)
        require(
            record["unit_ids"] == [f"unit_{i:04d}" for i in ids], "Paired original-unit subsets"
        )
    success = [(np.arange(task["n"]), point)] + [
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
    trace = arrays(folder / "density-trace.npz")
    used = set()
    for ids, a in success:
        audit_fusion(a)
        errors = []
        last_index = 0
        if arm.startswith("separate_") or arm == "pca4":
            parts = []
            for role in ("morphology", "ecology"):
                prefix = role + "__"
                b = {k.removeprefix(prefix): v for k, v in a.items() if k.startswith(prefix)}
                x = data[role][ids]
                parts.append(b["scores"])
                if neural:
                    z = (x[:, b["kept_indices"]] - b["means"][b["kept_indices"]]) / b["scales"][
                        b["kept_indices"]
                    ]
                    match = [
                        e
                        for e in events
                        if e["fit"] not in used
                        and e["status"] == "complete"
                        and e["input_sha256"] == digest_arrays({"input": z})
                    ]
                    require(bool(match), "Complete branch trajectory bound to raw IDs")
                    event = match[0]
                    last_index = event["fit"]
                    used.add(last_index)
                    audit_trajectory(
                        x, b, row["configuration"], event, evidence, last_index, cache, paired=False
                    )
                    errors.append(float(b["losses"][-1]))
                else:
                    errors.append(audit_pca(x, b))
            np.testing.assert_array_equal(a["scores"], np.column_stack(parts))
            if neural:
                np.testing.assert_array_equal(
                    a["losses"], 0.5 * (a["morphology__losses"] + a["ecology__losses"])
                )
        else:
            x = np.column_stack((data["morphology"][ids], data["ecology"][ids]))
            z = (
                (x[:, a["kept_indices"]] - a["means"][a["kept_indices"]])
                / a["scales"][a["kept_indices"]]
                * a["column_weights"]
            )
            match = [
                e
                for e in events
                if e["fit"] not in used
                and e["status"] == "complete"
                and e["input_sha256"] == digest_arrays({"input": z})
            ]
            require(bool(match), "Complete four-coordinate trajectory bound to raw IDs")
            event = match[0]
            last_index = event["fit"]
            used.add(last_index)
            canonical = {
                **row["configuration"],
                "model_family": "matched_morphology_ecology_autoencoder_hdbscan",
            }
            audit_trajectory(
                x,
                a,
                canonical,
                event,
                evidence,
                last_index,
                cache,
                paired=True,
                initial_weights=four_initial,
                latent_auditor=audit_four_latent,
            )
            decoded = neural_forward(neural_forward(z, a, "encoder"), a, "decoder")
            residual = (decoded - z) / a["column_weights"]
            split = int(a["retained_view_counts"][0])
            errors = [
                float(np.mean(residual[:, :split] ** 2)),
                float(np.mean(residual[:, split:] ** 2)),
            ]
        audit_density(a, trace, last_index)
        if len(ids) == task["n"]:
            np.testing.assert_allclose(summary["per_view_standardized_mse"], errors, atol=1e-11)
    if arm.startswith("separate_"):
        from super_deli.workflows.separate_neural import load_separate_neural_model

        model = load_separate_neural_model(folder)
        np.testing.assert_array_equal(
            model.encode(synthetic_tables(data))["scores"], point["scores"]
        )
        reconstructed = model.reconstruct(synthetic_tables(data))["views"]
        for role in ("morphology", "ecology"):
            b = {
                k.removeprefix(role + "__"): v
                for k, v in point.items()
                if k.startswith(role + "__")
            }
            kept = b["kept_indices"]
            np.testing.assert_allclose(
                np.mean(
                    ((reconstructed[role][:, kept] - data[role][:, kept]) / b["scales"][kept]) ** 2
                ),
                b["losses"][-1],
                atol=1e-11,
            )
    return {
        "neural_fits": len(events),
        "verified_updates": verified,
        "failed": sum(r["status"] != "complete" for r in records),
        "draws": len(records),
    }


def audit_dataset(folder, task, *, cache):
    data = audit_sources(folder, task, reference_function=reference)
    rows = read_json(folder / "fits.json")
    require([r["arm"] for r in rows] == list(ARMS), "Complete ordered arm accounting")
    results = [audit_arm(folder / r["arm"], task, data, r, cache) for r in rows]
    # Point branches must reproduce the corresponding unchanged standalone fits.
    for i, row in enumerate(rows):
        if not row["arm"].startswith("separate_") or row["status"] != "available":
            continue
        count = 0
        seed = row["arm"].rsplit("_", 1)[1]
        actual = arrays(folder / row["arm"] / "point.npz")
        for role in ("morphology", "ecology"):
            reference_row = next(r for r in rows if r["arm"] == role + "_" + seed)
            if reference_row["status"] != "available":
                continue
            original = arrays(folder / reference_row["arm"] / "point.npz")
            for key, value in original.items():
                if key not in ("labels", "strength"):
                    np.testing.assert_array_equal(actual[role + "__" + key], value)
            branch_events = read_json(folder / row["arm"] / "optimizer-checks.json")
            ref_events = read_json(folder / reference_row["arm"] / "optimizer-checks.json")
            event_index = 0 if role == "morphology" else 1
            require(
                branch_events[event_index]["initial_sha256"] == ref_events[0]["initial_sha256"],
                "Exact seeded branch initialization parity",
            )
            count += 1
        results[i]["branch_parity_checks"] = count
    return results


def audit_replay(out, source, entry, cache):
    require(read_json(out / "replay.json") == entry, "Prespecified archived replay")
    original = safe_path(source, entry["path"])
    data = audit_sources(original, entry["task"], reference_function=reference)
    rows = read_json(out / "fits.json")
    require(len(rows) == 1, "One legacy replay workflow")
    row = rows[0]
    result = audit_arm(out / row["arm"], {**entry["task"], "draws": 0}, data, row, cache)
    require(row.get("exact_numeric_replay"), "Complete exact legacy replay")
    replay_equal(original, out / row["arm"], entry)
    return [result]
