"""Independent source, full optimizer, native-support and exact-point replay audit."""

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.density_audit import _audit_support
from super_deli.benchmarking.matched_config import ARMS, ECO, MORPH, tasks
from super_deli.benchmarking.matched_data import configuration
from super_deli.benchmarking.matched_reference import (
    audit_density,
    audit_integrative,
    audit_latent,
    audit_optimizer,
    audit_pca,
    balanced_inputs,
    reference_data,
)
from super_deli.benchmarking.matched_study import check_frozen, guard, inventory
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.unknown_k_reference import audit_candidates
from super_deli.core.data import Feature
from super_deli.core.provenance import fingerprint_file
from super_deli.core.tabular import TabularTable
from super_deli.io.config import read_json, write_json


def synthetic_tables(data, indices=None):
    n = len(data["morphology"])
    indices = np.arange(n) if indices is None else indices
    result = []
    for name, kind, features in (
        ("morphology", "morphological", MORPH),
        ("ecology", "ecological", ECO),
    ):
        result.append(
            TabularTable(
                name,
                kind,
                "specimen",
                tuple(Feature(name=f, dtype="continuous") for f in features),
                tuple(f"{name}_{i:04d}" for i in indices),
                tuple(f"unit_{i:04d}" for i in indices),
                tuple(tuple(row) for row in data[name][indices].tolist()),
            )
        )
    return tuple(result)


def audit_sources(folder, task, *, reference_function=reference_data):
    data = arrays(folder / "generating-reference.npz")
    ref = reference_function(task)
    require(set(data) == set(ref), "Generating reference schema")
    for key in data:
        np.testing.assert_allclose(data[key], ref[key], atol=1e-11)
    for canonical in (False, True):
        base = folder / ("canonical" if canonical else "raw")
        ds = read_json(base / "dataset.json")
        require(
            [s["specimen_id"] for s in ds["specimens"]]
            == [f"unit_{i:04d}" for i in range(task["n"])],
            "Declared original identities",
        )
        for name, names in (("morphology", MORPH), ("ecology", ECO)):
            order = (
                np.arange(task["n"]) if canonical or name == "morphology" else data["ecology_order"]
            )
            with (base / f"{name}.tsv").open(newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
                require(
                    reader.fieldnames == ["observation_id", *names],
                    "No truth/identity proxy features",
                )
            require(
                [r["observation_id"] for r in rows] == [f"{name}_{i:04d}" for i in order],
                "Shuffled source observation mapping",
            )
            np.testing.assert_array_equal(
                [[float(r[f]) for f in names] for r in rows], data[name][order]
            )
            m = next(m for m in ds["modalities"] if m["name"] == name)
            require(
                m["level"] == "specimen"
                and m["observations"]
                == [
                    {"observation_id": f"{name}_{i:04d}", "unit_id": f"unit_{i:04d}"}
                    for i in range(task["n"])
                ],
                "One-to-one declared join",
            )
    return data


def audit_point(task, data, folder, row, cache, optimizer_checks=None, step_checks=None):
    arm = row["arm"]
    paired = arm.startswith("paired_")
    neural = arm.rsplit("_", 1)[-1] in {"0", "101", "202"}
    config = row["configuration"]
    summary = read_json(folder / "summary.json")
    actual = summary.get("configuration", summary.get("provenance", {}).get("configuration"))[
        "model"
    ]
    require(actual == config, "Frozen model configuration")
    units = [f"unit_{i:04d}" for i in range(task["n"])]
    require(summary["unit_ids"] == units, "Canonical unit labels")
    point = arrays(folder / ("mixture.npz" if arm.endswith("gmm") else "point.npz"))
    require(
        len(point["labels"]) == task["n"]
        and len(set(point["labels"]) - {-1}) == row["selected_k"] == summary["selected_k"],
        "Point count/identities",
    )
    if arm.endswith("gmm"):
        audit_candidates(folder, task["n"])
        trace = arrays(folder / "candidate-trace.npz")
        k = row["selected_k"]
        # The native mixture canonicalizes component order after the estimator
        # trace. Match components by their fitted means, never by truth labels.
        permutation = np.array(
            [np.argmin(np.linalg.norm(trace[f"k{k}_means"] - mu, axis=1)) for mu in point["means"]]
        )
        require(len(set(permutation)) == k, "Bijective native component canonicalization")
        for name in ("weights", "means", "covariances"):
            np.testing.assert_allclose(point[name], trace[f"k{k}_{name}"][permutation], atol=1e-12)
        membership = trace[f"k{k}_membership"][:, permutation]
        np.testing.assert_allclose(point["membership"], membership, atol=1e-12)
        np.testing.assert_array_equal(point["labels"], membership.argmax(axis=1))
        candidates = read_json(folder / "candidates.json")
        require(
            min(
                (c for c in candidates if c["status"] == "eligible"),
                key=lambda c: (c["bic"], c["k"]),
            )["k"]
            == k,
            "Eligible minimum BIC selection",
        )
        if arm == "integrative_gmm":
            expected = audit_integrative(data["morphology"], data["ecology"], folder)
        else:
            prep = arrays(folder / "preprocessing.npz")
            view = arm.split("_")[0]
            audit_pca(data[view], prep, point["scores"])
            expected = point["scores"]
        np.testing.assert_allclose(point["scores"], expected, atol=1e-10)
        return 0, 0
    if paired:
        cohort = read_json(folder / "cohort.json")
        require(cohort["unit_ids"] == units, "Paired unit ledger")
        for i, (name, features) in enumerate((("morphology", MORPH), ("ecology", ECO))):
            view = cohort["views"][i]
            require(
                view["modality"] == name and view["feature_names"] == list(features),
                "Paired view/feature order",
            )
            require(
                view["observation_ids"] == [f"{name}_{j:04d}" for j in range(task["n"])],
                "Paired observation ledger",
            )
            expected = np.arange(task["n"]) if i == 0 else np.argsort(data["ecology_order"])
            np.testing.assert_array_equal(view["source_row_indices"], expected)
    for relative, digest in summary.get("artifact_sha256", {}).items():
        require(
            fingerprint_file(folder / relative).sha256 == digest, "Workflow artifact fingerprint"
        )
    subs = read_json(folder / "subsamples.json")
    stable = arrays(folder / "stability.npz")
    require(len(subs) == row["draws_requested"], "Native perturbation budget")
    seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
    expected_draws = [
        np.sort(np.random.default_rng(s).choice(task["n"], int(0.8 * task["n"]), replace=False))
        for s in np.random.SeedSequence(seed).spawn(len(subs))
    ]
    for s, indices in zip(subs, expected_draws, strict=True):
        np.testing.assert_array_equal(s["training_indices"], indices)
        require(
            s["unit_ids"] == [units[j] for j in indices], "Matched identical individual subsets"
        )
    if stable:
        _audit_support(task["n"], subs, stable)
    else:
        require(
            not subs and summary["stability"]["status"] == "not_computed",
            "Empty native support is only valid when unrequested",
        )
    require(row["draws_attempted"] == len(subs), "Native attempt accounting")
    initial = arrays(folder / "initial-weights.npz") if neural else {}
    events = read_json(folder / "training-events.json") if neural else []
    require(
        not neural or len(events) == row["neural_fits_attempted"],
        "All neural initializations retained",
    )
    records = [(np.arange(task["n"]), point, 0)] + [
        (
            np.array(s["training_indices"]),
            {k[len(f"draw_{i}_") :]: v for k, v in stable.items() if k.startswith(f"draw_{i}_")},
            i + 1,
        )
        for i, s in enumerate(subs)
        if s["status"] == "complete"
    ]
    checked = 0
    for indices, a, event_index in records:
        x = (
            np.column_stack((data["morphology"][indices], data["ecology"][indices]))
            if paired
            else data[arm.split("_")[0]][indices]
        )
        if neural:
            z = balanced_inputs(x, a, paired)
            weights = {
                k[len(f"fit_{event_index}_") :]: v
                for k, v in initial.items()
                if k.startswith(f"fit_{event_index}_")
            }
            require(
                len(weights) == 12
                and next(e for e in events if e["fit"] == event_index)["dimensions"] == z.shape[1],
                "Complete fit-specific initialization",
            )
            key = (z.shape[1], config["random_seed"])
            if key not in cache:
                import torch

                from super_deli.models.tabular_neural import _network, _weights

                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(key[1])
                    cache[key] = _weights(_network(key[0]))
            for name in weights:
                np.testing.assert_array_equal(weights[name], cache[key][name])
            if step_checks is None:
                numerical_check = audit_optimizer(z, a, weights)
            else:
                from super_deli.benchmarking.matched_steps import digest_arrays

                evidence = step_checks[task["key"], arm, row["kind"], event_index]
                require(
                    evidence["passed"]
                    and evidence["updates"] == 500
                    and evidence["exact_initial_final_loss_parity"],
                    "Complete actual-path numerical verification",
                )
                require(
                    evidence["input_sha256"] == digest_arrays({"input": z})
                    and evidence["initial_sha256"] == digest_arrays(weights)
                    and evidence["final_sha256"] == digest_arrays({k: a[k] for k in weights})
                    and evidence["loss_sha256"] == digest_arrays({"losses": a["losses"]}),
                    "Every verified trajectory bound to the exact original fit",
                )
                audit_latent(z, a)
                numerical_check = {
                    "updates": 500,
                    "max_loss_absolute_difference": 0.0,
                    "max_parameter_absolute_difference": 0.0,
                    "mode": "independent_actual_path_steps_and_exact_training_replay",
                    "per_step_max_errors": evidence["max_errors"],
                }
            if optimizer_checks is not None:
                optimizer_checks.append(
                    {
                        "dataset": task["key"],
                        "arm": arm,
                        "kind": row["kind"],
                        "fit_index": event_index,
                        **numerical_check,
                    }
                )
            checked += 1
            if paired:
                from super_deli.models.matched_neural import MatchedNeuralModel
                from super_deli.models.matched_neural_config import MatchedNeuralConfig

                model = MatchedNeuralModel(MatchedNeuralConfig.from_dict(config), a)
                tables = synthetic_tables(data, indices)
                encoded = model.encode(tables)
                reconstructed = model.reconstruct(tables)
                np.testing.assert_allclose(encoded["scores"], a["scores"], atol=1e-11)
                residuals = [
                    np.mean(
                        (
                            (reconstructed["views"][name] - data[name][indices])
                            / a["scales"][section]
                        )
                        ** 2
                    )
                    for name, section in (("morphology", slice(0, 4)), ("ecology", slice(4, 12)))
                ]
                np.testing.assert_allclose(0.5 * sum(residuals), a["losses"][-1], atol=1e-11)
            else:
                from super_deli.models.tabular_neural import NeuralModel

                np.testing.assert_allclose(
                    NeuralModel(tuple(config["feature_names"]), a).encode(x),
                    a["scores"],
                    atol=1e-11,
                )
        else:
            audit_pca(x, a, a["scores"])
        audit_density(a)
    if paired:
        from super_deli.workflows.matched_neural import load_matched_neural_model

        model = load_matched_neural_model(folder)
        np.testing.assert_array_equal(
            model.encode(synthetic_tables(data))["scores"], point["scores"]
        )
        np.testing.assert_allclose(
            summary["per_view_standardized_mse"],
            [
                np.mean(
                    (
                        (model.reconstruct(synthetic_tables(data))["views"][name] - data[name])
                        / point["scales"][section]
                    )
                    ** 2
                )
                for name, section in (("morphology", slice(0, 4)), ("ecology", slice(4, 12)))
            ],
            atol=1e-11,
        )
    return checked, len(subs)


def replay_point(original, repeated):
    name = "mixture.npz" if (original / "mixture.npz").exists() else "point.npz"
    a, b = arrays(original / name), arrays(repeated / name)
    require(set(a) == set(b), "Point replay numeric schema")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    if (original / "initial-weights.npz").exists():
        a, b = arrays(original / "initial-weights.npz"), arrays(repeated / "initial-weights.npz")
        for key in b:
            np.testing.assert_array_equal(a[key], b[key])


def audit_phase(root, phase):
    root = Path(root)
    start = time.monotonic()
    f, config = check_frozen(root)
    require(
        inventory(root / phase) == read_json(root / f"{phase}-inventory.json"),
        "Complete result inventory",
    )
    checked = draws = replays = attempts = 0
    statuses = Counter()
    cache = {}
    optimizer_checks = []
    step_checks = None
    step_path = root / f"step-{phase}-verification-audit.json"
    if step_path.exists() or (root / "audit-amendment-approval.json").exists():
        from super_deli.benchmarking.matched_step_study import approval

        approval(root)
        step_audit = read_json(step_path)
        require(
            step_audit["passed"]
            and step_audit["freeze_sha256"] == fingerprint_file(root / "frozen.json").sha256,
            "Passed actual-path verification required",
        )
        require(
            inventory(root / f"step-{phase}-verification") == step_audit["evidence_inventory"],
            "Unchanged per-step evidence",
        )
        step_checks = {
            (r["dataset"], r["arm"], r["kind"], r["fit_index"]): r for r in step_audit["records"]
        }
        require(
            len(step_checks) == f["counts"][phase]["neural_fits"],
            "Exact prespecified verification grid",
        )
    datasets = []
    with threadpool_limits(limits=1):
        for number, task in enumerate(tasks(config, phase), 1):
            out = root / phase / task["key"]
            require(read_json(out / "task.json") == task, "Frozen source task")
            data = audit_sources(out, task)
            rows = read_json(out / "fits.json")
            require(
                len(rows) == 14 * (1 + task["replay"])
                and {(r["arm"], r["kind"]) for r in rows}
                == {
                    (a, k)
                    for a in ARMS
                    for k in (("point", "replay") if task["replay"] else ("point",))
                },
                "Exact arm/replay grid",
            )
            seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
            for row in rows:
                arm = row["arm"]
                folder = out / (arm + ("_replay" if row["kind"] == "replay" else ""))
                require(read_json(folder / "execution.json") == row, "Frozen per-workflow ledger")
                require(
                    row["configuration"]
                    == json.loads(
                        json.dumps(configuration(arm, seed, row["draws_requested"]).to_dict())
                    ),
                    "Prespecified arm settings",
                )
                expected_draws = (
                    task["draws"] if row["kind"] == "point" and not arm.endswith("gmm") else 0
                )
                require(row["draws_requested"] == expected_draws, "Native point/refit scope")
                if row["status"] == "failed":
                    require(
                        bool(row["error"]) and not (folder / "assignments.csv").exists(),
                        "Failure retained without assignments",
                    )
                else:
                    c, d = audit_point(
                        task, data, folder, row, cache, optimizer_checks, step_checks
                    )
                    checked += c
                    draws += d
                    if row["kind"] == "replay":
                        replay_point(out / arm, folder)
                        replays += 1
                attempts += 1 + row["draws_attempted"]
                statuses[row["status"]] += 1
            datasets.append({"key": task["key"], "n": task["n"], "records": rows})
            guard(root, config)
            if number % 10 == 0 or number == len(tasks(config, phase)):
                print({"audit": phase, "observations": number}, flush=True)
            if time.monotonic() - start > config.max_projected_seconds:
                raise ValueError("Audit runtime limit exceeded")
    execution = read_json(root / f"{phase}-execution.json")
    require(execution["datasets"] == datasets, "Complete execution ledger")
    require(all(execution[k] == v for k, v in f["counts"][phase].items()), "Frozen workload counts")
    checks_path = root / f"{phase}-training-audit.json"
    write_json(optimizer_checks, checks_path)
    require(len(optimizer_checks) == checked, "Every completed optimizer audit retained")
    result = {
        "passed": True,
        "phase": phase,
        **f["counts"][phase],
        "actual_workflow_attempts": attempts,
        "numpy_optimizer_replays": checked,
        "optimizer_updates_checked": 500 * checked,
        "training_audit_sha256": fingerprint_file(checks_path).sha256,
        "training_audit_mode": "actual_path_per_step"
        if step_checks is not None
        else "serial_rollout",
        "step_verification_audit_sha256": fingerprint_file(step_path).sha256
        if step_checks is not None
        else None,
        "max_training_loss_absolute_difference": max(
            (r["max_loss_absolute_difference"] for r in optimizer_checks), default=0
        ),
        "max_training_parameter_absolute_difference": max(
            (r["max_parameter_absolute_difference"] for r in optimizer_checks), default=0
        ),
        "native_draws_audited": draws,
        "successful_point_replays": replays,
        "point_statuses_including_replays": dict(statuses),
        "seconds": time.monotonic() - start,
        "execution_sha256": fingerprint_file(root / f"{phase}-execution.json").sha256,
        "audit_boundary": (
            "Independent source/gradient/Adam/PCA/density-input/support equations; "
            "HDBSCAN invariants, not a second HDBSCAN implementation. "
            "Failed fits retain error/partial evidence without a validity claim."
        ),
    }
    write_json(result, root / f"{phase}-audit.json")
    return result
