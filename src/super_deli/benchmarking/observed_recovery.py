"""Diagnostic continuation and oracle starts; never changes production selection."""

import hashlib
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.io.config import read_json, write_json
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.observed_config import ObservedGMMConfig
from super_deli.models.observed_gmm import Normalization, ObservedFit, fit_start
from super_deli.models.observed_likelihood import (
    Parameters,
    constrain_covariance,
    evaluate,
    patterns,
)

PARAMETERS = ("weights", "means", "covariances")
TOTAL_CAP = 10000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def get_parameters(arrays, prefix=""):
    return Parameters(*(arrays[prefix + name] for name in PARAMETERS))


def put_parameters(arrays, prefix, parameters):
    arrays.update({prefix + name: getattr(parameters, name) for name in PARAMETERS})


def continued_start(x, arrays, prefix, record, config, total_cap=TOTAL_CAP):
    """Extend exactly one exhausted path; original arrays/record remain immutable."""
    if (
        record["status"] != "failed"
        or record.get("error") != "Iteration limit reached"
        or record["iterations"] != config.max_iter
        or not config.max_iter < total_cap <= 20000
    ):
        raise ValueError("Only an exhausted original path can be continued")
    original_trace = arrays[prefix + "trace"]
    if len(original_trace) != config.max_iter + 1 or not np.isfinite(original_trace).all():
        raise ValueError("Invalid original trace anchor")
    initial = get_parameters(arrays, prefix + "final_")
    groups = patterns(x)
    value = float(evaluate(len(x), initial, groups)[0].sum())
    if value != original_trace[-1] or value != record["log_likelihood"]:
        raise ValueError("Continuation objective anchor changed")
    updated, evidence = fit_start(
        x, initial, groups, replace(config, max_iter=total_cap - config.max_iter)
    )
    changed = {
        **record,
        **updated,
        "iterations": config.max_iter + updated["iterations"],
        "original_iterations": config.max_iter,
        "additional_iterations": updated["iterations"],
        "original_status": record["status"],
        "original_error": record["error"],
    }
    output = {prefix + "trace": np.concatenate([original_trace, evidence["trace"][1:]])}
    put_parameters(output, prefix + "anchor_", initial)
    for name in ("previous", "final"):
        put_parameters(output, prefix + name + "_", evidence[name])
    return changed, output


def assemble(candidates, arrays, config):
    """Reconstruct a separate diagnostic choice using the unchanged production rule."""
    x = (arrays["raw"] - arrays["normalization_means"]) / arrays["scales"]
    groups = patterns(x)
    accepted, rebuilt = [], []
    failed = False
    for candidate in candidates:
        k = candidate["k"]
        starts = candidate["starts"]
        if candidate["status"] == "insufficient_samples":
            rebuilt.append(candidate)
            continue
        converged = [
            (s["log_likelihood"], -s["start"], s) for s in starts if s["status"] == "converged"
        ]
        if not converged:
            rebuilt.append(dict(k=k, status="numerical_failure", starts=starts))
            failed = True
            continue
        likelihood, neg, _ = max(converged, key=lambda item: (item[0], item[1]))
        p = get_parameters(arrays, f"k{k}_s{-neg}_final_")
        density, membership = evaluate(len(x), p, groups)
        sizes = np.bincount(membership.argmax(axis=1), minlength=k)
        status = "eligible" if sizes.min() >= config.min_cluster_size else "small_component"
        count = k * x.shape[1] + k * x.shape[1] * (x.shape[1] + 1) // 2 + k - 1
        bic = float(-2 * likelihood + count * np.log(len(x)))
        eigen = np.linalg.eigvalsh(p.covariances)
        rebuilt.append(
            dict(
                k=k,
                status=status,
                starts=starts,
                log_likelihood=likelihood,
                best_start=-neg,
                hard_cluster_sizes=sizes.tolist(),
                parameter_count=count,
                bic=bic,
                eigenvalues=eigen.tolist(),
                active_bound_eigenvalues=int(np.sum(eigen <= config.covariance_bound * (1 + 1e-7))),
            )
        )
        put_parameters(arrays, f"k{k}_", p)
        arrays[f"k{k}_membership"] = membership
        arrays[f"k{k}_density_scaled"] = density
        if status == "eligible":
            accepted.append((bic, k, p, membership))
    selected = None
    if accepted and not failed:
        _, k, p, membership = min(accepted, key=lambda item: (item[0], item[1]))
        labels = membership.argmax(axis=1)
        order = np.array(sorted(range(k), key=lambda c: int(np.flatnonzero(labels == c)[0])))
        selected = Parameters(*(getattr(p, name)[order] for name in PARAMETERS))
        put_parameters(arrays, "", selected)
        arrays["membership"] = membership[:, order]
        arrays["labels"] = arrays["membership"].argmax(axis=1)
    return rebuilt, selected


def extend_search(
    baseline,
    destination,
    evaluation,
    eval_mask,
    truth,
    total_cap=TOTAL_CAP,
    *,
    components=None,
    saved=None,
):
    baseline, destination = Path(baseline), Path(destination)
    summary = read_json(baseline / "summary.json")
    config = ObservedGMMConfig.from_dict(summary["config"])
    original = load_arrays(baseline / "fit.npz")
    if components is not None and (
        not components
        or len(set(components)) != len(components)
        or any(type(k) is not int or not 1 <= k <= config.max_k for k in components)
    ):
        raise ValueError("Invalid continuation component scope")
    saved_summary = read_json(Path(saved) / "summary.json") if saved is not None else None
    saved_arrays = load_arrays(Path(saved) / "fit.npz") if saved is not None else None
    if saved_summary is not None:
        if saved_summary["total_iteration_cap"] != total_cap or saved_summary[
            "baseline_sha256"
        ] != {name: digest(baseline / name) for name in ("summary.json", "fit.npz")}:
            raise ValueError("Saved continuation source/budget mismatch")
    arrays = {
        name: value.copy()
        for name, value in original.items()
        if re.match(r"k\d+_s\d+_", name)
        or name
        in ("raw", "observed", "coobserved", "kept_indices", "normalization_means", "scales")
    }
    x = (arrays["raw"] - arrays["normalization_means"]) / arrays["scales"]
    candidates, attempts = [], 0
    with FIT_LOCK, threadpool_limits(limits=1):
        for candidate in summary["candidates"]:
            starts = []
            for record in candidate["starts"]:
                if (
                    record["status"] == "failed"
                    and record.get("error") == "Iteration limit reached"
                    and (components is None or candidate["k"] in components)
                ):
                    prefix = f"k{candidate['k']}_s{record['start']}_"
                    if saved_summary is None:
                        record, changed = continued_start(
                            x, arrays, prefix, record, config, total_cap
                        )
                    else:
                        prior = next(
                            c for c in saved_summary["candidates"] if c["k"] == candidate["k"]
                        )
                        reused = next(s for s in prior["starts"] if s["start"] == record["start"])
                        if (
                            reused.get("original_iterations") != config.max_iter
                            or reused["iterations"] > total_cap
                            or not np.array_equal(
                                saved_arrays[prefix + "trace"][: config.max_iter + 1],
                                arrays[prefix + "trace"],
                            )
                            or any(
                                not np.array_equal(
                                    saved_arrays[prefix + "anchor_" + name],
                                    arrays[prefix + "final_" + name],
                                )
                                for name in PARAMETERS
                            )
                        ):
                            raise ValueError("Saved continuation anchor changed")
                        changed = {
                            key: value
                            for key, value in saved_arrays.items()
                            if key.startswith(prefix)
                        }
                        record = dict(reused)
                    arrays.update(changed)
                    attempts += 1
                starts.append(record)
            candidates.append({**candidate, "starts": starts})
        candidates, selected = assemble(candidates, arrays, config)
    # Remove obsolete successful-result fields before possibly withholding a result.
    result = {
        key: value
        for key, value in summary.items()
        if key
        not in (
            "ari",
            "mean_log_density",
            "complete_mean_log_density",
            "evaluation_coverage",
            "entropy",
        )
    }
    result.update(
        status="complete" if selected is not None else "failed",
        selected_k=None if selected is None else len(selected.weights),
        candidates=candidates,
        diagnostic_schema_version=1,
        total_iteration_cap=total_cap,
        continuation_attempts=attempts,
        baseline_sha256={name: digest(baseline / name) for name in ("summary.json", "fit.npz")},
        original_result_replaced=False,
        oracle_used=False,
    )
    if components is not None:
        result["continued_components"] = list(components)
    if saved is not None:
        result["reused_without_refitting"] = True
        result["reused_evidence_sha256"] = {
            name: digest(Path(saved) / name) for name in ("summary.json", "fit.npz")
        }
    destination.mkdir()
    np.savez_compressed(destination / "fit.npz", **arrays)
    if selected is not None:
        from sklearn.metrics import adjusted_rand_score

        fit = ObservedFit(
            result,
            arrays,
            Normalization(arrays["normalization_means"], arrays["scales"]),
            selected,
            arrays["kept_indices"],
        )
        density, membership = fit.predict(np.where(eval_mask, np.nan, evaluation))
        complete_density, complete_membership = fit.predict(evaluation)
        np.savez_compressed(
            destination / "evaluation.npz",
            density=density,
            membership=membership,
            complete_density=complete_density,
            complete_membership=complete_membership,
        )
        result.update(
            ari=float(adjusted_rand_score(truth[summary["source_indices"]], arrays["labels"])),
            mean_log_density=float(np.nanmean(density)),
            complete_mean_log_density=float(complete_density.mean()),
            evaluation_coverage=int(np.isfinite(density).sum()),
        )
    write_json(result, destination / "summary.json")
    return result


def oracle_initial(mean, scale, bound):
    signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    means = np.column_stack([3 * signs, 1.5 * signs])
    covariance = 0.3 ** np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
    constrained = constrain_covariance(covariance / scale[:, None] / scale[None, :], bound)
    return Parameters(np.full(4, 0.25), (means - mean) / scale, np.repeat(constrained[None], 4, 0))


def oracle_start(baseline, destination):
    """Known-law initialization is isolated and never enters an unsupervised choice."""
    baseline, destination = Path(baseline), Path(destination)
    summary = read_json(baseline / "summary.json")
    if summary["task"]["case"] != "four_groups":
        raise ValueError("Known four-group initialization requires its declared generating law")
    original = load_arrays(baseline / "fit.npz")
    config = replace(ObservedGMMConfig.from_dict(summary["config"]), max_iter=TOTAL_CAP)
    arrays = {
        name: original[name] for name in ("raw", "normalization_means", "scales", "kept_indices")
    }
    x = (arrays["raw"] - arrays["normalization_means"]) / arrays["scales"]
    initial = oracle_initial(
        arrays["normalization_means"], arrays["scales"], config.covariance_bound
    )
    with FIT_LOCK, threadpool_limits(limits=1):
        record, evidence = fit_start(x, initial, patterns(x), config)
    for stage in ("initial", "previous", "final"):
        put_parameters(arrays, stage + "_", evidence[stage])
    arrays["trace"] = evidence["trace"]
    eigen = np.linalg.eigvalsh(evidence["final"].covariances)
    result = dict(
        diagnostic_schema_version=1,
        **record,
        task=summary["task"],
        mask=summary["mask"],
        arm=summary["arm"],
        source_indices=summary["source_indices"],
        total_iteration_cap=TOTAL_CAP,
        excluded_from_selection=True,
        baseline_sha256={name: digest(baseline / name) for name in ("summary.json", "fit.npz")},
        initial_active_bound_eigenvalues=int(
            np.sum(np.linalg.eigvalsh(initial.covariances) <= config.covariance_bound * (1 + 1e-7))
        ),
        eigenvalues=eigen.tolist(),
    )
    if record["status"] == "converged":
        _, membership = evaluate(len(x), evidence["final"], patterns(x))
        arrays["membership"] = membership
        sizes = np.bincount(membership.argmax(axis=1), minlength=4)
        result.update(hard_cluster_sizes=sizes.tolist(), eligible=bool(sizes.min() >= 5))
    destination.mkdir()
    np.savez_compressed(destination / "fit.npz", **arrays)
    write_json(result, destination / "summary.json")
    return result
