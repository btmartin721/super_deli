"""Whole-individual subset refits through the unchanged native integrative estimators."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from super_deli.io.config import write_json
from super_deli.models import integrative
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples
from super_deli.models.unit_bootstrap import _bounds


def draw_plan(n, config, model_seed):
    indices = subsample_indices(n, config.n_subsamples, config.fraction, config.random_seed)
    fit_children = np.random.SeedSequence(model_seed).spawn(config.n_subsamples + 1)[1:]
    return [
        (ids, int(child.generate_state(2)[1]))
        for ids, child in zip(indices, fit_children, strict=True)
    ]


def _save_parameters(source, fitted, model, seed, directory, probabilities, projected):
    from super_deli.io.integrative_artifacts import write_integration_parameters
    from super_deli.models.unit_bootstrap import UnitStability

    for i in range(len(source.inputs)):
        (directory / f"view_{i:03d}").mkdir()
    draw = fitted.views[0].training_indices
    # All views refer to the same IDs, though their physical row orders can differ.
    fit = integrative.IntegrativeFit(
        fitted.mixture,
        fitted,
        UnitStability("not_computed", 0, 0, (), {}, (), (), {}),
        (),
        (),
        source.inputs,
        model,
        seed,
    )
    # Members in the parameter export describe the included biological units only.
    targets = source.inputs[0].target_ids
    fit = replace(fit, unit_ids=tuple(targets[i] for i in draw))
    write_integration_parameters(fit, directory)
    m = fitted.mixture.mixture
    with (directory / "fit.npz").open("xb") as f:
        np.savez(
            f,
            training_scores=fitted.scores,
            weights=m.weights_,
            means=m.means_,
            covariances=m.covariances_,
            component_order=np.array(fitted.mixture.component_order),
            training_membership=m.predict_proba(fitted.scores),
            projected_scores=projected,
            projected_membership=probabilities,
            projected_labels=probabilities.argmax(1),
        )


def fit_subsamples(source, config, directory):
    directory = Path(directory)
    directory.mkdir()
    n = len(source.units)
    size = int(np.floor(n * config.fraction))
    model = replace(source.model, n_bootstrap=0)
    if size < model.min_cluster_size:
        raise ValueError("The 80% subset cannot support even one declared component")
    if 24 * n * n * 8 > model.max_pairwise_bytes:
        raise ValueError("Dense included/omitted stability exceeds max_pairwise_bytes")
    records = []
    with FIT_LOCK, threadpool_limits(limits=1):
        for i, (draw, seed) in enumerate(draw_plan(n, config, model.random_seed)):
            output = directory / f"draw_{i:04d}"
            output.mkdir()
            absent = np.setdiff1d(np.arange(n), draw, assume_unique=True)
            record = dict(
                replicate=i,
                training_indices=draw.tolist(),
                training_unit_ids=[source.units[j] for j in draw],
                omitted_indices=absent.tolist(),
                distinct_training_units=len(draw),
                sampling_seed_sequence=dict(entropy=config.random_seed, spawn_key=[i]),
                fit_seed=seed,
                fit_seed_sequence=dict(entropy=model.random_seed, spawn_key=[i + 1], state_word=1),
                status="failed",
                parameters=f"draw_{i:04d}",
            )
            try:
                fitted = integrative._fit_once(source.inputs, source.units, model, seed, draw)
                projected = integrative.project_integrative(source.inputs, source.units, fitted)
                probabilities = fitted.mixture.mixture.predict_proba(projected)
                if not np.isfinite(probabilities).all():
                    raise ValueError("Nonfinite frozen projections")
                labels = probabilities.argmax(1)
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            else:
                _save_parameters(source, fitted, model, seed, output, probabilities, projected)
                record.update(
                    status="complete",
                    selected_k=fitted.mixture.selected_k,
                    labels=labels[draw].tolist(),
                    projected_labels=labels.tolist(),
                    candidates=fitted.mixture.candidates,
                    point_agreement_all=float(adjusted_rand_score(source.labels, labels)),
                    point_agreement_included=float(
                        adjusted_rand_score(source.labels[draw], labels[draw])
                    ),
                    point_agreement_omitted=float(
                        adjusted_rand_score(source.labels[absent], labels[absent])
                    )
                    if len(absent) > 1
                    else None,
                )
            write_json(record, output / "replicate.json")
            records.append(record)
    source.reverify()
    return tuple(records)


def summarize(source, records):
    n = len(source.units)
    included = summarize_subsamples(n, records)
    requested, successful, co = [np.zeros((n, n), dtype=np.uint64) for _ in range(3)]
    for r in records:
        absent = np.asarray(r["omitted_indices"], dtype=int)
        mask = np.zeros(n, dtype=bool)
        mask[absent] = True
        pairs = np.logical_and.outer(mask, mask)
        requested += pairs
        if r["status"] == "complete":
            labels = np.asarray(r["projected_labels"])
            successful += pairs
            co += np.equal.outer(labels, labels) & pairs
    low, high, frequency = _bounds(co, requested, successful)
    arrays = {"included_" + k: v for k, v in included.arrays.items()}
    arrays.update(
        omitted_requested=requested,
        omitted_successful=successful,
        omitted_coassignment_counts=co,
        omitted_coassignment_lower=low,
        omitted_coassignment_upper=high,
        omitted_coassignment=frequency,
    )
    support = []
    for label in range(source.summary["selected_k"]):
        group = set(np.flatnonzero(source.labels == label))
        counts = []
        scores = []
        exact = split = merge = represented = failed = 0
        for r in records:
            ids = np.asarray(r["training_indices"])
            members = group & set(ids)
            counts.append(len(members))
            if not members:
                continue
            represented += 1
            if r["status"] != "complete":
                failed += 1
                continue
            labels = np.asarray(r["labels"])
            parts = [set(ids[labels == j]) for j in np.unique(labels)]
            scores.append(max(len(members & p) / len(members | p) for p in parts))
            exact += members in parts
            split += sum(bool(members & p) for p in parts) > 1
            merge += any(members < p for p in parts)
        support.append(
            dict(
                point_cluster=f"cluster_{label + 1}",
                point_individuals=len(group),
                minimum_included=min(counts),
                maximum_included=max(counts),
                below_declared_minimum=sum(c < source.model.min_cluster_size for c in counts),
                absent=sum(c == 0 for c in counts),
                represented_attempts=represented,
                represented_failed=failed,
                best_jaccard_mean=None if not scores else float(np.mean(scores)),
                best_jaccard_lower=None if not represented else sum(scores) / represented,
                best_jaccard_upper=None
                if not represented
                else (sum(scores) + failed) / represented,
                exact_included_recovery_count=int(exact),
                split_count=int(split),
                merge_count=int(merge),
            )
        )
    complete = [r for r in records if r["status"] == "complete"]
    requested_count = len(records)
    failed_count = requested_count - len(complete)
    ks = [
        dict(
            k=k,
            count=sum(r["selected_k"] == k for r in complete),
            lower=sum(r["selected_k"] == k for r in complete) / requested_count,
            upper=(sum(r["selected_k"] == k for r in complete) + failed_count) / requested_count,
        )
        for k in range(1, source.model.max_k + 1)
    ]
    eligibility = []
    for k in range(1, source.model.max_k + 1):
        statuses = [next(c["status"] for c in r["candidates"] if c["k"] == k) for r in complete]
        eligibility.append(
            dict(
                k=k,
                eligible=statuses.count("eligible"),
                small_component=statuses.count("small_component"),
                insufficient_samples=statuses.count("insufficient_samples"),
                unavailable=failed_count,
            )
        )
    tri = np.triu_indices(n, 1)
    summary = dict(
        status="complete" if not failed_count else "incomplete",
        requested=requested_count,
        successful=len(complete),
        failed=failed_count,
        selected_k_frequencies=ks,
        candidate_eligibility=eligibility,
        included_pair_coverage=float(np.mean(included.arrays["pair_successful"][tri] > 0)),
        omitted_pair_coverage=float(np.mean(successful[tri] > 0)),
        included_denominator="Successful pair inclusions; attempt-based failure bounds retained",
        omitted_denominator=(
            "Pairs with both original individuals omitted; separate requested/successful counts"
        ),
        failure_bounds="Identification bounds, not confidence intervals",
        interpretation=(
            "Conditional sensitivity to leaving out individuals; "
            "no species probabilities or new significance"
        ),
    )
    return summary, arrays, support
