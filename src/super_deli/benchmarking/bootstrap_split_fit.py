"""Observe unchanged native numerical fits and frozen independent-cohort predictions."""

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.full_retention_observer import observe_pipeline
from super_deli.benchmarking.signal_loss_fit import observe_candidates
from super_deli.io.config import write_json
from super_deli.models import integrative
from super_deli.models.mixture import FIT_LOCK, select_gaussian_mixture


def run(
    directory,
    train,
    original,
    evaluation,
    model,
    *,
    ordinary=False,
    draw=None,
    seed=None,
    head=False,
    draws=25,
):
    directory = Path(directory)
    directory.mkdir()
    models = []
    predictions = []
    native_gmm = GaussianMixture.fit
    native_once = integrative._fit_once

    def remember(obj, x, *args, **kwargs):
        try:
            return native_gmm(obj, x, *args, **kwargs)
        finally:
            models.append(obj)

    def evaluate(fitted, current, start):
        if head:
            source_scores, eval_scores = original, evaluation
            point = current
        else:
            source_scores = integrative.project_integrative(*original, fitted)
            eval_scores = integrative.project_integrative(*evaluation, fitted)
            point = fitted.mixture
        arrays = dict(original_scores=source_scores, evaluation_scores=eval_scores)
        for obj in models[start:]:
            arrays[f"k{obj.n_components}_evaluation_logdensity"] = obj.score_samples(eval_scores)
            arrays[f"k{obj.n_components}_evaluation_membership"] = obj.predict_proba(eval_scores)
        arrays["original_membership"] = point.mixture.predict_proba(source_scores)
        arrays["original_labels"] = arrays["original_membership"].argmax(1)
        arrays["evaluation_labels"] = arrays[f"k{point.selected_k}_evaluation_membership"].argmax(1)
        key = f"prediction_{len(predictions):03d}.npz"
        with (directory / key).open("xb") as f:
            np.savez(f, **arrays)
        predictions.append(
            dict(
                index=len(predictions),
                arrays=key,
                selected_k=point.selected_k,
                candidates=point.candidates,
            )
        )

    def observed_once(inputs, units, config, fit_seed, indices, *, fold_minimum=None):
        start = len(models)
        fitted = native_once(inputs, units, config, fit_seed, indices, fold_minimum=fold_minimum)
        evaluate(fitted, None, start)
        return fitted

    error = None
    fit = None
    events = []
    with ExitStack() as stack:
        stack.enter_context(FIT_LOCK)
        stack.enter_context(threadpool_limits(limits=1))
        stack.enter_context(patch.object(GaussianMixture, "fit", remember))
        records = stack.enter_context(
            observe_candidates(directory / "candidates", initialization=True)
        )
        if not head:
            stack.enter_context(patch.object(integrative, "_fit_once", observed_once))
            events = stack.enter_context(observe_pipeline(directory / "pipeline", records))
        try:
            if head:
                point = select_gaussian_mixture(
                    train[draw], model, int(seed), covariance_type="tied", unit_indices=draw
                )
                evaluate(None, point, 0)
                events.append(
                    dict(
                        index=0,
                        seed=int(seed),
                        draw=np.asarray(draw).tolist(),
                        status="complete",
                        configuration=model.to_dict(),
                        candidates=point.candidates,
                        selected_k=point.selected_k,
                        candidate_start=0,
                        candidate_stop=len(records),
                    )
                )
            elif ordinary:
                fit = integrative.fit_integrative(
                    train[0], config=replace(model, n_bootstrap=draws)
                )
            else:
                integrative._fit_once(*train, model, int(seed), np.asarray(draw))
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if head:
                events.append(
                    dict(
                        index=0,
                        status="failed",
                        error=error,
                        draw=np.asarray(draw).tolist(),
                        seed=int(seed),
                    )
                )
    write_json(predictions, directory / "predictions.json")
    if head:
        write_json(events, directory / "events.json")
        with (directory / "head-inputs.npz").open("xb") as f:
            np.savez(f, train=train, original=original, evaluation=evaluation, draw=draw)
    elif fit is not None:
        write_json(fit.stability.replicates, directory / "native-bootstrap.json")
        with (directory / "native-stability.npz").open("xb") as f:
            np.savez(f, **fit.stability.arrays)
    result = dict(
        status="fit_error" if error else "complete",
        error=error,
        head_only=head,
        configuration=model.to_dict(),
        events=len(events),
        predictions=len(predictions),
    )
    write_json(result, directory / "fit.json")
    return result
