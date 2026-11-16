"""Read-only candidate prediction capture around unchanged native workflow calls."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.mixture import GaussianMixture

from super_deli.io.config import write_json
from super_deli.models import integrative
from super_deli.models.mixture import FIT_LOCK


@contextmanager
def predictions(directory, evaluation):
    directory = Path(directory)
    models = []
    records = []
    native_gmm = GaussianMixture.fit
    native_once = integrative._fit_once

    def remember(obj, x, *a, **kw):
        try:
            return native_gmm(obj, x, *a, **kw)
        finally:
            models.append(obj)

    def once(inputs, units, config, seed, draw, *, fold_minimum=None):
        first = len(models)
        fit = native_once(inputs, units, config, seed, draw, fold_minimum=fold_minimum)
        original = integrative.project_integrative(inputs, units, fit)
        external = integrative.project_integrative(*evaluation, fit)
        arrays = dict(original_scores=original, evaluation_scores=external)
        for obj in models[first:]:
            arrays[f"k{obj.n_components}_evaluation_logdensity"] = obj.score_samples(external)
            arrays[f"k{obj.n_components}_evaluation_membership"] = obj.predict_proba(external)
        arrays["original_membership"] = fit.mixture.mixture.predict_proba(original)
        arrays["original_labels"] = arrays["original_membership"].argmax(1)
        arrays["evaluation_labels"] = arrays[
            f"k{fit.mixture.selected_k}_evaluation_membership"
        ].argmax(1)
        name = f"prediction_{len(records):03d}.npz"
        with (directory / name).open("xb") as f:
            np.savez(f, **arrays)
        records.append(
            dict(
                index=len(records),
                arrays=name,
                selected_k=fit.mixture.selected_k,
                candidates=fit.mixture.candidates,
            )
        )
        return fit

    try:
        with (
            FIT_LOCK,
            patch.object(GaussianMixture, "fit", remember),
            patch.object(integrative, "_fit_once", once),
        ):
            yield records
    finally:
        directory.mkdir(exist_ok=True)
        write_json(records, directory / "predictions.json")
