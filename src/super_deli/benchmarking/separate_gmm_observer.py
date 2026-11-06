"""Read-only per-point/per-draw candidate evidence without retraining for heads."""

import time
from contextlib import contextmanager

import numpy as np

from super_deli.benchmarking.coordinate_fit import capture_candidates
from super_deli.io.config import write_json
from super_deli.models.tabular_neural import point_arrays


def evaluation_path(root, index, head):
    return root / "evaluations" / ("point" if index == 0 else f"draw_{index - 1}") / head


@contextmanager
def retain_points(root):
    from super_deli.models import separate_gmm

    original_point, original_select = (
        separate_gmm.fit_shared_point,
        separate_gmm.select_gaussian_mixture,
    )
    points = []
    current = {"index": -1}

    def select(scores, config, seed, **kwargs):
        out = evaluation_path(root, current["index"], "gmm_shared")
        out.mkdir(parents=True)
        started = time.monotonic()
        row = dict(
            configuration=dict(
                covariance_type="tied",
                max_k=config.max_k,
                min_cluster_size=config.min_cluster_size,
                n_init=config.n_init,
                max_iter=config.max_iter,
                tol=config.tol,
                reg_covar=config.reg_covar,
                random_seed=seed,
                selection_criterion="bic",
                max_working_bytes=config.max_working_bytes,
            )
        )
        with capture_candidates(scores) as (attempts, saved):
            try:
                fitted = original_select(scores, config, seed, **kwargs)
                row.update(
                    status="available",
                    selected_k=fitted.selected_k,
                    candidates=list(fitted.candidates),
                    component_order=list(fitted.component_order),
                )
                saved.update(labels=fitted.labels, membership=fitted.membership)
                return fitted
            except Exception as exc:
                row.update(status="failed", selected_k=None, error=str(exc))
                raise
            finally:
                row["candidate_attempts"] = attempts
                row["seconds"] = time.monotonic() - started
                np.savez_compressed(out / "fit.npz", **saved)
                write_json(row, out / "execution.json")

    def point(x, *, config):
        started = time.monotonic()
        index = len(points)
        current["index"] = index
        row = dict(index=index, status="started")
        points.append(row)
        out = evaluation_path(root, index, "coordinates")
        out.mkdir(parents=True)
        try:
            result = original_point(x, config=config)
            saved = point_arrays(result)
            row.update(status="available", selected_k=result.selected_k)
            return result
        except Exception as exc:
            saved = getattr(exc, "arrays", {})
            row.update(status="failed", error=str(exc))
            raise
        finally:
            row["seconds"] = time.monotonic() - started
            np.savez_compressed(out / "point.npz", **saved)
            write_json(row, out / "execution.json")

    separate_gmm.fit_shared_point, separate_gmm.select_gaussian_mixture = (point, select)
    try:
        yield points
    finally:
        separate_gmm.fit_shared_point, separate_gmm.select_gaussian_mixture = (
            original_point,
            original_select,
        )
