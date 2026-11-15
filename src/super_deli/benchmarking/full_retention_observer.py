"""Read-only native fit snapshots, including each raw-array bootstrap refit."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from super_deli.benchmarking.multimodal_validation import array_hash
from super_deli.io.config import write_json
from super_deli.models import integrative


def _view_arrays(fitted, prefix, arrays):
    arrays.update(
        {
            prefix + "profiles": fitted.profiles.copy(),
            prefix + "mean": fitted.mean.copy(),
            prefix + "scale": fitted.scale,
            prefix + "weight": fitted.normalized_weight,
            prefix + "indices": fitted.training_indices.copy(),
        }
    )
    if fitted.config.kind == "morphometric":
        model = fitted.model
        arrays.update(
            {
                prefix + "consensus": model.consensus,
                prefix + "pca_mean": model.tangent_mean,
                prefix + "components": model.components,
                prefix + "variance": model.explained_variance,
            }
        )
    else:
        if fitted.config.kind == "genomic":
            pre, pca = fitted.model
        else:
            pre, pca = fitted.model, fitted.model.pca
        arrays.update(
            {
                prefix + "kept": pre.kept_indices,
                prefix + "feature_means": pre.means,
                prefix + "feature_scales": pre.scales,
                prefix + "pca_mean": pca.mean_,
                prefix + "components": pca.components_,
                prefix + "variance": pca.explained_variance_,
            }
        )


@contextmanager
def observe_pipeline(directory, candidates):
    directory = Path(directory)
    directory.mkdir()
    records, state = [], {}
    native_once, native_view = integrative._fit_once, integrative._fit_view

    def view(data, config, units, draw, model, weight):
        lookup = {u: i for i, u in enumerate(data.target_ids)}
        indices = [lookup[units[i]] for i in draw]
        state["record"]["attempted_views"].append(
            dict(kind=data.kind, raw_training_sha256=array_hash(data.values[indices]))
        )
        fitted = native_view(data, config, units, draw, model, weight)
        i = len(state["record"]["views"])
        prefix = f"v{i}_"
        _view_arrays(fitted, prefix, state["arrays"])
        state["record"]["views"].append(
            dict(
                modality=data.name,
                kind=data.kind,
                configuration=config.__dict__.copy()
                if config.shape is None
                else {**config.__dict__, "shape": config.shape.to_dict()},
                raw_training_sha256=array_hash(data.values[fitted.training_indices]),
            )
        )
        return fitted

    def once(inputs, units, config, seed, draw, *, fold_minimum=None):
        if not records:
            for i, data in enumerate(inputs):
                with (directory / f"raw_{i}.npy").open("xb") as f:
                    np.save(f, data.values, allow_pickle=False)
            write_json(
                dict(
                    unit_ids=units,
                    views=[
                        dict(
                            name=d.name,
                            kind=d.kind,
                            unit_ids=d.unit_ids,
                            target_ids=d.target_ids,
                            observation_ids=d.observation_ids,
                            feature_names=d.feature_names,
                            feature_types=d.feature_types,
                        )
                        for d in inputs
                    ],
                ),
                directory / "raw-layout.json",
            )
        record = dict(
            index=len(records),
            seed=int(seed),
            draw=np.asarray(draw).tolist(),
            status="failed",
            error=None,
            candidate_start=len(candidates),
            views=[],
            attempted_views=[],
            configuration=config.to_dict(),
        )
        records.append(record)
        arrays = {}
        state.update(record=record, arrays=arrays)
        try:
            fitted = native_once(inputs, units, config, seed, draw, fold_minimum=fold_minimum)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            projected = integrative.project_integrative(inputs, units, fitted)
            probabilities = fitted.mixture.mixture.predict_proba(projected)
            arrays.update(
                fused=fitted.fused,
                scores=fitted.scores,
                components=fitted.pca.components_,
                pca_mean=fitted.pca.mean_,
                variance=fitted.pca.explained_variance_,
                projected=projected,
                projected_membership=probabilities,
                projected_labels=probabilities.argmax(1),
                component_order=np.asarray(fitted.mixture.component_order),
            )
            record.update(
                status="complete",
                selected_k=fitted.mixture.selected_k,
                candidates=fitted.mixture.candidates,
            )
            return fitted
        finally:
            record["candidate_stop"] = len(candidates)
            record["arrays"] = f"event_{record['index']:03d}.npz"
            with (directory / record["arrays"]).open("xb") as f:
                np.savez(f, **arrays)

    try:
        with (
            patch.object(integrative, "_fit_view", view),
            patch.object(integrative, "_fit_once", once),
        ):
            yield records
    finally:
        write_json(records, directory / "events.json")
