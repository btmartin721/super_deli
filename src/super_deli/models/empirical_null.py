"""A fitted full Gaussian over matched raw continuous features, before any PCA."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json


@dataclass(frozen=True)
class RawGaussian:
    values: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    layout: dict

    def model_arrays(self):
        return {"values": self.values, "mean": self.mean, "covariance": self.covariance}


def fit_null(dataset, tables, config):
    from super_deli.models.tabular_gmm import _matrix

    by_name = {t.modality_name: t for t in tables}
    blocks, layout, units = [], [], None
    expected = {s.specimen_id for s in dataset.specimens}
    for view in config.model.views:
        table = by_name[view.modality]
        if table.level != "specimen" or table.kind != view.kind:
            raise ValueError("Fitted null requires directly matched specimen-level tables")
        values, names, types = _matrix(table, view.tabular_config(config.model.max_working_bytes))
        if any(t != "continuous" for t in types) or not np.isfinite(values).all():
            raise ValueError(
                "Select complete finite continuous features; counts/missingness are unsupported"
            )
        if set(table.unit_ids) != expected:
            raise ValueError(
                "Every declared specimen must occur exactly once in each selected view"
            )
        if units is None:
            units = tuple(table.unit_ids)
        order = {u: i for i, u in enumerate(table.unit_ids)}
        blocks.append(values[[order[u] for u in units]])
        features = {f.name: f for f in table.features}
        layout.append(
            {
                "modality": view.modality,
                "kind": view.kind,
                "feature_names": names,
                "feature_units": [features[name].unit for name in names],
                "observation_ids": table.observation_ids,
                "unit_ids": table.unit_ids,
            }
        )
    n, p = len(units), sum(b.shape[1] for b in blocks)
    if n <= p or n < 2 * config.model.min_cluster_size:
        raise ValueError("Fitted raw covariance requires N>P and at least two feasible components")
    if 8 * (24 * n * p + 8 * p * p) > config.model.max_working_bytes:
        raise ValueError("Raw Gaussian fit exceeds max_working_bytes")
    values = np.concatenate(blocks, axis=1)
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / n
    scales = np.sqrt(np.diag(covariance))
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Fitted raw Gaussian covariance is rank deficient")
    eigenvalues = np.linalg.eigvalsh(covariance / scales[:, None] / scales[None, :])
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= config.rank_tol * max(
        eigenvalues[-1], 1.0
    ):
        raise ValueError(
            "Fitted raw Gaussian covariance is rank deficient; no regularization is applied"
        )
    return RawGaussian(
        values,
        mean,
        covariance,
        {
            "schema_version": 1,
            "unit_ids": units,
            "views": layout,
            "fit": "Joint raw mean and full maximum-likelihood covariance (divisor N), before PCA",
            "sampling": "Independent specimen rows; fitted cross-view covariance retained",
        },
    )


def simulate_null(null, seed):
    scales = np.sqrt(np.diag(null.covariance))
    factor = scales[:, None] * np.linalg.cholesky(
        null.covariance / scales[:, None] / scales[None, :]
    )
    return null.mean + np.random.default_rng(seed).standard_normal(null.values.shape) @ factor.T


def write_raw_views(values, layout, directory):
    """Write selected continuous features and identities; no truth enters this API."""
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    units = tuple(layout["unit_ids"])
    order = {u: i for i, u in enumerate(units)}
    if (
        values.shape != (len(units), sum(len(v["feature_names"]) for v in layout["views"]))
        or not np.isfinite(values).all()
    ):
        raise ValueError("Raw draw dimensions/values disagree with the frozen layout")
    modalities, offset = [], 0
    for i, view in enumerate(layout["views"]):
        names = view["feature_names"]
        path = f"view_{i:03d}.csv"
        with (directory / path).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("observation_id", *names))
            for observation, unit in zip(view["observation_ids"], view["unit_ids"], strict=True):
                writer.writerow((observation, *values[order[unit], offset : offset + len(names)]))
        offset += len(names)
        modalities.append(
            {
                "name": view["modality"],
                "kind": view["kind"],
                "level": "specimen",
                "format": "csv",
                "path": path,
                "observations": [
                    {"observation_id": o, "unit_id": u}
                    for o, u in zip(view["observation_ids"], view["unit_ids"], strict=True)
                ],
                "features": [
                    {"name": name, "dtype": "continuous", "unit": unit}
                    for name, unit in zip(names, view["feature_units"], strict=True)
                ],
            }
        )
    dataset = config_from_dict(
        {
            "schema_version": 1,
            "dataset_id": "continuous_raw_views",
            "specimens": [{"specimen_id": u} for u in units],
            "modalities": modalities,
        }
    )
    write_json(dataset.to_dict(), directory / "dataset.json")
    return dataset
