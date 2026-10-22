"""Identity-aware raw inputs and simulation for the fitted grouped Gaussian null."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.models.grouped_gaussian import fit_gaussian, sufficient_statistics


@dataclass(frozen=True)
class GroupedGaussian:
    values: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    layout: dict
    within: np.ndarray
    morphology: np.ndarray
    ecology: np.ndarray
    targets: np.ndarray
    counts: np.ndarray
    sampling_factor: np.ndarray | None = None
    within_sampling_factor: np.ndarray | None = None

    def model_arrays(self):
        arrays = {
            name: getattr(self, name)
            for name in (
                "values",
                "mean",
                "covariance",
                "within",
                "morphology",
                "ecology",
                "targets",
                "counts",
            )
        }

        if self.sampling_factor is not None:
            arrays["sampling_factor"] = self.sampling_factor
        if self.within_sampling_factor is not None:
            arrays["within_sampling_factor"] = self.within_sampling_factor
        return arrays


def fit_null(dataset, tables, config):
    from super_deli.models.tabular_gmm import _matrix
    from super_deli.workflows.integrative import _target_ids

    by_name = {t.modality_name: t for t in tables}
    blocks, views, units = {}, [], None
    expected_groups = set(
        getattr(
            dataset, "populations" if config.model.target_level == "population" else "localities"
        )
    )
    for view in config.model.views:
        table = by_name[view.modality]
        expected_level = "specimen" if view.kind == "morphological" else config.model.target_level
        if table.kind != view.kind or table.level != expected_level:
            raise ValueError(
                "Use specimen morphology and one directly measured ecology row per group"
            )
        values, names, types = _matrix(table, view.tabular_config(config.model.max_working_bytes))
        if any(t != "continuous" for t in types) or not np.isfinite(values).all():
            raise ValueError(
                "Select complete finite continuous features; counts/missingness unsupported"
            )
        if view.kind == "morphological" and set(table.unit_ids) != {
            s.specimen_id for s in dataset.specimens
        }:
            raise ValueError("Every declared specimen must have exactly one morphology observation")
        targets = _target_ids(dataset, table.level, table.unit_ids, view, config.model.target_level)
        if set(targets) != expected_groups:
            raise ValueError(
                "Every declared group must have morphology and exactly one ecology row"
            )
        if view.kind == "ecological" and len(targets) != len(set(targets)):
            raise ValueError("Ecology must be observed exactly once per group")
        if units is None:
            units = tuple(dict.fromkeys(targets))
        features = {f.name: f for f in table.features}
        views.append(
            {
                "modality": view.modality,
                "kind": view.kind,
                "level": table.level,
                "feature_names": names,
                "feature_units": [features[n].unit for n in names],
                "observation_ids": table.observation_ids,
                "unit_ids": table.unit_ids,
                "target_ids": targets,
            }
        )
        blocks[view.kind] = (values, targets)
    order = {u: i for i, u in enumerate(units)}
    morphology, target_ids = blocks["morphological"]
    targets = np.array([order[u] for u in target_ids], dtype=np.int64)
    eco, eco_ids = blocks["ecological"]
    eco_order = {u: i for i, u in enumerate(eco_ids)}
    ecology = eco[[eco_order[u] for u in units]]
    p, q, n, g = morphology.shape[1], ecology.shape[1], len(morphology), len(ecology)
    if 8 * (32 * n * p + 32 * g * (p + q) + 48 * (p + q) ** 2) > config.model.max_working_bytes:
        raise ValueError("Grouped Gaussian fit exceeds max_working_bytes")
    fitter = fit_gaussian
    if config.schema_version == 2:
        from super_deli.models.grouped_conditional import fit_gaussian as fitter
    mean, between, within, diagnostics = fitter(morphology, ecology, targets, config)
    sampling_factor = diagnostics.pop("sampling_factor", None)
    within_sampling_factor = diagnostics.pop("within_sampling_factor", None)
    values, counts, _ = sufficient_statistics(morphology, ecology, targets)
    layout = {
        "schema_version": config.schema_version,
        "unit_ids": units,
        "target_level": config.model.target_level,
        "views": views,
        "counts": counts.tolist(),
        "specimens": [
            {
                "specimen_id": s.specimen_id,
                f"{config.model.target_level}_id": getattr(s, f"{config.model.target_level}_id"),
            }
            for s in dataset.specimens
        ],
        "fit": (
            "Marginal maximum likelihood: joint between-group Gaussian "
            "and shared within-morphology Gaussian, before PCA"
        ),
        "sampling": (
            "Independent groups; iid morphology residuals independent of "
            "group effects; fixed noninformative observed counts; one ecology row/group"
        ),
        "parameter_order": "Morphology features then ecology features, independent of view order",
        "diagnostics": diagnostics,
    }
    return GroupedGaussian(
        values,
        mean,
        between,
        layout,
        within,
        morphology,
        ecology,
        targets,
        counts,
        None if sampling_factor is None else np.asarray(sampling_factor),
        None if within_sampling_factor is None else np.asarray(within_sampling_factor),
    )


def simulate_null(null, seed):
    rng = np.random.default_rng(seed)
    factor = (
        np.linalg.cholesky(null.covariance)
        if null.sampling_factor is None
        else null.sampling_factor
    )
    latent = null.mean + rng.standard_normal(null.values.shape) @ factor.T
    p = null.morphology.shape[1]
    within_factor = (
        np.linalg.cholesky(null.within)
        if null.within_sampling_factor is None
        else null.within_sampling_factor
    )
    morphology = (
        latent[null.targets, :p] + rng.standard_normal(null.morphology.shape) @ within_factor.T
    )
    ecology = latent[:, p:]
    order = {u: i for i, u in enumerate(null.layout["unit_ids"])}
    eco_view = next(v for v in null.layout["views"] if v["kind"] == "ecological")
    return {
        "morphological": morphology,
        "ecological": ecology[[order[u] for u in eco_view["target_ids"]]],
    }


def write_raw_views(values, layout, directory):
    """Write one ecology row/group and each declared morphology specimen; no labels."""
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    modalities = []
    for i, view in enumerate(layout["views"]):
        names, array = view["feature_names"], values[view["kind"]]
        if array.shape != (len(view["unit_ids"]), len(names)) or not np.isfinite(array).all():
            raise ValueError("Raw grouped draw dimensions/values disagree with the layout")
        identity_column = "observation_id"
        while identity_column in names:
            identity_column = "_" + identity_column
        path = f"view_{i:03d}.csv"
        with (directory / path).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow((identity_column, *names))
            for observation, row in zip(view["observation_ids"], array, strict=True):
                writer.writerow((observation, *row))
        modalities.append(
            {
                "name": view["modality"],
                "kind": view["kind"],
                "level": view["level"],
                "format": "csv",
                "path": path,
                "tabular_options": {"observation_id_column": identity_column},
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
            "dataset_id": "grouped_continuous_raw_views",
            "specimens": layout["specimens"],
            "modalities": modalities,
            "populations" if layout["target_level"] == "population" else "localities": list(
                layout["unit_ids"]
            ),
        }
    )
    write_json(dataset.to_dict(), directory / "dataset.json")
    return dataset
