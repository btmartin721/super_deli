"""Strict specimen joins and fitted view balance; no row-position or label joins."""

import numpy as np


def align_tables(tables, config, *, minimum=10):
    by_name = {t.modality_name: t for t in tables}
    if len(by_name) != len(tables) or set(by_name) != {config.morphology, config.ecology}:
        raise ValueError("Exactly the declared morphology and ecology tables required")
    matrices, manifest = [], {"views": []}
    canonical = None
    for name, kind, features in (
        (config.morphology, "morphological", config.morphology_features),
        (config.ecology, "ecological", config.ecology_features),
    ):
        table = by_name[name]
        if table.kind != kind or table.level != "specimen":
            raise ValueError(
                "Both selected views must be declared individual specimen measurements"
            )
        n = len(table.values)
        for ids in (table.unit_ids, table.observation_ids):
            if (
                len(ids) != n
                or any(not isinstance(s, str) or not s or s != s.strip() for s in ids)
                or len(set(ids)) != n
            ):
                raise ValueError("One unique observation and biological unit per row required")
        units = sorted(table.unit_ids)
        if canonical is None:
            canonical = units
        if units != canonical:
            raise ValueError(
                "Selected views must have exactly the same unit IDs; no intersection or broadcast"
            )
        mapping = {f.name: (i, f) for i, f in enumerate(table.features)}
        if len(mapping) != len(table.features) or set(features) - mapping.keys():
            raise ValueError("Selected features must occur uniquely in their declared view")
        columns = [mapping[f][0] for f in features]
        if any(
            mapping[f][1].dtype != "continuous" or mapping[f][1].role != "feature" for f in features
        ):
            raise ValueError("Only declared continuous analytical features are allowed")
        if any(len(row) != len(table.features) for row in table.values):
            raise ValueError("Row width does not match the feature schema")
        index = {unit: i for i, unit in enumerate(table.unit_ids)}
        order = [index[u] for u in canonical]
        values = [[table.values[i][j] for j in columns] for i in order]
        if any(type(v) not in (int, float) for row in values for v in row):
            raise ValueError("Every selected cell must be a finite observed real number")
        x = np.array(values, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite selected measurements")
        matrices.append(x)
        manifest["views"].append(
            {
                "modality": name,
                "kind": kind,
                "feature_names": list(features),
                "source_row_indices": order,
                "observation_ids": [table.observation_ids[i] for i in order],
            }
        )
    if len(canonical) < minimum:
        raise ValueError(f"At least {minimum} matched individuals required")
    x = np.column_stack(matrices)
    if 64 * len(x) ** 2 + 160 * x.size > config.max_working_bytes:
        raise ValueError("Matched training/projection workspace exceeds budget")
    manifest["unit_ids"] = canonical
    return x, manifest


def prepare_views(x, split):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        means, scales = x.mean(0), x.std(0)
        kept = np.flatnonzero(scales > 0)
        view = np.arange(x.shape[1]) >= split
        counts = np.array([sum(~view[kept]), sum(view[kept])])
        if np.any(counts < 3) or np.any(counts > 32):
            raise ValueError("Each fitted view must retain 3–32 nonconstant traits")
        weights = np.sqrt(len(kept) / (2 * counts))
        column_weights = weights[view[kept].astype(int)]
        scales[scales == 0] = 1
        standardized = (x[:, kept] - means[kept]) / scales[kept]
        weighted = standardized * column_weights
    if not np.isfinite(weighted).all():
        raise ValueError("Nonfinite view-balanced inputs")
    return weighted, {
        "means": means,
        "scales": scales,
        "kept_indices": kept,
        "view_weights": weights,
        "column_weights": column_weights,
        "retained_view_counts": counts,
        "selected_view_counts": np.array([split, x.shape[1] - split]),
    }
