"""Complete continuous input validation and raw-trait standardization."""

import numpy as np

from super_deli.models.spectral_config import SpectralConfig
from super_deli.models.spectral_graph import fit_scores


def prepare(table, config):
    n, p = len(table.values), len(table.features)
    if table.kind not in {"morphological", "ecological"} or table.level not in {
        "specimen",
        "population",
        "locality",
    }:
        raise ValueError("One morphology/ecology table at a declared biological level required")
    for ids in (table.unit_ids, table.observation_ids):
        if (
            len(ids) != n
            or any(not isinstance(s, str) or not s or s != s.strip() for s in ids)
            or len(set(ids)) != n
        ):
            raise ValueError("Unique nonempty unit and observation identities required")
    if any(len(row) != p for row in table.values):
        raise ValueError("Rows do not match the feature schema")
    by_name = {f.name: i for i, f in enumerate(table.features)}
    if len(by_name) != p or set(config.feature_names) - by_name.keys():
        raise ValueError("Selected features must have unique declared names")
    columns = [by_name[s] for s in config.feature_names]
    if any(
        table.features[j].dtype != "continuous" or table.features[j].role != "feature"
        for j in columns
    ):
        raise ValueError("Only explicitly analytical continuous features are supported")
    if n < 32 or config.max_k >= n or 96 * n * n + 32 * n * len(columns) > config.max_working_bytes:
        raise ValueError("Too few units or graph workspace exceeds the declared budget")
    values = [[row[j] for j in columns] for row in table.values]
    if any(type(v) not in (int, float) for row in values for v in row):
        raise ValueError("Complete finite real values required; no missing-value filling")
    x = np.array(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Complete finite real values required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        means, scales = x.mean(0), x.std(0)
        if np.any(scales <= 0):
            raise ValueError(
                "Selected constant columns are unsupported; choose features explicitly"
            )
        scores = (x - means) / scales
    if not np.isfinite(scores).all():
        raise ValueError("Standardization produced nonfinite values")
    return {"raw_values": x, "means": means, "scales": scales, "scores": scores}


def fit_tabular_spectral(table, *, config):
    config = SpectralConfig.from_dict(config.to_dict())
    arrays = prepare(table, config)
    record, graph_arrays = fit_scores(arrays["scores"], table.unit_ids, config)
    arrays.update(graph_arrays)
    return record, arrays
