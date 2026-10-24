"""Identity-preserving raw subsets and frozen exported PCA projection."""

from __future__ import annotations

import copy
from pathlib import Path

from super_deli.io.config import read_json


def load_sources(dataset, model, run):
    import numpy as np

    units = read_json(Path(run) / "summary.json")["unit_ids"]
    declared = dataset.populations if model.target_level == "population" else dataset.localities
    if set(units) != set(declared):
        raise ValueError("Every declared target must be retained in both complete views")
    layout = {
        "schema_version": 1,
        "unit_ids": units,
        "target_level": model.target_level,
        "specimens": [],
        "views": [],
    }
    values = {}
    modalities = {m.name: m for m in dataset.modalities}
    for i, view in enumerate(model.views):
        rows = read_json(Path(run) / f"view_{i:03d}/source_rows.json")
        array = np.load(Path(run) / f"view_{i:03d}/source_values.npy", allow_pickle=False)
        level = "specimen" if view.kind == "morphological" else model.target_level
        if (
            rows["source_level"] != level
            or any(t != "continuous" for t in rows["feature_types"])
            or not np.isfinite(array).all()
            or set(rows["target_ids"]) != set(units)
        ):
            raise ValueError(
                "Complete continuous specimen morphology and one ecology row/target required"
            )
        if view.kind == "ecological" and len(rows["target_ids"]) != len(units):
            raise ValueError("Ecology must occur exactly once per target")
        features = {f.name: f for f in modalities[view.modality].features}
        layout["views"].append(
            {
                "modality": view.modality,
                "kind": view.kind,
                "level": level,
                **{
                    k: rows[k]
                    for k in ("observation_ids", "unit_ids", "target_ids", "feature_names")
                },
                "feature_units": [features[n].unit for n in rows["feature_names"]],
            }
        )
        values[view.kind] = array
        if view.kind == "morphological":
            if set(rows["unit_ids"]) != {s.specimen_id for s in dataset.specimens}:
                raise ValueError("Every declared morphology specimen requires exactly one row")
            layout["specimens"] = [
                {"specimen_id": s, f"{model.target_level}_id": t}
                for s, t in zip(rows["unit_ids"], rows["target_ids"], strict=True)
            ]
            layout["counts"] = [rows["target_ids"].count(u) for u in units]
    return values, layout


def subset_raw(values, layout, indices):
    import numpy as np

    units = [layout["unit_ids"][i] for i in sorted(indices)]
    if not units or len(set(units)) != len(units):
        raise ValueError("Select nonempty distinct target units")
    selected = set(units)
    result = copy.deepcopy(layout)
    result["unit_ids"] = units
    result["counts"] = [layout["counts"][i] for i in sorted(indices)]
    result["specimens"] = [
        s for s in layout["specimens"] if s[f"{layout['target_level']}_id"] in selected
    ]
    arrays = {}
    for view in result["views"]:
        rows = np.array([i for i, u in enumerate(view["target_ids"]) if u in selected], dtype=int)
        arrays[view["kind"]] = values[view["kind"]][rows].copy()
        for key in ("observation_ids", "unit_ids", "target_ids"):
            view[key] = [view[key][i] for i in rows]
    return arrays, result


def project_exported(run, values, layout):
    import numpy as np

    blocks = []
    for i, view in enumerate(layout["views"]):
        path = Path(run) / f"view_{i:03d}/parameters"
        with np.load(path / "pca.npz", allow_pickle=False) as p:
            raw = values[view["kind"]]
            scores = (
                ((raw[:, p["kept_indices"]] - p["feature_means"]) / p["feature_scales"]) - p["mean"]
            ) @ p["components"].T
        targets = np.asarray(view["target_ids"])
        profiles = np.stack([scores[targets == u].mean(0) for u in layout["unit_ids"]])
        with np.load(path / "profiles.npz", allow_pickle=False) as p:
            blocks.append((profiles - p["mean"]) * np.sqrt(p["normalized_weight"]) / p["scale"])
    with np.load(Path(run) / "fusion.npz", allow_pickle=False) as p:
        return (np.concatenate(blocks, axis=1) - p["mean"]) @ p["components"].T
