"""Reconstruct fixed native transforms and quantify post-fit signal projection."""

import json
from pathlib import Path

import numpy as np

from super_deli.benchmarking.signal_loss_metrics import costs, geometry, projected_energy


class LocalReader:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, name):
        p = self.root / Path(name)
        if not p.resolve(strict=True).is_relative_to(self.root):
            raise ValueError("Escaping local context")
        return p

    def json(self, name):
        return json.loads(self.path(name).read_text())

    def arrays(self, name):
        value = np.load(self.path(name), allow_pickle=False)
        if isinstance(value, np.ndarray):
            return value.copy()
        with value:
            return {k: value[k].copy() for k in value.files}


def _order(actual, wanted):
    if len(actual) != len(set(actual)) or set(actual) != set(wanted):
        raise ValueError("Point cohort mismatch")
    lookup = {u: i for i, u in enumerate(actual)}
    return [lookup[u] for u in wanted]


def read_context(reader, combined, genomic, shape, truth):
    from super_deli.io.shape_model import read_shape_model
    from super_deli.models.shape_pca import project_shape_pca

    combined, genomic, shape = map(Path, (combined, genomic, shape))
    s = reader.json(combined / "summary.json")
    units = s["unit_ids"]
    if s["status"] != "complete" or s["covariance_type"] != "tied" or s["n_pcs"] != 2:
        raise ValueError("Declared complete shared-family baseline required")
    labels = reader.json(truth)
    lookup = dict(zip(labels["unit_ids"], labels["labels"], strict=True))
    if set(lookup) != set(units):
        raise ValueError("Evaluation cohort mismatch")
    y = np.asarray([lookup[u] for u in units], dtype=int)
    fusion = reader.arrays(combined / "fusion.npz")
    point = reader.arrays(combined / "mixture.npz")
    views = {}
    blocks = []
    slices = {}
    start = 0
    for i, view in enumerate(s["views"]):
        path = combined / f"view_{i:03d}"
        source = reader.json(path / "source_rows.json")
        raw = reader.arrays(path / "source_values.npy")
        order = _order(source["unit_ids"], units)
        params = path / "parameters"
        p = reader.arrays(params / "profiles.npz")
        profiles = p["profiles"]
        scale = float(p["scale"])
        weight = float(p["normalized_weight"])
        np.testing.assert_allclose(profiles.mean(0), p["mean"], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            np.sum((profiles - p["mean"]) ** 2) / (len(units) - 1), scale**2, rtol=1e-9
        )
        if view["kind"] == "morphometric":
            reader.path(params / "model.json")
            reader.path(params / "model.npz")
            model, _, _ = read_shape_model(reader.path(params / "model.json").parent)
            projection = project_shape_pca(raw, model=model)
            x = projection.tangent_coordinates.reshape(len(raw), -1)
            pc = projection.scores
            components = model.components
        else:
            pre = reader.arrays(params / "pca.npz")
            indices = pre["kept_indices"]
            mean = pre["feature_means"]
            scales = pre["feature_scales"]
            if len(mean) == raw.shape[1]:
                mean = mean[indices]
                scales = scales[indices]
            x = (raw[:, indices] - mean) / scales
            components = pre["components"]
            pc = (x - pre["mean"]) @ components.T
        x, pc = x[order], pc[order]
        np.testing.assert_allclose(pc, profiles, rtol=1e-7, atol=1e-9)
        block = (profiles - p["mean"]) * np.sqrt(weight) / scale
        np.testing.assert_allclose(np.sum(block**2) / (len(units) - 1), weight, rtol=1e-9)
        slices[view["modality"]] = (start, start + block.shape[1])
        start += block.shape[1]
        blocks.append(block)
        views[view["modality"]] = dict(
            raw_input=x,
            profiles=profiles,
            block=block,
            raw_geometry=geometry(x, y),
            pc_geometry=geometry(pc, y),
            view_pc_centroid_energy_retained=projected_energy(x, y, components),
            normalized_weight=weight,
        )
    fused = np.concatenate(blocks, axis=1)
    np.testing.assert_array_equal(fused, fusion["fused"])
    scores = (fused - fusion["mean"]) @ fusion["components"].T
    np.testing.assert_allclose(scores, point["scores"], rtol=1e-9, atol=1e-10)
    retention = {}
    for name, (a, b) in slices.items():
        isolated = np.zeros_like(fused)
        isolated[:, a:b] = fused[:, a:b]
        retention[name] = projected_energy(isolated, y, fusion["components"])
    single = {}
    for name, path in (("genomic", genomic), ("shape", shape)):
        native = reader.json(path / "summary.json")
        ids = native["specimen_ids"] if name == "genomic" else native["unit_ids"]
        order = _order(ids, units)
        arr = (
            dict(
                scores=reader.arrays(path / "pca_scores.npy"),
                labels=reader.arrays(path / "labels.npy"),
                membership=reader.arrays(path / "membership.npy"),
            )
            if name == "genomic"
            else reader.arrays(path / "mixture.npz")
        )
        single[name] = dict(
            summary=native,
            scores=arr["scores"][order],
            labels=arr["labels"][order],
            membership=arr["membership"][order],
        )
    candidates = reader.json(combined / "candidates.json")
    return dict(
        unit_ids=units,
        labels=y,
        views=views,
        slices=slices,
        fused=fused,
        scores=point["scores"].copy(),
        components=fusion["components"],
        summary=s,
        point=point,
        single=single,
        joint_geometry=geometry(scores, y),
        joint_view_centroid_energy_retained=retention,
        costs=costs(candidates, len(units), 2, "tied"),
    )


def public_trace(context):
    return dict(
        joint_geometry=context["joint_geometry"],
        joint_view_centroid_energy_retained=context["joint_view_centroid_energy_retained"],
        views={
            name: {k: v for k, v in value.items() if k not in {"raw_input", "profiles", "block"}}
            for name, value in context["views"].items()
        },
        costs=context["costs"],
        interpretation="Known simulated label geometry is evaluated after fitting, never optimized",
    )
