"""Frozen-coordinate comparisons; no neural training or biological label input."""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from super_deli.benchmarking.neural_analysis import arrays
from super_deli.benchmarking.neural_moons_config import DENSITY_ARMS
from super_deli.models.tabular_neural import forward_numpy


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Archive path escapes its source root")
    return path


def standardize(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 10 or not np.isfinite(x).all():
        raise ValueError("Complete finite coordinate matrices with at least ten rows required")
    scale = x.std(0)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Nonconstant coordinate axes required")
    return (x - x.mean(0)) / scale


def load_context(root, entry):
    """Evaluate final saved weights; preserve native coordinate bits for replay."""
    ref = arrays(safe_path(root, entry["reference"]))
    x, z, truth = ref["x"], ref["z"], ref["truth"]
    ids = np.asarray(entry["indices"])
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or len(set(ids)) != len(ids)
        or np.any(ids < 0)
        or np.any(ids >= len(x))
    ):
        raise ValueError("Invalid source identities or training indices")
    x, z, truth = x[ids], z[ids], truth[ids]
    if x.shape != (len(ids), 8) or z.shape != (len(ids), 2) or truth.shape != (len(ids),):
        raise ValueError("Reference dimensions do not match the declared cohort")
    context = {"x": x, "z": z, "truth": truth, "indices": ids}
    context["source_standard"] = standardize(z)
    context["traits_standard"] = standardize(x)
    diagnostics = {}
    for arm in DENSITY_ARMS:
        a = arrays(safe_path(root, entry["points"][arm]))
        if (
            a["scores"].shape != (len(ids), 2)
            or a["labels"].shape != (len(ids),)
            or a["strength"].shape != (len(ids),)
        ):
            raise ValueError("Invalid archived density array dimensions")
        np.testing.assert_array_equal(a["kept_indices"], np.arange(8))
        np.testing.assert_allclose(a["means"], x.mean(0), atol=1e-12)
        np.testing.assert_allclose(a["scales"], x.std(0), atol=1e-12)
        if np.any(a["scales"] <= 0):
            raise ValueError("Invalid raw scaling")
        if arm == "density":
            reconstructed = (context["traits_standard"] - a["pca_mean"]) @ a["components"].T
            np.testing.assert_allclose(a["scores"], reconstructed, atol=1e-11)
            context["pca_native"] = a["scores"].copy()
        else:
            for side, widths in (("encoder", [8, 16, 8, 2]), ("decoder", [2, 8, 16, 8])):
                for i, (left, right) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
                    for kind, shape in (("weight", (right, left)), ("bias", (right,))):
                        w = a[f"{side}_{i}_{kind}"]
                        if w.shape != shape or not np.isfinite(w).all():
                            raise ValueError("Invalid archived neural weights")
            raw = forward_numpy(context["traits_standard"], a, "encoder")
            if (
                a["latent_means"].shape != (2,)
                or a["latent_scales"].shape != (2,)
                or not np.isfinite(a["latent_scales"]).all()
                or np.any(a["latent_scales"] <= 0)
            ):
                raise ValueError("Invalid latent normalization")
            np.testing.assert_allclose(raw.mean(0), a["latent_means"], atol=1e-12)
            np.testing.assert_allclose(raw.std(0), a["latent_scales"], atol=1e-12)
            np.testing.assert_allclose(
                a["scores"], (raw - a["latent_means"]) / a["latent_scales"], atol=1e-11
            )
            residual = (forward_numpy(raw, a, "decoder") - context["traits_standard"]) ** 2
            np.testing.assert_allclose(residual.mean(), a["losses"][-1], atol=1e-12)
            diagnostics[arm] = {
                "raw_latent_scales": a["latent_scales"].tolist(),
                "raw_latent_singular_values": np.linalg.svd(
                    raw - raw.mean(0), compute_uv=False
                ).tolist(),
                "mse_by_feature": residual.mean(0).tolist(),
                "mse_by_generating_group": {
                    str(int(k)): float(residual[truth == k].mean()) for k in set(truth)
                },
                "training_losses": a["losses"].tolist(),
            }
            context[arm + "_raw"] = raw
            context[arm + "_standard"] = a["scores"].copy()
        context[arm + "_labels"] = a["labels"].copy()
        context[arm + "_strength"] = a["strength"].copy()
    return context, diagnostics


def neighborhood_metrics(scores, context):
    """Prespecified four-neighbor overlap and distance-rank agreement."""
    distances = pdist(scores)
    matrix = squareform(distances)
    np.fill_diagonal(matrix, np.inf)
    neighbors = np.argsort(matrix, axis=1, kind="stable")[:, :4]
    neighbor_arrays = {"neighbors": neighbors}
    result = {
        "cross_neighbor_fraction": float(
            np.mean(context["truth"][neighbors] != context["truth"][:, None])
        )
    }
    for name in ("source_standard", "traits_standard"):
        other = pdist(context[name])
        full = squareform(other)
        np.fill_diagonal(full, np.inf)
        reference = np.argsort(full, axis=1, kind="stable")[:, :4]
        neighbor_arrays[name + "_neighbors"] = reference
        overlap = [len(set(a) & set(b)) / 4 for a, b in zip(neighbors, reference, strict=True)]
        result[name + "_neighbor_overlap"] = float(np.mean(overlap))
        a, b = rankdata(distances), rankdata(other)
        result[name + "_distance_rank_correlation"] = float(np.corrcoef(a, b)[0, 1])
    # Count tolerance ties at the neighborhood cutoff; do not hide ID tie-breaking.
    boundary = np.take_along_axis(matrix, neighbors[:, -1:], axis=1)
    result["cutoff_tie_rows"] = int(
        np.sum(np.sum(np.isclose(matrix, boundary, rtol=1e-12, atol=1e-12), axis=1) > 1)
    )
    result["distance_tie_excess"] = int(len(distances) - len(np.unique(distances)))
    return result, neighbor_arrays
