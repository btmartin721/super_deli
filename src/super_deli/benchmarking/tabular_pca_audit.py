"""Independent PCA moments, training identities and group-profile projections."""

import io
import json


def _pca_moments(matrix, parameters, scaling):
    import numpy as np

    keep = np.flatnonzero(matrix.var(0) > 0)
    np.testing.assert_array_equal(parameters["kept_indices"], keep)
    expected_mean = matrix[:, keep].mean(0)
    scale = matrix[:, keep].std(0) if scaling in {None, "standard"} else np.ones(len(keep))
    np.testing.assert_allclose(parameters["feature_means"], expected_mean, atol=1e-11)
    np.testing.assert_allclose(parameters["feature_scales"], scale, atol=1e-11)
    standardized = (matrix[:, keep] - expected_mean) / scale
    np.testing.assert_allclose(parameters["mean"], standardized.mean(0), atol=1e-12)
    centered = standardized - standardized.mean(0)
    covariance = centered.T @ centered / (len(matrix) - 1)
    components = np.asarray(parameters["components"])
    values = np.linalg.eigvalsh(covariance)[::-1][: len(components)]
    np.testing.assert_allclose(components @ components.T, np.eye(len(components)), atol=1e-11)
    np.testing.assert_allclose(covariance @ components.T, components.T * values, atol=1e-10)
    np.testing.assert_allclose(parameters["explained_variance"], values, atol=1e-10)


def _target_parameters(diagnostic, means, counts, units, draw, view):

    if (
        diagnostic["pca_fit"] != "target_means"
        or diagnostic["pca_training_target_ids"] != [units[i] for i in draw]
        or diagnostic["pca_training_counts"] != counts[draw].tolist()
    ):
        raise ValueError("Target-mean PCA training identities/counts/multiplicities disagree")
    parameters = {
        key: diagnostic[value]
        for key, value in {
            "kept_indices": "pca_kept_indices",
            "feature_means": "pca_feature_means",
            "feature_scales": "pca_feature_scales",
            "components": "pca_components",
            "mean": "pca_center",
            "explained_variance": "pca_explained_variance",
        }.items()
    }
    _pca_moments(means[draw], parameters, view.scaling)
    return parameters


def audit_tabular_views(archive, prefix, model, values, layout):
    """Reconstruct both continuous views from raw rows without invoking a fit."""
    import numpy as np

    units = list(layout["unit_ids"])
    order = {u: i for i, u in enumerate(units)}
    projections = {}
    for j, (view, design) in enumerate(zip(model.views, layout["views"], strict=True)):
        path = prefix + f"view_{j:03d}/"
        source = json.loads(archive.read(path + "source_rows.json"))
        for key in ("unit_ids", "observation_ids", "target_ids", "feature_names"):
            if source[key] != list(design[key]):
                raise ValueError("Source identities/features disagree")
        raw = np.load(io.BytesIO(archive.read(path + "source_values.npy")), allow_pickle=False)
        np.testing.assert_array_equal(raw, values[view.kind])
        target = np.array([order[u] for u in design["target_ids"]])
        counts = np.bincount(target, minlength=len(units))
        means = np.stack([raw[target == i].mean(0) for i in range(len(units))])
        matrix = means if view.pca_fit == "target_means" else raw
        with np.load(
            io.BytesIO(archive.read(path + "parameters/pca.npz")), allow_pickle=False
        ) as pca:
            parameters = {k: pca[k] for k in pca.files}
        _pca_moments(matrix, parameters, view.scaling)
        summary = json.loads(archive.read(path + "parameters/summary.json"))
        diagnostic = summary["diagnostics"]
        expected = np.concatenate([np.flatnonzero(target == i) for i in range(len(units))])
        np.testing.assert_array_equal(summary["training_source_indices"], expected)
        if view.pca_fit == "target_means":
            other = _target_parameters(
                diagnostic, means, counts, units, np.arange(len(units)), view
            )
            for key, value in other.items():
                np.testing.assert_allclose(value, parameters[key], atol=1e-12)
        scores = (
            (raw[:, parameters["kept_indices"]] - parameters["feature_means"])
            / parameters["feature_scales"]
            - parameters["mean"]
        ) @ parameters["components"].T
        profiles = np.stack([scores[target == i].mean(0) for i in range(len(units))])
        with np.load(
            io.BytesIO(archive.read(path + "parameters/profiles.npz")), allow_pickle=False
        ) as saved:
            np.testing.assert_allclose(saved["profiles"], profiles, atol=1e-11)
        projections[view.kind] = (raw, target, counts, means, scores, profiles)
    return projections
