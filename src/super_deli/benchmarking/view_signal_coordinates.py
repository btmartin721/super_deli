"""Replay archived transforms without training, fitting PCA or using group labels."""

import numpy as np

from super_deli.benchmarking.matched_reference import audit_integrative, audit_pca, balanced_inputs
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_coordinates import safe_path, standardize
from super_deli.benchmarking.view_signal_config import REPRESENTATIONS, SEEDS
from super_deli.io.config import read_json
from super_deli.models.tabular_neural import forward_numpy


def native_arm(name):
    if name in ("morphology_density_pcs", "ecology_density_pcs"):
        return name.removesuffix("_pcs")
    if name.endswith("_standard") and name.split("_")[0] in {"paired", "morphology", "ecology"}:
        arm = name.removesuffix("_standard")
        if arm.rsplit("_", 1)[-1] in {str(s) for s in SEEDS}:
            return arm
    return None


def validate_context(context):
    n = len(context["truth"])
    require(context["truth"].shape == (n,) and n >= 32, "Four-group source size")
    require(set(context["truth"]) == {0, 1, 2, 3}, "Four evaluation groups required")
    require(len(set(context["unit_ids"].tolist())) == n, "Unique canonical individuals")
    for name in REPRESENTATIONS:
        value = context[name]
        require(value.ndim == 2 and value.shape[0] == n, "Coordinate identity/shape")
        require(value.shape[1] in {2, 4, 8, 12} and np.isfinite(value).all(), "Finite spaces")
        if native_arm(name):
            require(context[name + "_labels"].shape == (n,), "Native label coverage")
            require(context[name + "_strength"].shape == (n,), "Native strength coverage")
    return n


def load_context(source, entry):
    folder = safe_path(source, entry["path"])
    if entry.get("fixture"):
        context = arrays(folder / "context.npz")
        validate_context(context)
        return context, {}
    ref = arrays(folder / "generating-reference.npz")
    xm, xe = ref["morphology"], ref["ecology"]
    n = len(xm)
    context = {
        "morphology": xm,
        "ecology": xe,
        "latent_morphology": ref["latent_morphology"],
        "latent_ecology": ref["latent_ecology"],
        "truth": ref["truth"],
        "unit_ids": np.array([f"unit_{i:04d}" for i in range(n)]),
        "morphology_standard": standardize(xm),
        "ecology_standard": standardize(xe),
    }
    diagnostics = {}
    for view in ("paired", "morphology", "ecology"):
        x = np.column_stack((xm, xe)) if view == "paired" else ref[view]
        for seed in SEEDS:
            arm = f"{view}_{seed}"
            point = folder / arm
            a = arrays(point / "point.npz")
            require(
                read_json(point / "summary.json")["unit_ids"] == context["unit_ids"].tolist(),
                "Native individual alignment",
            )
            z = balanced_inputs(x, a, view == "paired")
            raw = forward_numpy(z, a, "encoder")
            require(np.all(a["latent_scales"] > 0), "Positive latent scales")
            np.testing.assert_allclose(raw.mean(0), a["latent_means"], atol=1e-11)
            np.testing.assert_allclose(raw.std(0), a["latent_scales"], atol=1e-11)
            np.testing.assert_allclose(
                (raw - a["latent_means"]) / a["latent_scales"], a["scores"], atol=1e-10
            )
            name = arm + "_standard"
            context[name] = a["scores"].copy()
            context[name + "_labels"] = a["labels"].copy()
            context[name + "_strength"] = a["strength"].copy()
            if view == "paired":
                context[arm + "_raw"] = raw
                if seed == 0:
                    context["balanced_traits"] = z
                else:
                    np.testing.assert_array_equal(context["balanced_traits"], z)
            reconstruction = forward_numpy(raw, a, "decoder")
            residual = reconstruction - z
            np.testing.assert_allclose(np.mean(residual**2), a["losses"][-1], atol=1e-11)
            weights = a["column_weights"] if view == "paired" else np.ones(z.shape[1])
            standardized_reconstruction = reconstruction / weights
            standardized_input = z / weights
            by_view = (slice(0, 4), slice(4, 12)) if view == "paired" else (slice(None),)
            diagnostics[arm] = {
                "latent_scales": a["latent_scales"].tolist(),
                "final_weighted_mse": float(np.mean(residual**2)),
                "standardized_mse_by_view": [
                    float(np.mean(((standardized_reconstruction - standardized_input)[:, s]) ** 2))
                    for s in by_view
                ],
                "centroid_mse_by_group_view": [
                    [
                        float(
                            np.mean(
                                (
                                    standardized_reconstruction[ref["truth"] == g, s].mean(0)
                                    - standardized_input[ref["truth"] == g, s].mean(0)
                                )
                                ** 2
                            )
                        )
                        for s in by_view
                    ]
                    for g in range(4)
                ],
                "trained_now": False,
            }
    root = folder / "integrative_gmm"
    joint = audit_integrative(xm, xe, root)
    for i, view in enumerate(("morphology", "ecology")):
        a = arrays(root / f"view_{i:03d}" / "parameters" / "profiles.npz")
        context[view + "_view_pcs"] = a["profiles"]
    context["fused_profiles"] = arrays(root / "fusion.npz")["fused"]
    context["joint_pcs"] = arrays(root / "mixture.npz")["scores"]
    np.testing.assert_allclose(context["joint_pcs"], joint, atol=1e-10)
    for view, x in (("morphology", xm), ("ecology", xe)):
        a = arrays(folder / (view + "_density") / "point.npz")
        audit_pca(x, a, a["scores"])
        name = view + "_density_pcs"
        context[name] = a["scores"]
        for key in ("labels", "strength"):
            context[name + "_" + key] = a[key]
    validate_context(context)
    return context, diagnostics
