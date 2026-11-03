"""Separate-encoder paired workflow with strict, training-free numeric reload."""

import html
import zipfile
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json
from super_deli.models.separate_config import SeparateNeuralConfig


def _render(out, summary, point, units, **kwargs):
    from super_deli.models.separate_neural import ROLES, unpack_branch
    from super_deli.models.tabular_neural import NeuralModel, NeuralPoint
    from super_deli.workflows._neural_common import _report

    summary.update(
        n_coordinates=4,
        fusion_rank=point.diagnostics["fusion_rank"],
        branch_parameter_counts=point.diagnostics["branch_parameter_counts"],
        interpretation=[
            "four concatenated coordinates; no joint compression",
            "individual ecology and independence are declared assumptions",
            "equal view influence is not equal biological importance",
            "no new-unit cluster prediction, p-value or species probability",
        ],
    )
    for role, name, names in zip(
        ROLES,
        (point.model.config.morphology, point.model.config.ecology),
        (point.model.config.morphology_features, point.model.config.ecology_features),
        strict=True,
    ):
        a = unpack_branch(point.model.arrays, role)
        proxy = NeuralPoint(
            NeuralModel(names, a),
            a["scores"],
            point.labels,
            point.strength,
            a["losses"],
            point.diagnostics,
        )
        target = out if role == "morphology" else out / "ecology-view"
        if target != out:
            target.mkdir()
        link = (
            '<a href="ecology-view/report.html">Ecology branch plot</a>'
            if role == "morphology"
            else '<a href="../report.html">Morphology branch plot</a>'
        )
        detail = (
            "<p>Clustering uses all four concatenated coordinates. "
            f"This panel shows only the {html.escape(name)} branch, "
            f"colored by the joint assignments. {link}</p>"
            "<p>Each encoder performs 500 updates. No species significance is computed.</p>"
        )
        _report(
            target,
            summary,
            proxy,
            units,
            title="Separate encoders and four-coordinate HDBSCAN",
            view_details=detail,
        )


def run_separate_neural(dataset, *, model_config, base_dir, output_dir):
    from super_deli.models.separate_neural import SEPARATE_TRAINING, fit_separate_neural
    from super_deli.workflows._paired_neural_common import run_paired_neural

    return run_paired_neural(
        dataset,
        model_config=model_config,
        base_dir=base_dir,
        output_dir=output_dir,
        config_type=SeparateNeuralConfig,
        fit_function=fit_separate_neural,
        training=SEPARATE_TRAINING,
        render=_render,
    )


def _branch_shapes(a, p, n):
    import numpy as np

    ids = a["kept_indices"]
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or not 3 <= len(ids) <= min(p, 32)
        or not np.array_equal(ids, np.unique(ids))
        or np.any(ids < 0)
        or np.any(ids >= p)
    ):
        raise ValueError("Invalid fitted branch feature indices")
    shapes = {
        "means": (p,),
        "scales": (p,),
        "latent_means": (2,),
        "latent_scales": (2,),
        "scores": (n, 2),
        "losses": (501,),
        "latent_singular_values": (2,),
    }
    for side, widths in (("encoder", [len(ids), 16, 8, 2]), ("decoder", [2, 8, 16, len(ids)])):
        for i, (left, right) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            shapes[f"{side}_{i}_weight"] = (right, left)
            shapes[f"{side}_{i}_bias"] = (right,)
    if set(a) != set(shapes) | {"kept_indices"}:
        raise ValueError("Unexpected/missing branch arrays")
    for key, shape in shapes.items():
        if a[key].shape != shape or a[key].dtype.kind not in "if" or not np.isfinite(a[key]).all():
            raise ValueError("Invalid branch array: " + key)
    if np.any(a["scales"] <= 0) or np.any(a["latent_scales"] <= 0) or np.any(a["losses"] < 0):
        raise ValueError("Invalid branch scales/losses")
    singular = np.linalg.svd(a["scores"], compute_uv=False)
    if singular[-1] <= 1e-12 * singular[0]:
        raise ValueError("Collapsed saved branch")
    if not np.allclose(a["scores"].mean(0), 0, atol=1e-10) or not np.allclose(
        a["scores"].std(0), 1, atol=1e-10
    ):
        raise ValueError("Invalid fitted latent normalization")
    if not np.allclose(singular, a["latent_singular_values"], atol=1e-10, rtol=1e-10):
        raise ValueError("Invalid branch spectrum")


def load_separate_neural_model(directory):
    import numpy as np

    from super_deli.models.separate_neural import (
        ROLES,
        SeparateNeuralModel,
        fusion_spectrum,
        unpack_branch,
    )

    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    try:
        config = SeparateNeuralConfig.from_dict(summary["configuration"]["model"])
        if (
            summary["status"] != "complete"
            or summary["model"] != config.model_family
            or summary["n_coordinates"] != 4
        ):
            raise ValueError("Not a completed separate-encoder model")
        if (
            configuration_hash(summary["configuration"]) != summary["configuration_sha256"]
            or fingerprint_file(directory / "point.npz").sha256 != summary["model_sha256"]
        ):
            raise ValueError("Separate model/configuration fingerprint mismatch")
        with zipfile.ZipFile(directory / "point.npz") as archive:
            if sum(member.file_size for member in archive.infolist()) > config.max_working_bytes:
                raise ValueError("Numeric model exceeds its workspace")
        with np.load(directory / "point.npz", allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in archive.files}
        units = summary["unit_ids"]
        if (
            len(units) < 10
            or any(not isinstance(u, str) or not u or u != u.strip() for u in units)
            or units != sorted(set(units))
        ):
            raise ValueError("Invalid saved unit identities")
        n = len(units)
        branches = [unpack_branch(a, role) for role in ROLES]
        for branch, p in zip(
            branches, (len(config.morphology_features), len(config.ecology_features)), strict=True
        ):
            _branch_shapes(branch, p, n)
        expected_keys = {
            role + "__" + key for role, b in zip(ROLES, branches, strict=True) for key in b
        }
        if set(a) != expected_keys | {
            "fusion_singular_values",
            "fusion_rank",
            "scores",
            "labels",
            "strength",
            "losses",
        }:
            raise ValueError("Unexpected separate-model arrays")
        expected = np.column_stack([b["scores"] for b in branches])
        if not np.array_equal(a["scores"], expected):
            raise ValueError("Invalid fused coordinate order/values")
        singular, rank = fusion_spectrum(a["scores"])
        if (
            a["fusion_rank"].shape != ()
            or a["fusion_rank"].dtype.kind not in "iu"
            or int(a["fusion_rank"]) != rank
            or summary["fusion_rank"] != rank
        ):
            raise ValueError("Invalid fused numerical rank")
        if a["fusion_singular_values"].shape != (4,) or not np.allclose(
            singular, a["fusion_singular_values"], atol=1e-10, rtol=1e-10
        ):
            raise ValueError("Invalid fused spectrum")
        if not np.array_equal(a["losses"], 0.5 * (branches[0]["losses"] + branches[1]["losses"])):
            raise ValueError("Invalid descriptive mean branch loss")
        labels, strength = a["labels"], a["strength"]
        if (
            labels.shape != (n,)
            or labels.dtype.kind not in "iu"
            or np.any(labels < -1)
            or strength.shape != (n,)
            or strength.dtype.kind not in "if"
            or not np.isfinite(strength).all()
            or np.any((strength < 0) | (strength > 1))
            or np.any(strength[labels < 0] != 0)
        ):
            raise ValueError("Invalid saved density assignments")
        if len(set(labels) - {-1}) != summary["selected_k"] or any(
            sum(labels == k) < 5 for k in set(labels) - {-1}
        ):
            raise ValueError("Invalid saved density count/size")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed separate-encoder model") from exc
    return SeparateNeuralModel(config, a)
