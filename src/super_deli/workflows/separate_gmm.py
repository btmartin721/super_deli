"""Opt-in paired neural/shared-mixture I/O and strict numeric frozen reload."""

import zipfile
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.separate_gmm_config import SeparateGMMConfig


def assignment_data(point, alignment):
    p = point.model.arrays["membership"]
    return (
        [
            "unit_id",
            "morphology_observation_id",
            "ecology_observation_id",
            "cluster",
            "maximum_mixture_membership",
            *[f"component_{j + 1}_probability" for j in range(p.shape[1])],
            *[f"latent_{j + 1}" for j in range(4)],
        ],
        zip(
            alignment["unit_ids"],
            alignment["views"][0]["observation_ids"],
            alignment["views"][1]["observation_ids"],
            point.labels,
            point.strength,
            *p.T,
            *point.scores.T,
            strict=True,
        ),
    )


def render(out, summary, point, units, **kwargs):
    from super_deli.workflows.separate_neural import _render

    _render(
        out,
        summary,
        point,
        units,
        **kwargs,
        cluster_title="Separate encoders and shared Gaussian mixture",
        report_options={
            "group_label": "Gaussian mixture components",
            "strength_label": "Maximum conditional mixture membership",
            "plot_label": "Branch coordinates colored by mixture component",
            "support_explanation": (
                "Mixture membership conditions on fitted parameters. Subsampling refits both "
                "encoders and clustering. Neither is species probability or significance; "
                "p-values are not computed."
            ),
        },
    )
    summary.update(
        covariance_type="tied",
        search_boundary=point.selected_k == point.model.config.max_k,
        candidates=point.diagnostics["candidates"],
        interpretation=[
            "four separate-encoder coordinates and prespecified shared-covariance BIC",
            "complete independent matched individuals and individual ecology are assumptions",
            "mixture memberships condition on fitted parameters; not species probabilities",
            "frozen new-unit projections are not external predictive validation",
            "subsampling is conditional sensitivity; no significance computed",
        ],
    )
    write_json(point.diagnostics["candidates"], out / "candidates.json")


def run_separate_gmm(dataset, *, model_config, base_dir, output_dir):
    from super_deli.models.separate_gmm import fit_separate_gmm
    from super_deli.models.separate_neural import SEPARATE_TRAINING
    from super_deli.workflows._paired_neural_common import run_paired_neural

    return run_paired_neural(
        dataset,
        model_config=model_config,
        base_dir=base_dir,
        output_dir=output_dir,
        config_type=SeparateGMMConfig,
        fit_function=fit_separate_gmm,
        training=SEPARATE_TRAINING,
        render=render,
        assignment_factory=assignment_data,
    )


def load_separate_gmm_model(directory):
    import numpy as np

    from super_deli.models.separate_gmm import SeparateGMMModel, mixture_membership
    from super_deli.models.separate_neural import ROLES, fusion_spectrum, unpack_branch
    from super_deli.workflows.separate_neural import _branch_shapes

    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    try:
        config = SeparateGMMConfig.from_dict(summary["configuration"]["model"])
        if (
            summary["status"] != "complete"
            or summary["model"] != config.model_family
            or summary["n_coordinates"] != 4
            or summary["covariance_type"] != "tied"
        ):
            raise ValueError("Not a completed separate-encoder shared GMM")
        if (
            configuration_hash(summary["configuration"]) != summary["configuration_sha256"]
            or fingerprint_file(directory / "point.npz").sha256 != summary["model_sha256"]
        ):
            raise ValueError("Shared-mixture model/configuration fingerprint mismatch")
        with zipfile.ZipFile(directory / "point.npz") as archive:
            if sum(m.file_size for m in archive.infolist()) > config.max_working_bytes:
                raise ValueError("Numeric model exceeds its workspace")
        with np.load(directory / "point.npz", allow_pickle=False) as archive:
            a = {k: archive[k].copy() for k in archive.files}
        units = summary["unit_ids"]
        if (
            len(units) < 10
            or units != sorted(set(units))
            or any(not isinstance(u, str) or not u or u != u.strip() for u in units)
        ):
            raise ValueError("Invalid saved unit identities")
        n = len(units)
        branches = [unpack_branch(a, role) for role in ROLES]
        for b, p in zip(
            branches, (len(config.morphology_features), len(config.ecology_features)), strict=True
        ):
            _branch_shapes(b, p, n)
        expected = {role + "__" + k for role, b in zip(ROLES, branches, strict=True) for k in b}
        if set(a) != expected | {
            "fusion_singular_values",
            "fusion_rank",
            "scores",
            "labels",
            "strength",
            "losses",
            "membership",
            "mixture_weights",
            "mixture_means",
            "mixture_covariance",
        }:
            raise ValueError("Unexpected shared-mixture arrays")
        scores = np.column_stack([b["scores"] for b in branches])
        if not np.array_equal(a["scores"], scores):
            raise ValueError("Invalid fused score order/values")
        singular, rank = fusion_spectrum(scores)
        if (
            a["fusion_rank"].shape != ()
            or a["fusion_rank"].dtype.kind not in "iu"
            or int(a["fusion_rank"]) != rank
            or summary["fusion_rank"] != rank
        ):
            raise ValueError("Invalid fused rank")
        if a["fusion_singular_values"].shape != (4,):
            raise ValueError("Invalid fused spectrum shape")
        np.testing.assert_allclose(singular, a["fusion_singular_values"], rtol=1e-10, atol=1e-10)
        if not np.array_equal(a["losses"], 0.5 * (branches[0]["losses"] + branches[1]["losses"])):
            raise ValueError("Invalid branch loss summary")
        k = summary["selected_k"]
        if type(k) is not int or not 1 <= k <= config.max_k:
            raise ValueError("Invalid selected mixture size")
        for key, shape in {
            "mixture_weights": (k,),
            "mixture_means": (k, 4),
            "mixture_covariance": (4, 4),
            "membership": (n, k),
            "labels": (n,),
            "strength": (n,),
        }.items():
            if (
                a[key].shape != shape
                or a[key].dtype.kind not in "if"
                or not np.isfinite(a[key]).all()
            ):
                raise ValueError("Invalid mixture array: " + key)
        w, cov = a["mixture_weights"], a["mixture_covariance"]
        if (
            np.any(w <= 0)
            or not np.isclose(w.sum(), 1, atol=1e-12)
            or not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("Invalid mixture weights/covariance")
        p = mixture_membership(scores, a)
        if np.any((a["membership"] < 0) | (a["membership"] > 1)):
            raise ValueError("Invalid conditional membership range")
        np.testing.assert_allclose(p, a["membership"], rtol=1e-8, atol=1e-8)
        if (
            a["labels"].dtype.kind not in "iu"
            or not np.array_equal(p.argmax(1), a["labels"])
            or any(sum(a["labels"] == j) < config.min_cluster_size for j in range(k))
        ):
            raise ValueError("Invalid saved mixture assignments")
        np.testing.assert_allclose(p.max(1), a["strength"], rtol=1e-8, atol=1e-8)
        if tuple(int(np.flatnonzero(a["labels"] == j)[0]) for j in range(k)) != tuple(
            sorted(int(np.flatnonzero(a["labels"] == j)[0]) for j in range(k))
        ):
            raise ValueError("Invalid canonical component order")
    except (KeyError, TypeError, IndexError, AssertionError, np.linalg.LinAlgError) as exc:
        raise ValueError("Malformed separate-encoder shared GMM") from exc
    return SeparateGMMModel(config, a)
