"""Separate paired workflow, strict numeric reload and source-bound model artifacts."""

from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json
from super_deli.models.matched_neural_config import MatchedNeuralConfig
from super_deli.workflows._neural_common import _report


def run_matched_neural(dataset, *, model_config, base_dir, output_dir):
    from super_deli.models.matched_neural import fit_matched_neural
    from super_deli.models.neural_config import TRAINING
    from super_deli.workflows._paired_neural_common import run_paired_neural

    return run_paired_neural(
        dataset,
        model_config=model_config,
        base_dir=base_dir,
        output_dir=output_dir,
        config_type=MatchedNeuralConfig,
        fit_function=fit_matched_neural,
        training=TRAINING,
        render=_report,
    )


def load_matched_neural_model(directory):
    import numpy as np

    from super_deli.models.matched_neural import MatchedNeuralModel

    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    config = MatchedNeuralConfig.from_dict(summary["configuration"]["model"])
    if summary.get("model") != config.model_family or summary.get("status") != "complete":
        raise ValueError("Not a completed matched neural model")
    if configuration_hash(summary["configuration"]) != summary["configuration_sha256"]:
        raise ValueError("Matched configuration hash mismatch")
    if fingerprint_file(directory / "point.npz").sha256 != summary["model_sha256"]:
        raise ValueError("Matched model hash mismatch")
    try:
        import zipfile

        with zipfile.ZipFile(directory / "point.npz") as z:
            if sum(i.file_size for i in z.infolist()) > config.max_working_bytes:
                raise ValueError("Numeric model exceeds its declared workspace")
        with np.load(directory / "point.npz", allow_pickle=False) as f:
            a = {k: f[k].copy() for k in f.files}
        p = len(config.morphology_features) + len(config.ecology_features)
        if not np.array_equal(
            a["selected_view_counts"],
            [len(config.morphology_features), len(config.ecology_features)],
        ):
            raise ValueError("Invalid selected view counts")
        kept = a["kept_indices"]
        if (
            kept.ndim != 1
            or kept.dtype.kind not in "iu"
            or not np.array_equal(kept, np.unique(kept))
            or np.any(kept < 0)
            or np.any(kept >= p)
        ):
            raise ValueError("Invalid kept indices")
        count = np.array(
            [
                sum(kept < len(config.morphology_features)),
                sum(kept >= len(config.morphology_features)),
            ]
        )
        if (
            np.any(count < 3)
            or np.any(count > 32)
            or not np.array_equal(count, a["retained_view_counts"])
        ):
            raise ValueError("Invalid per-view fitted counts")
        from super_deli.models.matched_preprocessing import effective_view_scales

        weights = effective_view_scales(count, config.loss_fractions())
        if config.schema_version == 2 and (
            not np.array_equal(a["declared_view_loss_weights"], config.view_loss_weights)
            or not np.array_equal(a["normalized_view_loss_weights"], config.loss_fractions())
        ):
            raise ValueError("Fitted view-loss weights differ from configuration")
        if config.schema_version == 1 and (
            "declared_view_loss_weights" in a or "normalized_view_loss_weights" in a
        ):
            raise ValueError("Unexpected view-loss fractions in a legacy model")
        if not np.array_equal(weights, a["view_weights"]) or not np.array_equal(
            a["column_weights"], weights[(kept >= len(config.morphology_features)).astype(int)]
        ):
            raise ValueError("Invalid view balance")
        shapes = {
            "means": (p,),
            "scales": (p,),
            "latent_means": (2,),
            "latent_scales": (2,),
            "losses": (501,),
        }
        for side, widths in (
            ("encoder", [len(kept), 16, 8, 2]),
            ("decoder", [2, 8, 16, len(kept)]),
        ):
            for i, (left, right) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
                shapes[f"{side}_{i}_weight"] = (right, left)
                shapes[f"{side}_{i}_bias"] = (right,)
        for name, shape in shapes.items():
            if (
                a[name].shape != shape
                or a[name].dtype.kind not in "if"
                or not np.isfinite(a[name]).all()
            ):
                raise ValueError("Invalid model array " + name)
        if np.any(a["scales"] <= 0) or np.any(a["latent_scales"] <= 0):
            raise ValueError("Invalid fitted scales")
        if len(set(summary["unit_ids"])) != len(summary["unit_ids"]):
            raise ValueError("Invalid saved identities")
        n = len(summary["unit_ids"])
        if (
            a["scores"].shape != (n, 2)
            or not np.isfinite(a["scores"]).all()
            or a["labels"].shape != (n,)
            or a["strength"].shape != (n,)
        ):
            raise ValueError("Invalid saved point dimensions")
        if (
            a["labels"].dtype.kind not in "iu"
            or np.any(a["labels"] < -1)
            or not np.isfinite(a["strength"]).all()
            or np.any((a["strength"] < 0) | (a["strength"] > 1))
            or np.any(a["strength"][a["labels"] < 0] != 0)
        ):
            raise ValueError("Invalid saved density assignments")
        if len(set(a["labels"]) - {-1}) != summary["selected_k"]:
            raise ValueError("Saved density count mismatch")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed matched model arrays") from exc
    return MatchedNeuralModel(config, a)
