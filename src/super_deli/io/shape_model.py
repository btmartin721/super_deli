"""Versioned, hashed NumPy model parameters; no pickle or executable model loading."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.shape_config import ShapeConfig

ARRAYS = (
    "consensus",
    "tangent_mean",
    "components",
    "explained_variance",
    "explained_variance_ratio",
)


def write_shape_model(
    model,
    *,
    directory: Path,
    landmark_ids: tuple[str, ...],
    observation_ids: tuple[str, ...],
    unit_ids: tuple[str, ...],
) -> dict:
    import numpy as np

    path = directory / "model.npz"
    with path.open("xb") as handle:
        np.savez(handle, **{name: getattr(model, name) for name in ARRAYS})
    metadata = {
        "schema_version": 1,
        "config": model.config.to_dict(),
        "landmark_ids": list(landmark_ids),
        "dimensions": int(model.consensus.shape[1]),
        "training_observation_ids": list(observation_ids),
        "training_unit_ids": list(unit_ids),
        "array_path": "model.npz",
        "array_sha256": fingerprint_file(path).sha256,
        "numerical_rank": model.numerical_rank,
        "shape_dimension": model.shape_dimension,
        "total_variance": model.total_variance,
    }
    write_json(metadata, directory / "model.json")
    return metadata


def read_shape_model(directory: str | Path):
    import numpy as np

    from super_deli.models.shape_pca import ShapePCAModel

    directory = Path(directory).resolve()
    metadata_path, array_path = directory / "model.json", directory / "model.npz"
    metadata_fingerprint = fingerprint_file(metadata_path)
    metadata = read_json(metadata_path)
    expected = {
        "schema_version",
        "config",
        "landmark_ids",
        "dimensions",
        "training_observation_ids",
        "training_unit_ids",
        "array_path",
        "array_sha256",
        "numerical_rank",
        "shape_dimension",
        "total_variance",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected:
        raise ValueError("Invalid shape model metadata fields")
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != 1
        or metadata["array_path"] != "model.npz"
    ):
        raise ValueError("Unsupported shape model schema or array reference")
    config = ShapeConfig.from_dict(metadata["config"])
    array_fingerprint = fingerprint_file(array_path)
    if array_fingerprint.sha256 != metadata["array_sha256"]:
        raise ValueError("Shape model array SHA-256 does not match metadata")
    try:
        with ZipFile(array_path) as archive:
            if {i.filename for i in archive.infolist()} != {f"{n}.npy" for n in ARRAYS} or len(
                archive.infolist()
            ) != len(ARRAYS):
                raise ValueError("Unexpected model array members")
            if sum(i.file_size for i in archive.infolist()) > config.max_working_bytes:
                raise ValueError("Model arrays exceed max_working_bytes")
        with np.load(array_path, allow_pickle=False) as loaded:
            values = {name: loaded[name].copy() for name in ARRAYS}
    except BadZipFile as exc:
        raise ValueError("Invalid shape model array archive") from exc
    ids = metadata["landmark_ids"]
    train, units = metadata["training_observation_ids"], metadata["training_unit_ids"]
    for name, items in (
        ("landmark_ids", ids),
        ("training_observation_ids", train),
        ("training_unit_ids", units),
    ):
        if not isinstance(items, list) or any(
            not isinstance(x, str) or not x or x != x.strip() or "\x00" in x for x in items
        ):
            raise ValueError(f"Invalid model {name}")
    d = metadata["dimensions"]
    if type(d) is not int or d not in (2, 3) or len(ids) < 3 or len(set(ids)) != len(ids):
        raise ValueError("Invalid model landmark geometry")
    n, k, p = len(train), len(ids), len(ids) * d
    if n < 2 or len(set(train)) != n or len(units) != n:
        raise ValueError("Invalid model training links")
    if config.repeated_policy == "error" and len(set(units)) != n:
        raise ValueError("Repeated model training units conflict with policy")
    dof = k * d - d - 1 - d * (d - 1) // 2
    rank = metadata["numerical_rank"]
    if (
        type(rank) is not int
        or not 0 <= rank <= min(n - 1, dof)
        or metadata["shape_dimension"] != dof
    ):
        raise ValueError("Invalid model shape rank")
    if values["components"].ndim != 2:
        raise ValueError("Invalid model component matrix dimensionality")
    r = values["components"].shape[0]
    shapes = {
        "consensus": (k, d),
        "tangent_mean": (p,),
        "components": (r, p),
        "explained_variance": (r,),
        "explained_variance_ratio": (r,),
    }
    if r != min(config.n_components or rank, rank):
        raise ValueError("Model component count conflicts with fitted rank/configuration")
    for name, value in values.items():
        if (
            value.dtype != np.dtype("float64")
            or value.shape != shapes[name]
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid shape model array {name}")
    consensus, components = values["consensus"], values["components"]
    if not np.allclose(consensus.mean(axis=0), 0, atol=1e-9) or not np.isclose(
        np.linalg.norm(consensus), 1, atol=1e-9
    ):
        raise ValueError("Model consensus must be centered and unit centroid size")
    if not np.allclose(components @ components.T, np.eye(r), atol=1e-9):
        raise ValueError("Model PCA components must be orthonormal")
    tangent = np.vstack((values["tangent_mean"][None, :], components)).reshape(-1, k, d)
    cross = np.einsum("ki,nkj->nij", consensus, tangent)
    if (
        not np.allclose(tangent.mean(axis=1), 0, atol=1e-8)
        or not np.allclose(np.einsum("nkd,kd->n", tangent, consensus), 0, atol=1e-8)
        or not np.allclose(cross, cross.transpose(0, 2, 1), atol=1e-8)
    ):
        raise ValueError("Model parameters leave the centered horizontal tangent space")
    from super_deli.models.procrustes import preshapes

    preshapes(consensus[None, :, :], config)
    total = metadata["total_variance"]
    if (
        type(total) not in (int, float)
        or not 0 <= total < float("inf")
        or (rank == 0) != (total == 0)
    ):
        raise ValueError("Invalid model total variance")
    if np.any(values["explained_variance"] <= 0) or np.any(values["explained_variance_ratio"] < 0):
        raise ValueError("Invalid model component variance")
    if r and not np.allclose(
        values["explained_variance_ratio"], values["explained_variance"] / total
    ):
        raise ValueError("Model variance ratios disagree with total variance")
    if values["explained_variance_ratio"].sum() > 1 + 1e-9:
        raise ValueError("Model explained variance exceeds total variance")
    if (
        fingerprint_file(array_path).sha256 != array_fingerprint.sha256
        or fingerprint_file(metadata_path).sha256 != metadata_fingerprint.sha256
    ):
        raise OSError("Shape model changed while loading")
    model = ShapePCAModel(
        config=config,
        total_variance=float(total),
        numerical_rank=rank,
        shape_dimension=dof,
        **values,
    )
    return model, metadata, [asdict(metadata_fingerprint), asdict(array_fingerprint)]
