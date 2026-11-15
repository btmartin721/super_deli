"""Invented complete matched inputs; source labels never enter discovery models."""

import importlib.util
from dataclasses import replace
from pathlib import Path


def _base(directory, seed, law, n, draws):
    path = Path(__file__).resolve().parents[1] / "matched-genomic" / "smoke.py"
    spec = importlib.util.spec_from_file_location("genomic_input_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.invented(directory, seed=seed, law=law, n=n, draws=draws)


def write_shape(path, values, observation_ids, landmark_ids, *, format="tps"):
    import numpy as np

    values = np.asarray(values)
    if format == "tps":
        token = "LM" if values.shape[2] == 2 else "LM3"
        path.write_text(
            "".join(
                f"{token}={len(landmark_ids)}\n"
                + "".join(" ".join(f"{v:.17g}" for v in row) + "\n" for row in points)
                + f"ID={name}\n"
                for name, points in zip(observation_ids, values, strict=True)
            )
        )
    else:
        path.write_text(
            "observation_id,landmark_id,"
            + ",".join("xyz"[: values.shape[2]])
            + "\n"
            + "".join(
                name + "," + landmark + "," + ",".join(f"{v:.17g}" for v in row) + "\n"
                for name, points in zip(observation_ids, values, strict=True)
                for landmark, row in zip(landmark_ids, points, strict=True)
            )
        )


def invented(
    directory, *, seed=4200, law="complementary", n=64, draws=0, dimensions=2, format="tps"
):
    import numpy as np

    from super_deli.core.data import Modality, MorphometricReaderOptions, Observation
    from super_deli.io.config import read_json, write_json
    from super_deli.models.integrative_config import ViewConfig
    from super_deli.models.multimodal_analysis_config import (
        MultimodalAnalysisArm,
        MultimodalAnalysisConfig,
    )
    from super_deli.models.shape_cluster_config import ShapeClusterConfig
    from super_deli.models.shape_config import ShapeConfig

    mapping = dict(
        continuous_control="common",
        redundant="redundant",
        complementary="genomic_only",
        conflicting="conflicting",
        genomic_only="genomic_only",
        shape_only="common",
    )
    directory = Path(directory)
    directory.mkdir()
    dataset, old = _base(directory / "base", seed, mapping[law], n, draws)
    dataset = replace(
        dataset, modalities=tuple(replace(m, path="base/" + m.path) for m in dataset.modalities)
    )
    truth = read_json(directory / "base/truth-evaluation-only.json")
    y = np.asarray(truth["labels"])
    a, b = 2 * (y // 2) - 1, 2 * (y % 2) - 1
    rng = np.random.default_rng(seed + 100000)
    angles = np.arange(6) * np.pi / 3
    base = np.column_stack((np.cos(angles), np.sin(angles)))
    if dimensions == 3:
        base = np.column_stack((base, [0.1, -0.2, 0.15, -0.1, 0.2, -0.15]))
    base -= base.mean(axis=0)
    first = np.zeros_like(base)
    first[0, 0] = 1
    first[3, 0] = -1
    second = np.zeros_like(base)
    second[1, 1] = 1
    second[4, 1] = -1
    modes = []
    nuisance = [base]
    if dimensions == 2:
        nuisance.append(base @ np.array([[0, -1], [1, 0]]))
    else:
        for axis in np.eye(3):
            nuisance.append(np.cross(base, axis))
    orthogonal = []
    for value in [*nuisance, first, second]:
        v = value.copy()
        v -= v.mean(axis=0)
        for q in orthogonal:
            v -= np.sum(v * q) * q
        v /= np.linalg.norm(v)
        orthogonal.append(v)
    modes = orthogonal[-2:]
    values = base[None, :, :] + rng.normal(0, 0.025, size=(n, 6, dimensions))
    if law in {"redundant", "shape_only"}:
        values += 0.18 * (a[:, None, None] * modes[0] + b[:, None, None] * modes[1])
    elif law == "complementary":
        values += 0.18 * b[:, None, None] * modes[0]
    elif law == "conflicting":
        values += 0.18 * (a * b)[:, None, None] * modes[0]
    ids = truth["unit_ids"]
    order = rng.permutation(n)
    landmarks = tuple(f"L{i + 1}" for i in range(6))
    filename = "shape.tps" if format == "tps" else "shape.csv"
    write_shape(
        directory / filename,
        values[order],
        [f"shape{i:03d}" for i in order],
        landmarks,
        format=format,
    )
    modality = Modality(
        "outline",
        "morphometric",
        "specimen",
        format if format == "tps" else "landmarks_csv",
        filename,
        tuple(Observation(f"shape{i:03d}", u) for i, u in enumerate(ids)),
        morphometric_options=MorphometricReaderOptions(dimensions, landmarks),
    )
    dataset = replace(dataset, modalities=(*dataset.modalities, modality))
    shape = ShapeClusterConfig(
        True,
        n_pcs=2,
        max_k=4,
        min_cluster_size=5,
        n_bootstrap=draws,
        n_init=10,
        random_seed=301,
        shape=ShapeConfig(),
    )
    singleton = [
        MultimodalAnalysisArm(a.name, a.family, a.modalities, a.model, a.block_ids)
        for a in old.arms
        if a.family != "integrative"
    ]
    singleton.append(MultimodalAnalysisArm("shape", "shape", ("outline",), shape))
    integrated = old.arms[-1].model
    integrated = replace(
        integrated,
        views=(*integrated.views, ViewConfig("outline", "morphometric", shape=shape.shape)),
    )
    analysis = MultimodalAnalysisConfig(
        (
            *singleton,
            MultimodalAnalysisArm(
                "combined", "integrative", tuple(v.modality for v in integrated.views), integrated
            ),
        ),
        "individual",
    )
    write_json(dataset.to_dict(), directory / "dataset.json")
    write_json(analysis.to_dict(), directory / "analysis.json")
    write_json(
        {**truth, "law": law, "shape_dimensions": dimensions},
        directory / "truth-evaluation-only.json",
    )
    return dataset, analysis


def select_views(dataset, analysis, names):
    from super_deli.models.multimodal_analysis_config import MultimodalAnalysisArm

    singles = tuple(
        a for a in analysis.arms if a.family != "integrative" and a.modalities[0] in names
    )
    model = replace(
        analysis.arms[-1].model,
        views=tuple(v for v in analysis.arms[-1].model.views if v.modality in names),
    )
    combined = MultimodalAnalysisArm(
        "combined", "integrative", tuple(v.modality for v in model.views), model
    )
    analysis = replace(
        analysis,
        arms=(*singles, combined),
        ecology_measurement_scope="individual" if "habitat" in names else None,
    )
    return replace(
        dataset, modalities=tuple(m for m in dataset.modalities if m.name in names)
    ), analysis
