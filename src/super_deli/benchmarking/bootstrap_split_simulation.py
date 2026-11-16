"""Known raw-population parameters, independent innovations and fixed VCF coding."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from super_deli.io.config import write_json
from super_deli.models.full_retention import full_retention_model


def parameters(seed, law):
    if law not in {"continuous_control", "shape_only"}:
        raise ValueError("Known raw law required")
    angles = np.arange(6) * np.pi / 3
    base = np.column_stack((np.cos(angles), np.sin(angles)))
    first = np.zeros_like(base)
    first[0, 0] = 1
    first[3, 0] = -1
    second = np.zeros_like(base)
    second[1, 1] = 1
    second[4, 1] = -1
    basis = []
    for v in (base, base @ np.array([[0, -1], [1, 0]]), first, second):
        q = v - v.mean(0)
        for old in basis:
            q = q - np.sum(q * old) * old
        basis.append(q / np.linalg.norm(q))
    return dict(
        seed=seed,
        law=law,
        allele_frequencies=np.random.default_rng(seed).uniform(0.2, 0.8, 64).tolist(),
        base=base.tolist(),
        modes=np.array(basis[-2:]).tolist(),
        shape_sd=0.025,
        shape_amplitude=0.18,
        trait_sd=1.0,
        ref="A",
        alt="C",
    )


def cohort(directory, parameter, n, seed, tag, generator):
    directory = Path(directory)
    spec = importlib.util.spec_from_file_location("bootstrap_split_layout", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset, analysis = module.invented(
        directory, n=n, seed=seed, law="continuous_control", draws=0
    )
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 4 if parameter["law"] == "shape_only" else np.zeros(n, dtype=int)
    rng.shuffle(y)
    arrays = dict(
        genomic=rng.binomial(2, np.asarray(parameter["allele_frequencies"]), size=(n, 64)),
        morphological=rng.normal(size=(n, 4)),
        ecological=rng.normal(size=(n, 4)),
    )
    shape = np.asarray(parameter["base"])[None] + rng.normal(
        0, parameter["shape_sd"], size=(n, 6, 2)
    )
    if parameter["law"] == "shape_only":
        a, b = 2 * (y // 2) - 1, 2 * (y % 2) - 1
        shape += parameter["shape_amplitude"] * (
            a[:, None, None] * np.asarray(parameter["modes"])[0]
            + b[:, None, None] * np.asarray(parameter["modes"])[1]
        )
    arrays["morphometric"] = shape
    unit_map = {s.specimen_id: tag + "_" + s.specimen_id for s in dataset.specimens}
    mods = []
    for source in dataset.modalities:
        observations = tuple(
            replace(o, observation_id=tag + o.observation_id, unit_id=unit_map[o.unit_id])
            for o in source.observations
        )
        order = rng.permutation(n)
        values = arrays[source.kind]
        if source.kind == "genomic":
            source = replace(source, format="vcf", path="calls.vcf")
            header = (
                "##fileformat=VCFv4.2\n##contig=<ID=panel,length=64>\n"
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            )
            header += (
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(observations[i].observation_id for i in order)
                + "\n"
            )
            header += "".join(
                "panel\t"
                + str(j + 1)
                + "\tL"
                + str(j)
                + "\tA\tC\t.\tPASS\t.\tGT\t"
                + "\t".join(("0/0", "0/1", "1/1")[values[i, j]] for i in order)
                + "\n"
                for j in range(64)
            )
            (directory / source.path).write_text(header)
        elif source.kind == "morphometric":
            module.write_shape(
                directory / source.path,
                values[order],
                [observations[i].observation_id for i in order],
                tuple(f"L{i + 1}" for i in range(6)),
            )
        else:
            (directory / source.path).write_text(
                "observation_id,f0,f1,f2,f3\n"
                + "".join(
                    observations[i].observation_id
                    + ","
                    + ",".join(f"{v:.17g}" for v in values[i])
                    + "\n"
                    for i in order
                )
            )
        mods.append(replace(source, observations=observations))
    dataset = replace(
        dataset,
        dataset_id="diagnostic-" + tag,
        modalities=tuple(mods),
        specimens=tuple(replace(s, specimen_id=unit_map[s.specimen_id]) for s in dataset.specimens),
    )
    model = full_retention_model(
        replace(analysis.arms[-1].model, n_init=100, n_bootstrap=0), min_cluster_size=9
    )
    truth = dict(
        unit_ids=list(unit_map.values()),
        labels=y.tolist(),
        law=parameter["law"],
        seed=seed,
        tag=tag,
    )
    for unused in (
        "base/calls.phy",
        "base/dataset.json",
        "base/analysis.json",
        "base/truth-evaluation-only.json",
        "analysis.json",
        "dataset.json",
        "truth-evaluation-only.json",
    ):
        (directory / unused).unlink()
    write_json(model.to_dict(), directory / "model.json")
    write_json(dataset.to_dict(), directory / "dataset.json")
    write_json(truth, directory / "truth-evaluation-only.json")
    write_json(parameter, directory / "known-parameters.json")
    with (directory / "known-raw.npz").open("xb") as f:
        np.savez(f, **arrays)
    return dataset, model, truth
