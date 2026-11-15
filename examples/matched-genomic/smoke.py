"""Invented matched inputs and installed CLI smoke; no biological reference labels."""

import argparse
import json
from pathlib import Path


def invented(directory, *, seed=301, law="complementary", n=64, draws=0):
    import numpy as np

    from super_deli.core.data import DatasetConfig, Feature, Modality, Observation, Specimen
    from super_deli.io.config import write_json
    from super_deli.models.config import BaselineConfig
    from super_deli.models.genomic_analysis_config import GenomicAnalysisArm, GenomicAnalysisConfig
    from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
    from super_deli.models.tabular_cluster_config import TabularClusterConfig

    directory = Path(directory)
    directory.mkdir()
    rng = np.random.default_rng(seed)
    ids = [f"unit_{i:03d}" for i in range(n)]
    truth = np.arange(n) % 4
    rng.shuffle(truth)
    a, b = 2 * (truth // 2) - 1, 2 * (truth % 2) - 1
    p = rng.uniform(0.2, 0.8, 64)[None, :] + np.zeros((n, 64))
    m = rng.normal(size=(n, 4))
    e = rng.normal(size=(n, 4))
    if law != "common":
        if law != "phenotype_only":
            p += 0.15 * a[:, None] * np.tile([1, -1], 32)[None, :]
        if law != "genomic_only":
            m[:, 0] += 3 * b
            e[:, 0] += 3 * (a if law == "redundant" else b)
        if law == "redundant":
            m[:, 1] += 3 * a
            e[:, 1] += 3 * b
        if law == "conflicting":
            m[:, 1] += 3 * a * b
            e[:, 0] = rng.normal(size=n) + 3 * a * b
    raw = rng.binomial(2, np.clip(p, 0.03, 0.97))
    # Deliberately distinct source row orders; links must resolve identities.
    gorder = rng.permutation(n)
    (directory / "calls.phy").write_text(
        f"{n} 64\n"
        + "".join(f"g{i:03d} " + "".join("AMC"[v] for v in raw[i]) + "\n" for i in gorder)
    )
    mods = [
        Modality(
            "snp",
            "genomic",
            "specimen",
            "phylip",
            "calls.phy",
            tuple(Observation(f"g{i:03d}", ids[i]) for i in range(n)),
        )
    ]
    features = tuple(Feature(f"f{j}", "continuous") for j in range(4))
    for name, kind, x in (("traits", "morphological", m), ("habitat", "ecological", e)):
        order = rng.permutation(n)
        (directory / (name + ".csv")).write_text(
            "observation_id,f0,f1,f2,f3\n"
            + "".join(
                f"{name}{i:03d}," + ",".join(format(v, ".17g") for v in x[i]) + "\n" for i in order
            )
        )
        mods.append(
            Modality(
                name,
                kind,
                "specimen",
                "csv",
                name + ".csv",
                tuple(Observation(f"{name}{i:03d}", ids[i]) for i in range(n)),
                features,
            )
        )
    dataset = DatasetConfig("invented-" + law, tuple(mods), tuple(Specimen(u) for u in ids))
    common = dict(
        n_pcs=2, max_k=4, min_cluster_size=5, n_bootstrap=draws, n_init=10, random_seed=301
    )
    g = BaselineConfig(2, "error", True, **common)
    t = TabularClusterConfig(True, feature_names=tuple(f.name for f in features), **common)
    views = (
        ViewConfig("snp", "genomic", ploidy=2),
        ViewConfig("traits", "morphological", feature_names=t.feature_names),
        ViewConfig("habitat", "ecological", feature_names=t.feature_names),
    )
    blocks = tuple(f"b{i // 4}" for i in range(64))
    arms = [
        GenomicAnalysisArm("genomic", "genomic", ("snp",), g, blocks),
        GenomicAnalysisArm("morphology", "tabular", ("traits",), t),
        GenomicAnalysisArm("ecology", "tabular", ("habitat",), t),
    ]
    for name, selected in (("phenotypes", views[1:]), ("combined", views)):
        model = IntegrativeConfig(True, views=selected, covariance_type="tied", **common)
        arms.append(
            GenomicAnalysisArm(name, "integrative", tuple(v.modality for v in selected), model)
        )
    config = GenomicAnalysisConfig(tuple(arms), "individual")
    write_json(dataset.to_dict(), directory / "dataset.json")
    write_json(config.to_dict(), directory / "analysis.json")
    write_json(
        dict(unit_ids=ids, labels=truth.tolist(), law=law, seed=seed),
        directory / "truth-evaluation-only.json",
    )
    return dataset, config


def smoke(output):
    from super_deli.cli import main

    output = Path(output)
    output.mkdir()
    invented(output / "source")
    status = main(
        [
            "analyze-matched-genomic",
            str(output / "source/dataset.json"),
            "--analysis-config",
            str(output / "source/analysis.json"),
            "--output-dir",
            str(output / "analysis"),
        ]
    )
    assert status == 0
    summary = json.loads((output / "analysis/summary.json").read_text())
    assert summary["status"] == "complete" and len(summary["models"]) == 5
    assert len(summary["comparisons"]) == 10 and summary["inference"]["p_value"] is None
    assert all(r["stability"]["requested"] == 0 for r in summary["models"])
    print(json.dumps(dict(passed=True, models=5, n_units=summary["n_units"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    smoke(parser.parse_args().output_dir)
