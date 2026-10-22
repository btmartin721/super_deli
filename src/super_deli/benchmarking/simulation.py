"""Four-view generative stress tests, not an evolutionary species simulator."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from super_deli.benchmarking.config import KINDS, Scenario, SimulationDimensions, derived_seed
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import write_json


@dataclass(frozen=True)
class SimulatedData:
    scenario: Scenario
    seed: int
    target_ids: tuple[str, ...]
    primary_labels: tuple[int, ...]
    alternative_labels: tuple[int, ...] | None
    rows: dict
    n_loci: int

    @property
    def common_ids(self):
        shared = set.intersection(*({r["target"] for r in self.rows[k]} for k in KINDS))
        return tuple(t for t in self.target_ids if t in shared)


def simulate(scenario: Scenario, config: SimulationDimensions, seed: int) -> SimulatedData:
    """Sample targets first; source individuals are conditionally independent members."""
    rng = np.random.default_rng(derived_seed(seed, scenario.name, 0, "targets"))
    n = config.n_units
    labels = rng.permutation(np.repeat(np.arange(4), n // 4))
    alternative = (
        rng.permutation(np.repeat(np.arange(4), n // 4))
        if scenario.pattern == "conflicting"
        else None
    )
    centers = np.column_stack((2 * (labels // 2) - 1, 2 * (labels % 2) - 1))
    latent = scenario.strength * centers + rng.normal(0, 0.35, (n, 2))
    targets = tuple(f"unit_{i:04d}" for i in range(n))
    rows = {k: [] for k in KINDS}
    for kind in KINDS:
        rng = np.random.default_rng(derived_seed(seed, scenario.name, 0, kind))
        # Baseline locus frequencies vary, but there is no LD, migration or genealogy.
        logits = rng.uniform(-0.5, 0.5, config.n_loci)
        for i, target in enumerate(targets):
            if kind == "ecological" and rng.random() < scenario.view_missing_rate:
                continue
            count = (
                int(rng.integers(1, config.max_group_members + 1))
                if scenario.unequal_groups and kind != "ecological"
                else (2 if scenario.linkage == "grouped" and kind != "ecological" else 1)
            )
            for member in range(count):
                z = latent[i].copy()
                if kind == "ecological" and alternative is not None:
                    a = alternative[i]
                    z = scenario.strength * np.array([2 * (a // 2) - 1, 2 * (a % 2) - 1])
                    z = z + rng.normal(0, 0.35, 2)
                if scenario.pattern == "complementary":
                    z[1 if kind in {"genomic", "ecological"} else 0] = 0
                z += rng.normal(0, 0.3, 2)
                observation = f"{kind}_{i:04d}_{member}"
                source = (
                    f"individual_{observation}"
                    if scenario.linkage == "grouped" and kind != "ecological"
                    else target
                )
                if kind == "genomic":
                    effects = np.repeat(z, config.n_loci // 2)
                    probabilities = 1 / (1 + np.exp(-(logits + 0.65 * effects)))
                    values = rng.binomial(2, probabilities).astype(float)
                elif kind in {"morphological", "ecological"}:
                    values = np.array([z[0], z[1], z[0] + 0.3 * z[1], z[1] - 0.2 * z[0]])
                    values += rng.normal(0, 1, 4)
                    if kind == "morphological":
                        values = np.append(values, rng.poisson(np.exp(1.8 + 0.15 * z.sum())))
                else:
                    values = np.array([[-1, -0.7], [0, -1], [1, -0.7], [1, 0.7], [0, 1], [-1, 0.7]])
                    values[4, 1] += 0.08 * z[0]
                    values[2:4, 0] += 0.08 * z[1]
                    values += rng.normal(0, 0.02, values.shape)
                    angle = rng.uniform(-np.pi, np.pi)
                    rotation = np.array(
                        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                    )
                    values = values @ rotation * rng.uniform(0.5, 3) + rng.uniform(-10, 10, 2)
                if kind != "morphometric":
                    values[rng.random(values.shape) < scenario.cell_missing_rate] = np.nan
                rows[kind].append(
                    {
                        "observation": observation,
                        "source": source,
                        "target": target,
                        "values": values,
                    }
                )
        rng.shuffle(rows[kind])
    return SimulatedData(
        scenario,
        seed,
        targets,
        tuple(0 for _ in labels) if scenario.pattern == "null" else tuple(map(int, labels)),
        tuple(map(int, alternative)) if alternative is not None else None,
        rows,
        config.n_loci,
    )


def _write_dataset(data, directory, allowed, *, canonical_order=False):
    directory.mkdir()
    rows = {kind: [r for r in data.rows[kind] if r["target"] in allowed] for kind in KINDS}
    if canonical_order:
        order = {t: i for i, t in enumerate(data.target_ids)}
        for records in rows.values():
            records.sort(key=lambda r: (order[r["target"]], r["observation"]))
    with (directory / "genotypes.phy").open("x", encoding="utf-8") as handle:
        handle.write(f"{len(rows['genomic'])} {data.n_loci}\n")
        for row in rows["genomic"]:
            calls = "".join("N" if np.isnan(v) else "ARG"[int(v)] for v in row["values"])
            handle.write(f"{row['observation']} {calls}\n")
    for kind in ("morphological", "ecological", "morphometric"):
        with (directory / f"{kind}.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            if kind == "morphometric":
                writer.writerow(("observation_id", "landmark_id", "x", "y"))
                for row in rows[kind]:
                    for j, point in enumerate(row["values"]):
                        writer.writerow((row["observation"], f"L{j + 1}", *point))
            else:
                p = 5 if kind == "morphological" else 4
                writer.writerow(("observation_id", *(f"feature_{j}" for j in range(p))))
                for row in rows[kind]:
                    writer.writerow(
                        (
                            row["observation"],
                            *(
                                "" if np.isnan(v) else (int(v) if j == 4 else v)
                                for j, v in enumerate(row["values"])
                            ),
                        )
                    )
    modalities, specimens = [], {}
    if data.scenario.linkage == "matched":
        specimens = {t: {"specimen_id": t} for t in data.target_ids}
    for kind in KINDS:
        grouped = data.scenario.linkage == "grouped"
        level = "locality" if grouped and kind == "ecological" else "specimen"
        if grouped and level == "specimen":
            specimens.update(
                {
                    r["source"]: {"specimen_id": r["source"], "locality_id": r["target"]}
                    for r in rows[kind]
                }
            )
        modality = {
            "name": kind,
            "kind": kind,
            "level": level,
            "format": "phylip"
            if kind == "genomic"
            else ("landmarks_csv" if kind == "morphometric" else "csv"),
            "path": "genotypes.phy" if kind == "genomic" else f"{kind}.csv",
            "observations": [
                {"observation_id": r["observation"], "unit_id": r["source"]} for r in rows[kind]
            ],
        }
        if kind in {"morphological", "ecological"}:
            modality["features"] = [
                {"name": f"feature_{j}", "dtype": "count" if j == 4 else "continuous"}
                for j in range(5 if kind == "morphological" else 4)
            ]
        if kind == "morphometric":
            modality["morphometric_options"] = {
                "dimensions": 2,
                "landmark_ids": [f"L{j + 1}" for j in range(6)],
                "coordinate_unit": "arbitrary",
            }
        modalities.append(modality)
    dataset = {
        "schema_version": 1,
        "dataset_id": f"simulation_{data.scenario.name}",
        "specimens": list(specimens.values()),
        "modalities": modalities,
        "localities": list(data.target_ids) if data.scenario.linkage == "grouped" else [],
    }
    write_json(dataset, directory / "dataset.json")
    return dataset


def write_simulation(data: SimulatedData, directory: Path):
    """Preserve observed coverage, and stage a common cohort for every comparison arm."""
    directory.mkdir(exist_ok=False)
    _write_dataset(data, directory / "observed", set(data.target_ids))
    cohort = _write_dataset(data, directory / "cohort", set(data.common_ids), canonical_order=True)
    write_json(
        {
            "schema_version": 1,
            "seed": data.seed,
            "scope": "Constructed generating partitions, not empirical species identities",
            "units": [
                {
                    "unit_id": t,
                    "primary": data.primary_labels[i],
                    "alternative": data.alternative_labels[i] if data.alternative_labels else None,
                }
                for i, t in enumerate(data.target_ids)
            ],
        },
        directory / "truth.json",
    )
    write_json(
        {
            "schema_version": 1,
            "target_ids": data.target_ids,
            "common_ids": data.common_ids,
            "excluded_ids": [t for t in data.target_ids if t not in set(data.common_ids)],
            "coverage": {k: sorted({r["target"] for r in data.rows[k]}) for k in KINDS},
            "source_counts": {k: len(data.rows[k]) for k in KINDS},
            "input_hashes": {
                str(p.relative_to(directory)): fingerprint_file(p).sha256
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            },
        },
        directory / "manifest.json",
    )
    return cohort
