"""Frozen, truth-separated paired density/GMM study with audited storage gating."""

from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from importlib.metadata import version
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json

CASES = ("gaussian", "four_spherical", "four_unequal", "moons", "background", "cline")


@dataclass(frozen=True)
class DensityStudyConfig:
    schema_version: int = 1
    n_datasets: int = 25
    stability_datasets: int = 5
    n_subsamples: int = 25
    sample_sizes: tuple[int, ...] = (64, 256)
    main_seed: int = 94217
    preflight_seed: int = 93997
    data_seed: int = 93709
    subsampling_seed: int = 93737
    optimizer_seed: int = 71323
    artifact_cap_bytes: int = 1073741824
    min_free_bytes: int = 8589934592

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported density study version")
        for f in fields(self):
            if f.name in {"sample_sizes", "schema_version"}:
                continue
            if type(getattr(self, f.name)) is not int or getattr(self, f.name) < 1:
                raise ValueError(f"{f.name} must be a positive integer")
        if not 1 <= self.stability_datasets <= self.n_datasets:
            raise ValueError("Invalid stability_datasets")
        if (
            not isinstance(self.sample_sizes, tuple)
            or not self.sample_sizes
            or any(type(n) is not int or n < 32 or n % 8 for n in self.sample_sizes)
            or len(set(self.sample_sizes)) != len(self.sample_sizes)
        ):
            raise ValueError("Unique sample sizes divisible by eight and at least 32 required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or "schema_version" not in value
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Malformed density study configuration")
        value = dict(value)
        if "sample_sizes" in value:
            if not isinstance(value["sample_sizes"], (tuple, list)):
                raise ValueError("sample_sizes must be an array")
            value["sample_sizes"] = tuple(value["sample_sizes"])
        return cls(**value)

    def plan(self, phase):
        if phase not in {"preflight", "main"}:
            raise ValueError("Unknown study phase")
        count = 1 if phase == "preflight" else self.n_datasets
        root = self.preflight_seed if phase == "preflight" else self.main_seed
        return [
            {
                "case": case,
                "n": n,
                "dataset": i,
                "phase": phase,
                "data_seed": [self.data_seed, root, c, n, i],
                "subsampling_seed": [self.subsampling_seed, root, c, n, i],
                "draws": self.n_subsamples
                if phase == "preflight" or i < self.stability_datasets
                else 0,
                "key": f"{case}_n{n}_{i:04d}",
            }
            for c, case in enumerate(CASES)
            for n in self.sample_sizes
            for i in range(count)
        ]


def generate_raw(case, n, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    centers = np.array([[-3, -3], [-3, 3], [3, -3], [3, 3]])
    if case == "gaussian":
        x, truth = rng.normal(size=(n, 2)), np.zeros(n, dtype=int)
    elif case == "cline":
        u = rng.uniform(-3, 3, n)
        x, truth = np.column_stack([u, 0.5 * u + rng.normal(0, 0.3, n)]), None
    elif case == "moons":
        a, b = rng.uniform(0, np.pi, (2, n // 2))
        x = np.concatenate(
            [
                np.column_stack([np.cos(a), np.sin(a)]),
                np.column_stack([1 - np.cos(b), 0.5 - np.sin(b)]),
            ]
        )
        x += rng.normal(0, 0.08, x.shape)
        truth = np.repeat(np.arange(2), n // 2)
    elif case in {"four_spherical", "four_unequal", "background"}:
        count = n - n // 8 if case == "background" else n
        counts = [count // 4] * 4
        counts[-1] += count - sum(counts)
        cov = [np.eye(2) * (0.36 if case == "background" else 1)] * 4
        if case == "four_unequal":
            counts = [int(0.4 * n), int(0.3 * n), int(0.2 * n)]
            counts.append(n - sum(counts))
            cov = [0.25 * np.eye(2), np.eye(2)]
            for angle, diagonal in [(35, [2.25, 0.36]), (70, [1.44, 0.16])]:
                t = np.deg2rad(angle)
                r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                cov.append(r @ np.diag(diagonal) @ r.T)
        x = np.concatenate(
            [
                rng.multivariate_normal(m, v, size=k)
                for m, v, k in zip(centers, cov, counts, strict=True)
            ]
        )
        truth = np.concatenate([np.full(k, i) for i, k in enumerate(counts)])
        if case == "background":
            x = np.concatenate([x, rng.uniform(-10, 10, (n // 8, 2))])
            truth = np.concatenate([truth, np.full(n // 8, -1)])
    else:
        raise ValueError("Unknown simulation case")
    order = rng.permutation(n)
    return x[order], truth[order] if truth is not None else None


def write_raw(x, directory):
    from super_deli.core.schema import config_from_dict

    out = Path(directory)
    out.mkdir()
    ids = [f"unit_{i:04d}" for i in range(len(x))]
    with (out / "traits.tsv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["observation_id", "trait_1", "trait_2"])
        writer.writerows([u, *row] for u, row in zip(ids, x, strict=True))
    payload = {
        "schema_version": 1,
        "dataset_id": "invented_density_data",
        "specimens": [{"specimen_id": u} for u in ids],
        "modalities": [
            {
                "name": "traits",
                "kind": "morphological",
                "level": "specimen",
                "format": "tsv",
                "path": "traits.tsv",
                "features": [{"name": f"trait_{i}", "dtype": "continuous"} for i in (1, 2)],
                "observations": [{"observation_id": u, "unit_id": u} for u in ids],
            }
        ],
    }
    dataset = config_from_dict(payload)
    write_json(dataset.to_dict(), out / "dataset.json")
    return dataset


def recovery_metrics(labels, truth):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    assigned = labels >= 0
    result = {
        "selected_k": len(set(labels) - {-1}),
        "assigned_fraction": float(np.mean(assigned)),
        "noise_count": int(np.sum(~assigned)),
    }
    if truth is None:
        return {
            **result,
            "correct_k": None,
            "ari": None,
            "assigned_ari": None,
            "reference": "continuous cline; no discrete generating partition",
        }
    core = truth >= 0
    predicted = labels[core].copy()
    lost = predicted < 0
    predicted[lost] = np.arange(np.sum(lost)) + max(int(labels.max()), 0) + 1
    paired = core & assigned
    splits = sum(len(set(labels[(truth == k) & assigned])) > 1 for k in set(truth[core]))
    merges = sum(len(set(truth[(labels == k) & core])) > 1 for k in set(labels[assigned]))
    return {
        **result,
        "correct_k": result["selected_k"] == len(set(truth[core])),
        "ari": float(adjusted_rand_score(truth[core], predicted)),
        "assigned_ari": float(adjusted_rand_score(truth[paired], labels[paired]))
        if paired.sum() >= 2
        else None,
        "core_assigned_fraction": float(np.mean(assigned[core])),
        "split_generating_groups": splits,
        "merged_predicted_groups": merges,
        "background_recall": float(np.mean(~assigned[~core])) if (~core).any() else None,
        "noise_precision": float(np.mean(~core[~assigned])) if (~assigned).any() else None,
    }


def _gmm_family(table, config, draws, seed):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.models.mixture import FIT_LOCK
    from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples
    from super_deli.models.tabular_gmm import _fit_once, _matrix
    from super_deli.models.tabular_hdbscan import DensityPoint, point_arrays

    prep = replace(config.preprocessing_config(), min_cluster_size=5, random_seed=seed, n_init=10)
    x, names, types = _matrix(table, prep)
    arrays, records = {}, []
    points = (
        (
            np.arange(len(x)),
            *subsample_indices(len(x), draws, config.subsample_fraction, config.random_seed),
        )
        if draws
        else (np.arange(len(x)),)
    )
    point_meta = None
    with FIT_LOCK, threadpool_limits(limits=1):
        for i, ids in enumerate(points):
            record = {"training_indices": ids.tolist(), "draw": i - 1}
            try:
                fitted = _fit_once(x[ids], names, types, prep, seed)
                mix = fitted.mixture
                point = DensityPoint(
                    fitted.model, fitted.scores, mix.labels, mix.membership.max(axis=1)
                )
                prefix = "point" if i == 0 else f"draw_{i - 1}"
                values = {
                    **point_arrays(point),
                    "weights": mix.mixture.weights_[list(mix.component_order)],
                    "mixture_means": mix.mixture.means_[list(mix.component_order)],
                    "covariances": mix.mixture.covariances_[list(mix.component_order)],
                }
                arrays.update({f"{prefix}_{k}": v for k, v in values.items()})
                record.update(
                    status="complete",
                    labels=mix.labels.tolist(),
                    candidates=list(mix.candidates),
                    selected_k=mix.selected_k,
                    excluded_features=list(fitted.model.exclusions),
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                record.update(status="failed", error=str(exc))
            if i == 0:
                point_meta = record
            else:
                records.append(record)
    stability = summarize_subsamples(len(x), records)
    arrays.update(stability.arrays)
    return point_meta, records, stability.summary, arrays


def run_density_observation(task):
    import numpy as np

    from super_deli.io.tabular import ingest_tabular
    from super_deli.models.hdbscan_config import HDBSCANConfig
    from super_deli.models.tabular_hdbscan import fit_tabular_hdbscan, point_arrays

    root, raw_config, row = task
    config = DensityStudyConfig.from_dict(raw_config)
    out = Path(root) / row["key"]
    out.mkdir()
    write_json(row, out / "plan.json")
    x, truth = generate_raw(row["case"], row["n"], row["data_seed"])
    dataset = write_raw(x, out / "raw")
    if truth is not None:
        np.savez_compressed(out / "truth.npz", labels=truth)
    table = ingest_tabular(dataset, base_dir=out / "raw", output_dir=out / "ingestion").tables[0]
    seed = int(np.random.SeedSequence(row["subsampling_seed"]).generate_state(1)[0])
    hconfig = HDBSCANConfig(True, n_subsamples=row["draws"], random_seed=seed)
    write_json(hconfig.to_dict(), out / "density-config.json")
    result = {
        "key": row["key"],
        "case": row["case"],
        "n": row["n"],
        "dataset": row["dataset"],
        "families": {},
    }
    for family in ("hdbscan", "gmm"):
        if family == "hdbscan":
            try:
                fit = fit_tabular_hdbscan(table, config=hconfig)
                point = {
                    "status": "complete",
                    "labels": fit.point.labels.tolist(),
                    "excluded_features": list(fit.point.model.exclusions),
                }
                records, stability = fit.stability.records, fit.stability.summary
                arrays = {
                    **{f"point_{k}": v for k, v in point_arrays(fit.point).items()},
                    **fit.stability.arrays,
                }
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                point, records, stability, arrays = (
                    {"status": "failed", "error": str(exc)},
                    (),
                    {},
                    {},
                )
        else:
            point, records, stability, arrays = _gmm_family(
                table, hconfig, row["draws"], config.optimizer_seed
            )
        np.savez_compressed(out / f"{family}.npz", **arrays)
        write_json(
            {"point": point, "draws": records, "stability": stability}, out / f"{family}.json"
        )
        # Truth enters evaluation only after all candidate fitting and selection.
        metrics = (
            recovery_metrics(np.array(point["labels"]), truth)
            if point["status"] == "complete"
            else {}
        )
        done = sum(r["status"] == "complete" for r in records)
        result["families"][family] = {
            "status": point["status"],
            "metrics": metrics,
            "stability": stability,
            "attempted": 1 + len(records),
            "successful": int(point["status"] == "complete") + done,
            "failed": int(point["status"] == "failed") + len(records) - done,
            "unattempted": row["draws"] - len(records),
        }
    write_json(result, out / "summary.json")
    write_json(
        {
            p.relative_to(out).as_posix(): fingerprint_file(p).sha256
            for p in out.rglob("*")
            if p.is_file()
        },
        out / "checksums.json",
    )
    return result


def retained_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def run_density_study(config, *, phase, output_dir, workers=1, gate=None):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    c = DensityStudyConfig.from_dict(config.to_dict())
    out = Path(output_dir).resolve()
    if not out.parent.is_dir() or shutil.disk_usage(out.parent).free < c.min_free_bytes:
        raise ValueError("Existing output parent and sufficient free space are required")
    if phase == "main":
        g = read_json(gate) if gate else {}
        if (
            g.get("status") != "passed"
            or g.get("configuration_sha256") != configuration_hash(c.to_dict())
            or not out.is_relative_to(Path(g["artifact_root"]))
        ):
            raise ValueError("Matching independently audited storage preflight required")
        if fingerprint_file(g["audit_file"]).sha256 != g["audit_sha256"]:
            raise ValueError("Preflight audit changed")
    plan = c.plan(phase)
    out.mkdir(exist_ok=False)
    from super_deli.models.hdbscan_config import HDBSCANConfig

    model = HDBSCANConfig(True)
    write_json(
        {
            "configuration": c.to_dict(),
            "density_model": model.to_dict(),
            "gmm_model": replace(
                model.preprocessing_config(),
                min_cluster_size=5,
                random_seed=c.optimizer_seed,
                n_init=10,
            ).to_dict(),
            "phase": phase,
            "plan": plan,
            "model": "frozen HDBSCAN and full-GMM/BIC; optimizer seed applies directly",
            "software": {n: version(n) for n in ("numpy", "scipy", "scikit-learn")},
        },
        out / "frozen-plan.json",
    )
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool, (out / "ledger.jsonl").open("x") as f:
        for row in pool.map(
            run_density_observation, [(str(out), c.to_dict(), row) for row in plan]
        ):
            rows.append(row)
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            if (
                retained_bytes(out.parent) > c.artifact_cap_bytes
                or shutil.disk_usage(out).free < c.min_free_bytes
            ):
                raise RuntimeError("Retained storage/free-space bound failed; evidence preserved")
    summary = {
        "status": "complete",
        "phase": phase,
        "observations": len(rows),
        "planned_pipelines": sum(2 * (1 + r["draws"]) for r in plan),
        **{
            k: sum(r["families"][m][k] for r in rows for m in ("hdbscan", "gmm"))
            for k in ("attempted", "successful", "failed", "unattempted")
        },
    }
    write_json(summary, out / "summary.json")
    return summary


def storage_gate(config, *, probe_dir, audit_file, artifact_root, output_file):
    root, probe = Path(artifact_root).resolve(), Path(probe_dir).resolve()
    audit = read_json(audit_file)
    frozen = read_json(probe / "frozen-plan.json")
    if (
        audit.get("status") != "passed"
        or audit.get("phase") != "preflight"
        or frozen["phase"] != "preflight"
        or configuration_hash(frozen["configuration"]) != configuration_hash(config.to_dict())
    ):
        raise ValueError("Matching complete audited preflight required")
    if not probe.is_relative_to(root) or not Path(output_file).resolve().is_relative_to(root):
        raise ValueError("Gate and probe must be retained under artifact_root")
    actual = retained_bytes(root)
    ratio = sum(2 * (1 + r["draws"]) for r in config.plan("main")) / sum(
        2 * (1 + r["draws"]) for r in config.plan("preflight")
    )
    projected = actual + int(retained_bytes(probe) * ratio * 1.25) + 16 * 1024**2
    passed = (
        projected <= config.artifact_cap_bytes
        and shutil.disk_usage(root).free >= config.min_free_bytes
    )
    gate = {
        "status": "passed" if passed else "failed",
        "artifact_root": str(root),
        "configuration_sha256": configuration_hash(config.to_dict()),
        "measured_probe_bytes": retained_bytes(probe),
        "retained_bytes": actual,
        "projected_bytes_with_25pct_allowance": projected,
        "artifact_cap_bytes": config.artifact_cap_bytes,
        "free_bytes": shutil.disk_usage(root).free,
        "audit_file": str(Path(audit_file).resolve()),
        "audit_sha256": fingerprint_file(audit_file).sha256,
    }
    write_json(gate, output_file)
    if not passed:
        raise RuntimeError("Measured storage gate failed; amendment required")
    return gate
