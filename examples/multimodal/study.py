"""Frozen bounded multimodal orchestration study; detailed outputs remain local."""

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from synthetic import invented, write_shape

from super_deli.benchmarking.multimodal_validation import (
    equal_point_arrays,
    observe_shape_refits,
    selected_density,
)
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json
from super_deli.io.matched_analysis import verify_sources
from super_deli.workflows.genomic_analysis_adapters import run_genomic_arm
from super_deli.workflows.multimodal_analysis import _read_arm, run_multimodal_analysis

LAWS = (
    "continuous_control",
    "redundant",
    "complementary",
    "conflicting",
    "genomic_only",
    "shape_only",
)
SETTINGS = dict(
    schema_version=1,
    main_seeds=list(range(5100, 5124)),
    preflight_seeds=[5000, 5001],
    laws=list(LAWS),
    n=64,
    loci=64,
    traits_per_table=4,
    landmarks=6,
    dimensions=2,
    n_pcs=2,
    max_k=4,
    min_cluster_size=5,
    n_init=10,
    model_seed=301,
    stability_draws=3,
    cap_bytes=1024**3,
    free_reserve=8 * 1024**3,
    serial_seconds=1800,
    allowance=1.25,
    geometry_rtol=1e-7,
    geometry_atol=1e-9,
)


def geometry_checks(root, dataset, config, baseline, cohort):
    collection = read_json(baseline / "inputs/shape-ingestion/modality_000/landmarks.json")
    records = collection["records"]
    raw = np.asarray([r["coordinates"] for r in records])
    n = len(raw)
    rng = np.random.default_rng(931007)
    angle = rng.uniform(-np.pi, np.pi, n)
    rotation = np.stack(
        (
            np.stack((np.cos(angle), -np.sin(angle)), axis=1),
            np.stack((np.sin(angle), np.cos(angle)), axis=1),
        ),
        axis=1,
    )
    rotations = np.einsum("nld,ndk->nlk", raw, rotation)
    scales = rng.uniform(0.4, 4, n)[:, None, None]
    shifts = rng.normal(0, 4, (n, 1, 2))
    variants = dict(
        translation=raw + shifts,
        rotation=rotations,
        scale=raw * scales,
        all=rotations * scales + shifts,
    )
    results = []
    for name, values in variants.items():
        source = root / (name + "-source")
        shutil.copytree(baseline / "inputs", source)
        write_shape(
            source / "transformed.tps",
            values,
            [r["observation_id"] for r in records],
            collection["landmark_ids"],
        )
        transformed = replace(
            dataset,
            modalities=tuple(
                replace(m, path="transformed.tps") if m.kind == "morphometric" else m
                for m in dataset.modalities
            ),
        )
        for arm in (a for a in config.arms if a.name in {"shape", "combined"}):
            point_arm = replace(arm, model=replace(arm.model, n_bootstrap=0))
            output = root / (name + "-" + arm.name)
            run_genomic_arm(point_arm, transformed, source, output)
            point = _read_arm(point_arm, output, cohort, transformed)["point"]
            original = read_json(baseline / "arms" / arm.name / "comparison-point.json")
            np.testing.assert_allclose(
                pdist(point["scores"]),
                pdist(original["scores"]),
                rtol=SETTINGS["geometry_rtol"],
                atol=SETTINGS["geometry_atol"],
            )
            a, b = np.asarray(point["labels"]), np.asarray(original["labels"])
            assert np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])
            results.append(dict(variant=name, arm=arm.name, passed=True))
    return results


def study(root, phase):
    root = Path(root)
    root.mkdir(exist_ok=True)
    frozen = root / "settings.json"
    if frozen.exists():
        assert read_json(frozen) == SETTINGS
    else:
        write_json(SETTINGS, frozen)
    if phase == "main":
        assert read_json(root / "resource-gate.json")["passed"]
    out = root / phase
    out.mkdir()
    start = time.monotonic()
    rows = []
    density = []
    events = []
    parity = 0
    geometry = []
    seeds = SETTINGS["preflight_seeds"] if phase == "preflight" else SETTINGS["main_seeds"]
    for index, seed in enumerate(seeds):
        law = LAWS[index % 6] if phase == "main" else ("continuous_control", "complementary")[index]
        draws = 3 if phase == "preflight" or index < 6 else 0
        source = out / f"{index:02d}-{law}-source"
        d, c = invented(source, seed=seed, law=law, draws=draws)
        destination = out / f"{index:02d}-{law}-analysis"
        with observe_shape_refits() as captured:
            s = run_multimodal_analysis(
                d, analysis_config=c, base_dir=source, output_dir=destination
            )
        write_json(captured, out / f"{index:02d}-shape-refits.json")
        events.extend(captured)
        assert all(r["status"] == "complete" for r in captured), "Numerical shape audit failed"
        manifest = read_json(destination / "manifest.json")
        dataset = config_from_dict(manifest["frozen_dataset"])
        cohort = manifest["cohort"]
        verify_sources(destination / "inputs", manifest["sources"])
        truth = read_json(source / "truth-evaluation-only.json")
        labels = dict(zip(truth["unit_ids"], truth["labels"], strict=True))
        for arm, row in zip(c.arms, s["models"], strict=True):
            record = dict(
                index=index,
                seed=seed,
                law=law,
                name=arm.name,
                status=row["status"],
                selected_k=row["selected_k"],
                error=row["error"],
                latent_label_ari=None,
            )
            if row["status"] != "failed":
                native = destination / "arms" / arm.name
                point = _read_arm(arm, native, cohort, dataset)["point"]
                record["latent_label_ari"] = float(
                    adjusted_rand_score([labels[u] for u in cohort["unit_ids"]], point["labels"])
                )
                point_arm = replace(arm, model=replace(arm.model, n_bootstrap=0))
                replay = out / f"{index:02d}-{law}-replay-{arm.name}"
                run_genomic_arm(point_arm, dataset, destination / "inputs", replay)
                assert point == _read_arm(point_arm, replay, cohort, dataset)["point"]
                parity += equal_point_arrays(native, replay)
                density.append(selected_density(native, arm))
                assert row["stability"]["requested"] == draws
            rows.append(record)
        for pair in s["comparisons"]:
            if pair["status"] == "available":
                a = read_json(destination / "arms" / pair["left"] / "comparison-point.json")[
                    "labels"
                ]
                b = read_json(destination / "arms" / pair["right"] / "comparison-point.json")[
                    "labels"
                ]
                np.testing.assert_allclose(
                    pair["adjusted_rand"], adjusted_rand_score(a, b), atol=1e-14
                )
        if phase == "preflight" and index == 1:
            geometry = geometry_checks(out, dataset, c, destination, cohort)
        assert s["inference"]["p_value"] is None
        print(
            json.dumps(
                dict(phase=phase, completed=index + 1, total=len(seeds), status=s["status"])
            ),
            flush=True,
        )
    elapsed = time.monotonic() - start
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    result = dict(
        passed=True,
        phase=phase,
        observations=len(seeds),
        point_pipelines=10 * len(seeds),
        bootstrap_refits=30 if phase == "preflight" else 90,
        geometry=geometry,
        records=rows,
        density_audits=density,
        exact_parity_arrays=parity,
        audited_gpa_refits=len(events),
        elapsed_seconds=elapsed,
        retained_bytes=size,
        failed_points=sum(r["status"] == "failed" for r in rows),
    )
    write_json(result, out / "summary.json")
    if phase == "preflight":
        gate = dict(
            passed=1.25 * 12 * elapsed < 1800
            and 1.25 * 12 * size + 32 * 1024**2 < 1024**3
            and shutil.disk_usage(root).free >= 8 * 1024**3,
            projected_serial_seconds=1.25 * 12 * elapsed,
            projected_bytes=int(1.25 * 12 * size + 32 * 1024**2),
            free_bytes=shutil.disk_usage(root).free,
        )
        write_json(gate, root / "resource-gate.json")
        assert gate["passed"], gate
    else:
        assert elapsed < 1800 and size < 1024**3 and shutil.disk_usage(root).free >= 8 * 1024**3
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "main"), required=True)
    args = parser.parse_args()
    study(args.output_dir, args.phase)
