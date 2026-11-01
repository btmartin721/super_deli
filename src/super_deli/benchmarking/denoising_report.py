"""Paired pilot summaries; generating labels are used only after all fitting."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import beta

from super_deli.benchmarking.denoising_study import ARMS, CASES, CONDITIONS, assert_frozen, tasks
from super_deli.benchmarking.moons_geometry import graph_diagnostics
from super_deli.benchmarking.neural_analysis import arrays, neural_forward, require
from super_deli.benchmarking.neural_moons_coordinates import neighborhood_metrics
from super_deli.benchmarking.neural_moons_fit import recovery
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def interval(success, total, alpha=0.05):
    if not 0 <= success <= total or total < 1:
        raise ValueError("Binomial interval requires valid counts")
    return [
        float(beta.ppf(alpha / 2, success, total - success + 1)) if success else 0.0,
        float(beta.ppf(1 - alpha / 2, success + 1, total - success)) if success < total else 1.0,
    ]


def noise_ari(a, b):
    from sklearn.metrics import adjusted_rand_score

    def resolved(labels):
        labels = np.array(labels).copy()
        missing = labels < 0
        labels[missing] = np.arange(sum(missing)) + max(int(labels.max()), 0) + 1
        return labels

    return float(adjusted_rand_score(resolved(a), resolved(b)))


def rate(rows):
    done = [r for r in rows if r["status"] == "complete"]
    success = sum(r["correct_k"] for r in done)
    n = len(rows)
    failure = n - len(done)
    return {
        "requested": n,
        "resolved": len(done),
        "failed": failure,
        "correct_k": success,
        "resolved_correct_k_rate": success / len(done) if done else None,
        "failure_bounds": [success / n, (success + failure) / n],
        "simultaneous_95_interval": [
            interval(success, n, 0.05 / 60)[0],
            interval(success + failure, n, 0.05 / 60)[1],
        ],
        "mean_ari": float(np.mean([r["ari"] for r in done])) if done else None,
        "mean_coverage": float(np.mean([r["coverage"] for r in done])) if done else None,
        "k_counts": dict(Counter(str(r["selected_k"]) for r in done)),
    }


def paired(rows):
    n = len(rows)
    done = [(a, b) for a, b in rows if a["status"] == b["status"] == "complete"]
    # First arm is denoising. Discordant outcomes are mutually exclusive.
    p10 = sum(a["correct_k"] and not b["correct_k"] for a, b in done)
    p01 = sum(b["correct_k"] and not a["correct_k"] for a, b in done)
    unresolved = n - len(done)
    lo10 = interval(p10, n, 0.05 / 60)[0]
    hi10 = interval(p10 + unresolved, n, 0.05 / 60)[1]
    lo01 = interval(p01, n, 0.05 / 60)[0]
    hi01 = interval(p01 + unresolved, n, 0.05 / 60)[1]
    return {
        "requested_pairs": n,
        "resolved_pairs": len(done),
        "failed_pairs": unresolved,
        "denoising_only_correct": p10,
        "ordinary_only_correct": p01,
        "resolved_difference": (p10 - p01) / len(done) if done else None,
        "failure_bounds": [(p10 - p01 - unresolved) / n, (p10 - p01 + unresolved) / n],
        "simultaneous_95_difference_interval": [lo10 - hi01, hi10 - lo01],
    }


def report(root):
    assert_frozen(root)
    audit = read_json(root / "main-audit.json")
    require(audit["passed"], "Main audit must pass before reporting")
    rows, geometry, stability, seeds = [], [], [], []
    for task in tasks("main"):
        out = root / "main" / task["key"]
        reference = arrays(out / "generating-reference.npz")
        records = read_json(out / "fits.json")
        truth = reference["truth"]
        source = reference["z"]
        points = {}
        for r in records:
            if r["kind"] != "point":
                continue
            row = {
                "key": task["key"],
                "case": task["case"],
                "n": task["n"],
                "condition": r["condition"],
                "arm": r["arm"],
                "status": r["status"],
            }
            if r["status"] == "complete":
                labels = np.array(r["labels"])
                row.update(recovery(labels, truth))
                row["correct_k"] = row["selected_k"] == len(set(truth))
                points[(r["condition"], r["arm"])] = labels
                a = arrays(out / r["path"] / "point.npz")
                g = {k: row[k] for k in ("key", "case", "n", "condition", "arm")}
                if r["arm"] == "source_reference":
                    observed = source
                else:
                    observed = reference["clean"] + (
                        reference["noise"] if r["condition"] == "noisy" else 0
                    )
                context = {
                    "truth": truth,
                    "source_standard": (source - source.mean(0)) / source.std(0),
                    "traits_standard": (observed - observed.mean(0)) / observed.std(0),
                }
                g["neighborhood"], _ = neighborhood_metrics(a["scores"], context)
                g["graph"], _, _ = graph_diagnostics(a["scores"], truth, 5)
                g["score_singular_values"] = np.linalg.svd(
                    a["scores"] - a["scores"].mean(0), compute_uv=False
                ).tolist()
                if r["arm"].startswith(("neural_", "denoising_")):
                    standardized = (observed - a["means"]) / a["scales"]
                    raw = neural_forward(standardized, a, "encoder")
                    residual = (neural_forward(raw, a, "decoder") - standardized) ** 2
                    g.update(
                        raw_latent_scales=a["latent_scales"].tolist(),
                        raw_latent_singular_values=np.linalg.svd(
                            raw - raw.mean(0), compute_uv=False
                        ).tolist(),
                        mse_by_feature=residual.mean(0).tolist(),
                        mse_by_generating_group={
                            str(int(k)): float(residual[truth == k].mean()) for k in set(truth)
                        },
                        clean_initial_final_mse=[float(a["losses"][0]), float(a["losses"][-1])],
                    )
                geometry.append(g)
            rows.append(row)
        for condition in CONDITIONS:
            for arm in ARMS:
                s = read_json(out / condition / f"{arm}-support.json")
                draws = [
                    r
                    for r in records
                    if r["condition"] == condition and r["arm"] == arm and r["kind"] == "draw"
                ]
                metrics = []
                for d in draws:
                    if d["status"] == "complete":
                        ix = np.array(d["training_indices"])
                        metric = recovery(np.array(d["labels"]), truth[ix])
                        if (condition, arm) in points:
                            metric["point_agreement"] = noise_ari(
                                d["labels"], points[(condition, arm)][ix]
                            )
                        metrics.append(metric)
                if draws:
                    stability.append(
                        {
                            "key": task["key"],
                            "case": task["case"],
                            "n": task["n"],
                            "condition": condition,
                            "arm": arm,
                            "native": s,
                            "mean_draw_ari": float(np.mean([d["ari"] for d in metrics]))
                            if metrics
                            else None,
                            "mean_draw_coverage": float(np.mean([d["coverage"] for d in metrics]))
                            if metrics
                            else None,
                            "mean_point_agreement": float(
                                np.mean(
                                    [
                                        d["point_agreement"]
                                        for d in metrics
                                        if "point_agreement" in d
                                    ]
                                )
                            )
                            if (condition, arm) in points and metrics
                            else None,
                        }
                    )
            for family in ("neural", "denoising"):
                for a, b in ((0, 101), (0, 202), (101, 202)):
                    keys = [(condition, f"{family}_{s}") for s in (a, b)]
                    if all(k in points for k in keys):
                        seeds.append(
                            {
                                "key": task["key"],
                                "condition": condition,
                                "family": family,
                                "seeds": [a, b],
                                "full_unit_ari": noise_ari(points[keys[0]], points[keys[1]]),
                            }
                        )
    primary = []
    differences = []
    for case in CASES:
        for n in (64, 256):
            for condition in CONDITIONS:
                selected = [
                    r
                    for r in rows
                    if r["case"] == case and r["n"] == n and r["condition"] == condition
                ]
                for arm in ("density", "neural_0", "denoising_0"):
                    primary.append(
                        {
                            "case": case,
                            "n": n,
                            "condition": condition,
                            "arm": arm,
                            **rate([r for r in selected if r["arm"] == arm]),
                        }
                    )
                pairs = []
                for task in tasks("main"):
                    if task["case"] == case and task["n"] == n:
                        pair = [
                            next(r for r in selected if r["key"] == task["key"] and r["arm"] == a)
                            for a in ("denoising_0", "neural_0")
                        ]
                        pairs.append(pair)
                differences.append({"case": case, "n": n, "condition": condition, **paired(pairs)})
    require(len(primary) == 36 and len(differences) == 12, "Primary outcome accounting")
    summary = {
        "schema_version": 1,
        "independent_observations": 60,
        "paired_tables": 120,
        "main_audit_sha256": fingerprint_file(root / "main-audit.json").sha256,
        "primary_rates": primary,
        "primary_paired_differences": differences,
        "point_failures": sum(r["status"] != "complete" for r in rows),
        "simultaneous_uncertainty": (
            "60 exact binomial intervals; Bonferroni alpha=0.05/60; "
            "paired bounds from discordant categories"
        ),
        "interpretation": [
            "Ten independent observations/cell; broad pilot uncertainty",
            "Clean means no extra feature noise; source moons remain noisy",
            "Measurement noise is raw-scale; training corruption is standardized-scale",
            "Correct K does not imply recovered groups or species boundaries",
            "Seeds, paired tables and subsamples are repeated measurements",
            "No new p-values, best seed or automatic default change",
        ],
    }
    for name, value in (
        ("point-assessments.json", rows),
        ("geometry-assessments.json", geometry),
        ("stability-assessments.json", stability),
        ("initialization-agreement.json", seeds),
        ("aggregate.json", summary),
    ):
        write_json(value, root / name)
    print(
        {
            "point_rows": len(rows),
            "geometry_rows": len(geometry),
            "primary_rates": 36,
            "paired_differences": 12,
        },
        flush=True,
    )
    return summary
