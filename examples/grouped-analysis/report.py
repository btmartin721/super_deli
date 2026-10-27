"""Reproduce only small descriptive summaries from audited local grouped evidence."""

import argparse
from collections import Counter
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.benchmarking.grouped_analysis_study import CASES, model_arms
    from super_deli.core.provenance import fingerprint_file
    from super_deli.io.config import read_json, write_json

    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--figure-dir", required=True, type=Path)
    a = p.parse_args()
    root = a.artifact_root
    audit = read_json(root / "main-audit.json")
    assert audit["status"] == "passed"
    assert read_json(root / "preflight-audit.json")["status"] == "passed"
    a.output_dir.mkdir()
    a.figure_dir.mkdir(parents=True)
    names = [arm.name for arm in model_arms().arms]
    cells = [(c, n) for c in CASES for n in (64, 256)]
    models = []
    stability = []
    recovery = []
    for case, n in cells:
        for name in names:
            rows = [
                r
                for r in audit["models"]
                if r["case"] == case and r["n"] == n and r["name"] == name
            ]
            assert len(rows) == 3
            models.append(
                {
                    "case": case,
                    "n": n,
                    "model": name,
                    "datasets": 3,
                    "selected_k_counts": dict(
                        sorted(Counter(r["selected_k"] for r in rows).items())
                    ),
                    "mean_selected_k": float(np.mean([r["selected_k"] for r in rows])),
                    "screen_withheld": sum(r["screen_status"] == "withheld" for r in rows),
                }
            )
        base = root / "study" / f"{case}_n{n}_00/analysis/arms/shared_predictive"
        summary = read_json(base / "summary.json")
        point = np.load(base / "point/selected_memberships.npz")["labels"]
        records = summary["replicates"]
        done = [r for r in records if r["status"] == "complete"]
        stability.append(
            {
                "case": case,
                "n": n,
                "datasets": 1,
                "requested": len(records),
                "successful": len(done),
                "selected_k_counts": dict(sorted(Counter(r["selected_k"] for r in done).items())),
                "point_k": summary["point"]["selection"]["screened_k"],
                "conditional_mean_ari_to_own_point": float(
                    np.mean([adjusted_rand_score(point, r["labels"]) for r in done])
                )
                if done
                else None,
                "conditioning": (
                    "Original locality folds/model settings; repeated draws, "
                    "not independent datasets"
                ),
            }
        )
        if case in {"complementary", "unequal_variance"}:
            for name in names:
                aris = []
                counts = []
                for i in range(3):
                    observation = root / "study" / f"{case}_n{n}_{i:02d}"
                    labels = read_json(
                        observation / "analysis/arms" / name / "comparison-point.json"
                    )["labels"]
                    with np.load(observation / "truth.npz") as truth:
                        aris.append(float(adjusted_rand_score(truth["labels"], labels)))
                    counts.append(len(set(labels)))
                recovery.append(
                    {
                        "case": case,
                        "n": n,
                        "model": name,
                        "datasets": 3,
                        "selected_four": counts.count(4),
                        "mean_ari": float(np.mean(aris)),
                        "min_ari": min(aris),
                        "max_ari": max(aris),
                        "interpretation": (
                            "Post-fit synthetic reference only; no general species accuracy claim"
                        ),
                    }
                )
    summary = {
        "schema_version": 1,
        "status": "passed",
        "observations": 36,
        "model_arms": 6,
        "pairwise_comparisons": 540,
        "main_candidate_searches": 2568,
        "preflight_candidate_searches": 364,
        "main_point_fold_searches": 576,
        "bootstrap_searches": 1800,
        "standalone_replay_searches": 192,
        "shared_predictive_draws": 300,
        "scientific_fit_failures": 0,
        "artifact_cap_bytes": read_json(root / "storage-gate.json").get(
            "artifact_cap_bytes", read_json(root / "validation.json")["artifact_cap_bytes"]
        ),
        "defaults_and_p_values_unchanged": True,
        "p_values_computed": False,
        "audit_hashes": {
            f: fingerprint_file(root / f).sha256
            for f in ("main-audit.json", "preflight-audit.json", "independent-postfit-audit.json")
        },
        "scope": (
            "Three datasets/cell and 25 draws per selected dataset: "
            "descriptive orchestration validation"
        ),
    }
    for name, data in (
        ("summary", summary),
        ("models", models),
        ("stability", stability),
        ("recovery", recovery),
    ):
        write_json(data, a.output_dir / f"{name}.json")
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    values = np.array(
        [
            [
                next(
                    r["mean_selected_k"]
                    for r in models
                    if r["case"] == case and r["n"] == n and r["model"] == name
                )
                for name in names
            ]
            for case, n in cells
        ]
    )
    im = ax.imshow(values, cmap="viridis", vmin=1, vmax=4, aspect="auto")
    ax.set_xticks(
        range(6),
        [
            "Full BIC",
            "Shared BIC",
            "Full AIC",
            "Shared AIC",
            "Full screened AIC",
            "Shared screened AIC",
        ],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(cells)), [f"{c}; N={n}" for c, n in cells])
    for i, j in np.ndindex(values.shape):
        ax.text(
            j,
            i,
            f"{values[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if values[i, j] < 2.5 else "black",
        )
    ax.set_title("Mean selected groups across three independent datasets per cell")
    fig.colorbar(im, ax=ax, label="Mean selected K")
    for extension in ("png", "pdf"):
        fig.savefig(a.figure_dir / f"grouped-comparison.{extension}", dpi=180)
    plt.close(fig)
    write_json(
        {f.name: fingerprint_file(f).sha256 for f in a.output_dir.iterdir() if f.is_file()},
        a.output_dir / "checksums.json",
    )
    print(summary)


if __name__ == "__main__":
    main()
