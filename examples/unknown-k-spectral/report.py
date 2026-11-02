"""Small public aggregates and figures from an independently audited local study."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save(fig, folder, name):
    for suffix in ("png", "pdf", "svg"):
        path = folder / f"{name}.{suffix}"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        if suffix == "svg":
            lines = (line.rstrip() for line in path.read_text().splitlines())
            path.write_text("\n".join(lines) + "\n")
    plt.close(fig)


def export(root, public, figure_dir):
    summary = read(root / "aggregate.json")
    assert not summary["fixture"] and summary["observations"] == 320
    assert len(summary["primary_rates"]) == 128
    main = read(root / "main-audit.json")
    assert main["passed"] and main["evaluations"] == 2592
    overview = {
        k: summary[k]
        for k in (
            "schema_version",
            "observations",
            "main_evaluations",
            "point_workflows",
            "statuses",
            "primary_interval_family",
            "interpretation",
        )
    }
    overview.update(
        independent_datasets_per_cell=20,
        paired_conditions=2,
        numerical_audit=main,
        new_pvalues=False,
        neural_training_fits=0,
    )
    rows = read(root / "point-assessments.json")
    spectra = []
    for group in summary["groups"]:
        selected = [
            r for r in rows if all(r[k] == group[k] for k in ("case", "n", "condition", "arm"))
        ]
        complete = [r for r in selected if r["status"] == "available"]
        for metric in ("splits", "merges", "noise_count"):
            values = [r[metric] for r in complete if metric in r]
            group["available_mean_" + metric] = float(np.mean(values)) if values else None
        sizes = [
            max(r["cluster_sizes"]) / min(r["cluster_sizes"])
            for r in complete
            if r["cluster_sizes"]
        ]
        group["available_mean_size_ratio"] = float(np.mean(sizes)) if sizes else None
        if group["arm"].startswith("spectral"):
            proposed = [r.get("proposed_k") for r in selected]
            group["proposed_k_counts"] = {
                str(k): proposed.count(k) for k in sorted({v for v in proposed if v is not None})
            }
            gaps = []
            values = []
            for row in selected:
                path = root / "main" / row["key"] / row["condition"] / row["arm"] / "point.npz"
                if not path.exists():
                    continue
                with np.load(path, allow_pickle=False) as a:
                    if "gaps" in a:
                        gaps.append(a["gaps"].copy())
                        values.append(a["eigenvalues"][:7].copy())
            spectra.append(
                {
                    **{k: group[k] for k in ("case", "n", "condition", "arm", "requested")},
                    "trace_available": len(gaps),
                    "gap_means": np.mean(gaps, axis=0).tolist() if gaps else None,
                    "gap_minima": np.min(gaps, axis=0).tolist() if gaps else None,
                    "gap_maxima": np.max(gaps, axis=0).tolist() if gaps else None,
                    "first_seven_eigenvalue_means": np.mean(values, axis=0).tolist()
                    if values
                    else None,
                }
            )
    write(public / "summary.json", overview)
    write(public / "all-groups.json", summary["groups"])
    write(public / "primary-rates.json", summary["primary_rates"])
    write(public / "paired-comparisons.json", summary["paired_comparisons"])
    write(public / "independent-audit.json", main)
    write(public / "gap-spectra.json", spectra)
    figure_dir.mkdir(parents=True, exist_ok=True)
    arms = ("spectral_5", "spectral_15", "gmm", "density")
    columns = (
        "Raw spectral\n5 neighbors",
        "Raw spectral\n15 neighbors",
        "PCA / GMM\nBIC, K<=4",
        "PCA / HDBSCAN\nwith noise",
    )
    plans = (
        (
            "Gaussian controls",
            ("gaussian", "gaussian_8d"),
            "nontrivial_return",
            "gaussian-controls",
        ),
        (
            "Generating-group recovery",
            ("four_spherical", "moons", "four_unequal"),
            "returned_correct_k_ari08",
            "group-recovery",
        ),
    )
    for title, cases, endpoint, filename in plans:
        labels, cells = [], []
        for case in cases:
            for n in (64, 256):
                for condition in ("clean", "noisy"):
                    labels.append(f"{case.replace('_', ' ')} / N={n} / {condition}")
                    cells.append(
                        [
                            next(
                                r
                                for r in summary["primary_rates"]
                                if (r["case"], r["n"], r["condition"], r["arm"], r["endpoint"])
                                == (case, n, condition, arm, endpoint)
                            )
                            for arm in arms
                        ]
                    )
        values = np.array([[r["rate"] for r in row] for row in cells])
        fig, ax = plt.subplots(figsize=(10, 0.43 * len(labels) + 2.5), layout="constrained")
        im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(4), columns)
        ax.set_yticks(range(len(labels)), labels)
        for i, row in enumerate(cells):
            for j, r in enumerate(row):
                available = r["requested"] - r["unavailable"]
                ax.text(
                    j,
                    i,
                    f"{r['count']}/{r['requested']} returned\n{available} available",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if r["rate"] < 0.5 else "black",
                )
        metric = (
            "Returned K>1 / all requested"
            if endpoint == "nontrivial_return"
            else "Correct returned K AND ARI>=0.8 / all requested"
        )
        ax.set_title(f"{title}\n{metric}", fontsize=11)
        fig.colorbar(im, ax=ax, label="Observed algorithm-return rate", shrink=0.65)
        fig.text(
            0.5,
            -0.10,
            "20 independent datasets/cell; conditions and arms are paired. "
            "Abstentions/failures are explicit.\n"
            "Gaussian controls assess these laws, not species error. "
            "Exact marginal/simultaneous intervals: primary-rates.json.",
            ha="center",
            fontsize=8,
        )
        save(fig, figure_dir, filename)
    write(
        figure_dir / "figure-data.json",
        {"groups": summary["groups"], "primary_rates": summary["primary_rates"]},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--figure-dir", type=Path, required=True)
    args = p.parse_args()
    export(args.root, args.output_dir, args.figure_dir)
