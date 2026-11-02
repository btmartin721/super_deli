"""Export small descriptive aggregates and figures from an audited local study."""

import argparse
import json
from collections import Counter
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


def export(root, destination):
    summary = read(root / "aggregate.json")
    assert not summary["fixture"] and summary["selected_k"] is None
    rows = read(root / "point-assessments.json")
    baselines = read(root / "density-assessments.json")
    groups = []
    for group in summary["groups"]:
        selected = [
            r
            for r in rows
            if all(r[k] == group[k] for k in ("case", "n", "space", "neighbors", "target_k"))
        ]
        done = [r for r in selected if r["status"] == "available"]
        derived = {}
        for name in ("splits", "merges", "normalized_cut", "max_conductance", "oracle_edge_mixing"):
            derived["available_mean_" + name] = (
                float(np.mean([r[name] for r in done])) if done else None
            )
        sizes = [max(r["cluster_sizes"]) / min(r["cluster_sizes"]) for r in done]
        derived["available_mean_size_ratio"] = float(np.mean(sizes)) if sizes else None
        derived["component_partitions"] = sum(r.get("method") == "components" for r in done)
        derived["graph_component_counts"] = dict(
            Counter(str(r["graph_components"]) for r in selected)
        )
        groups.append({**group, **derived})
    density = []
    for case in ("gaussian", "four_spherical", "moons"):
        for n in (64, 256):
            for space in dict.fromkeys(r["space"] for r in baselines):
                selected = [
                    r for r in baselines if (r["case"], r["n"], r["space"]) == (case, n, space)
                ]
                done = [r for r in selected if r["status"] == "complete"]
                density.append(
                    {
                        "case": case,
                        "n": n,
                        "space": space,
                        "requested": len(selected),
                        "complete": len(done),
                        "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in done)),
                        **{
                            "mean_" + name: float(np.mean([r[name] for r in done]))
                            if done
                            else None
                            for name in ("ari", "coverage", "noise_count", "splits", "merges")
                        },
                        "assigned_ari_available": sum(r["assigned_ari"] is not None for r in done),
                        "mean_assigned_ari": float(
                            np.mean(
                                [r["assigned_ari"] for r in done if r["assigned_ari"] is not None]
                            )
                        )
                        if any(r["assigned_ari"] is not None for r in done)
                        else None,
                    }
                )
    spectra = []
    for folder in sorted((root / "main/graphs").iterdir()):
        record = read(folder / "result.json")
        if record["job"]["kind"] != "point":
            continue
        with np.load(folder / "arrays.npz", allow_pickle=False) as a:
            values = a["eigenvalues"]
            spectra.append(
                {
                    **record["job"],
                    "n": record["n"],
                    "components": record["components"],
                    "first_five_eigenvalues": values[:5].tolist(),
                    "K2_cutoff_gap": float(values[2] - values[1]),
                    "K4_cutoff_gap": float(values[4] - values[3]),
                }
            )
    # Export distributions, not per-specimen coordinates, labels or learned weights.
    spectrum_groups = []
    for group in groups:
        if group["target_k"] != 2:
            continue
        # Case names contain underscores; bind keys explicitly instead of parsing.
        keys = {r["key"] for r in rows if r["case"] == group["case"]}
        selected = [
            r
            for r in spectra
            if r["key"] in keys and all(r[k] == group[k] for k in ("n", "space", "neighbors"))
        ]
        spectrum_groups.append(
            {
                **{k: group[k] for k in ("case", "n", "space", "neighbors")},
                "graphs": len(selected),
                "eigenvalue_mean": np.mean(
                    [r["first_five_eigenvalues"] for r in selected], axis=0
                ).tolist(),
                **{
                    name + "_range": [
                        min(r[name] for r in selected),
                        max(r[name] for r in selected),
                    ]
                    for name in ("K2_cutoff_gap", "K4_cutoff_gap")
                },
            }
        )
    audit = read(root / "main-audit.json")
    overview = {
        k: summary[k]
        for k in (
            "schema_version",
            "source_observations",
            "supplied_k",
            "selected_k",
            "neural_training_fits",
            "main_evaluations",
            "partitions",
            "interpretation",
        )
    }
    overview["independent_datasets_per_cell"] = 10
    overview["numerical_audit"] = audit
    overview["withheld_reasons"] = dict(
        Counter(r.get("reason") for r in rows if r["status"] != "available")
    )
    write(destination / "summary.json", overview)
    write(destination / "all-settings.json", groups)
    write(destination / "density-baselines.json", density)
    write(destination / "spectra.json", spectrum_groups)
    write(destination / "independent-audit.json", audit)
    return groups, density


def figures(groups, density, folder):
    folder.mkdir(parents=True, exist_ok=True)
    spaces = list(dict.fromkeys(r["space"] for r in groups))
    titles = ["Generating 2D (oracle)"] + [
        f"{condition}: {name}"
        for condition in ("clean", "noisy")
        for name in (
            "traits 8D",
            "PCA 2D",
            "AE seed 0",
            "AE seed 101",
            "AE seed 202",
            "DAE seed 0",
            "DAE seed 101",
            "DAE seed 202",
        )
    ]
    for case, target, label in (("moons", 2, "moons-K2"), ("four_spherical", 4, "four-groups-K4")):
        matrix = np.full((17, 6), np.nan)
        annotations = np.empty((17, 6), dtype=object)
        for i, space in enumerate(spaces):
            for j, (n, neighbors) in enumerate(((64, 5), (64, 15), (256, 5), (256, 15))):
                g = next(
                    g
                    for g in groups
                    if (g["case"], g["n"], g["space"], g["neighbors"], g["target_k"])
                    == (case, n, space, neighbors, target)
                )
                value = g["available_mean_ari"]
                matrix[i, j] = value if value is not None else np.nan
                annotations[i, j] = (
                    f"{value:.2f}\n{g['available']}/10" if value is not None else "withheld\n0/10"
                )
            for j, n in enumerate((64, 256), 4):
                g = next(g for g in density if (g["case"], g["n"], g["space"]) == (case, n, space))
                matrix[i, j] = g["mean_ari"]
                annotations[i, j] = f"{g['mean_ari']:.2f}\n{g['mean_coverage']:.0%} cov."
        fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#e5e5e5")
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_yticks(range(17), titles)
        ax.set_xticks(
            range(6),
            ["N64 / k5", "N64 / k15", "N256 / k5", "N256 / k15", "HDB N64", "HDB N256"],
            rotation=30,
            ha="right",
        )
        for i in range(17):
            for j in range(6):
                ax.text(
                    j,
                    i,
                    annotations[i, j],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if np.isfinite(matrix[i, j]) and matrix[i, j] < 0.5 else "black",
                )
        ax.set_title(
            f"{'Two moons' if target == 2 else 'Four spherical groups'}: supplied K={target}\n"
            "Available-only mean ARI / available requests; "
            "HDBSCAN shows full-unit ARI / coverage",
            fontsize=11,
        )
        fig.colorbar(im, ax=ax, label="Agreement with generating membership (ARI)", shrink=0.65)
        fig.text(
            0.5,
            -0.025,
            "Retrospective: 10 datasets per cell. Supplied K is diagnostic; "
            "no K discovery or species claim.\nUnavailable requests are retained in "
            "bounds in all-settings.json; seeds and conditions are paired.",
            ha="center",
            fontsize=8,
        )
        for extension in ("png", "pdf", "svg"):
            path = folder / f"{label}.{extension}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            if extension == "svg":
                lines = (line.rstrip() for line in path.read_text().splitlines())
                path.write_text("\n".join(lines) + "\n")
        plt.close(fig)
    write(folder / "figure-data.json", {"settings": groups, "density": density})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    groups, density = export(args.root, args.output_dir)
    figures(groups, density, args.figure_dir)
