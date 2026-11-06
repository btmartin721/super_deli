"""Publish audited cell aggregates only; individual fit/trajectory evidence stays local."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.neural_analysis import arrays
from super_deli.benchmarking.separate_gmm_config import HEADS, NATIVE
from super_deli.benchmarking.separate_gmm_observer import evaluation_path
from super_deli.benchmarking.view_signal_report import spread
from super_deli.models.mixture_parameters import mixture_parameter_count


def read(path):
    return json.loads(path.read_text())


def aggregate_candidates(root, rows):
    out = []
    for case, n, arm in sorted({(r["case"], r["n"], r["arm"]) for r in rows}):
        selected = [r for r in rows if (r["case"], r["n"], r["arm"]) == (case, n, arm)]
        representation, head = arm.split("__")
        kind = "full" if head == "gmm_full" else "tied"
        dimension = 12 if representation == "raw_shared" else 4
        byk = []
        for k in range(1, 5):
            gains = []
            improvements = []
            eligible = 0
            completed = 0
            candidate_ari = []
            minimum_eigen = []
            k4_vs2 = []
            k4_gain_vs2 = []
            for row in selected:
                attempts = {a["k"]: a for a in row["attempts"]}
                attempt = attempts.get(k)
                if not attempt or attempt["status"] != "complete":
                    continue
                completed += 1
                p = evaluation_path(root / "main" / row["key"] / representation, 0, head)
                a = arrays(p / "fit.npz")
                labels = a[f"k{k}_labels"]
                sizes = np.bincount(labels, minlength=k)
                eligible += int(sizes.min() >= 5)
                reference = arrays(root / "main" / row["key"] / "generating-reference.npz")
                candidate_ari.append(float(adjusted_rand_score(reference["truth"], labels)))
                cov = a[f"k{k}_covariances"]
                minimum_eigen.append(float(np.linalg.eigvalsh(cov).min()))
                if 1 in attempts and attempts[1]["status"] == "complete":
                    gain = 2 * (attempt["log_likelihood"] - attempts[1]["log_likelihood"])
                    cost = (
                        mixture_parameter_count(k, dimension, kind)
                        - mixture_parameter_count(1, dimension, kind)
                    ) * np.log(n)
                    improvement = attempts[1]["bic"] - attempt["bic"]
                    np.testing.assert_allclose(improvement, gain - cost, atol=1e-05, rtol=0)
                    gains.append(gain)
                    improvements.append(improvement)
                if k == 4 and 2 in attempts and (attempts[2]["status"] == "complete"):
                    k4_vs2.append(attempts[2]["bic"] - attempt["bic"])
                    k4_gain_vs2.append(
                        2 * (attempt["log_likelihood"] - attempts[2]["log_likelihood"])
                    )
            byk.append(
                dict(
                    k=k,
                    complete_candidates=completed,
                    eligible_candidates=eligible,
                    candidate_ari=spread(candidate_ari),
                    minimum_covariance_eigenvalue=spread(minimum_eigen),
                    twice_loglik_gain_vs_k1=spread(gains),
                    bic_improvement_vs_k1=spread(improvements),
                    bic_parameter_cost_vs_k1=(
                        mixture_parameter_count(k, dimension, kind)
                        - mixture_parameter_count(1, dimension, kind)
                    )
                    * float(np.log(n)),
                    k4_bic_improvement_vs_k2=spread(k4_vs2) if k == 4 else None,
                    k4_twice_loglik_gain_vs_k2=spread(k4_gain_vs2) if k == 4 else None,
                )
            )
        out.append(
            dict(
                case=case,
                n=n,
                arm=arm,
                requested=len(selected),
                score_dimensions=dimension,
                available_searches=sum(r["status"] == "available" for r in selected),
                candidates=byk,
                interpretation=(
                    "All deltas compare K within one fixed coordinate space and covariance family"
                ),
            )
        )
    return out


def aggregate_stability(rows):
    out = []
    for case, n, arm in sorted({(r["case"], r["n"], r["arm"]) for r in rows}):
        selected = [r for r in rows if (r["case"], r["n"], r["arm"]) == (case, n, arm)]
        frequency = Counter()
        for row in selected:
            frequency.update(row["support"]["k_frequencies"])
        aris = [d["point_ari"] for r in selected for d in r["draws"] if d["point_ari"] is not None]
        out.append(
            dict(
                case=case,
                n=n,
                arm=arm,
                independent_datasets=len(selected),
                requested_draws=sum(r["requested"] for r in selected),
                attempted_pipelines=sum(r["support"]["requested"] for r in selected),
                successful_draws=sum(r["support"]["successful"] for r in selected),
                failed_draws=sum(r["support"]["failed"] for r in selected),
                selected_k_frequencies=dict(frequency),
                within_dataset_point_ari=spread(aris),
                unit_coverage=spread([r["support"]["unit_coverage"] for r in selected]),
                pair_coverage=spread([r["support"]["pair_coverage"] for r in selected]),
                coassignment=spread(
                    [r["mean_coassignment"] for r in selected if r["mean_coassignment"] is not None]
                ),
                coassignment_lower=spread(
                    [
                        r["mean_coassignment_lower"]
                        for r in selected
                        if r["mean_coassignment_lower"] is not None
                    ]
                ),
                coassignment_upper=spread(
                    [
                        r["mean_coassignment_upper"]
                        for r in selected
                        if r["mean_coassignment_upper"] is not None
                    ]
                ),
                interpretation=(
                    "One dataset per cell; fixed-seed 80% matched subsampling is conditional "
                    "sensitivity, not independent replicates or species support"
                ),
            )
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    root = parser.parse_args().study.resolve()
    repo = Path(__file__).resolve().parents[2]
    assert read(root / "main-audit.json")["passed"] and (not (root / "STOP.json").exists())
    target = Path(__file__).parent / "results"
    target.mkdir(exist_ok=False)
    for name in ("summary", "all-groups", "primary-rates", "paired-comparisons"):
        value = read(root / (name + ".json"))
        (target / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    audit = read(root / "summary.json")["audits"]
    assert all("records" not in a for a in audit.values())
    (target / "independent-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    frozen = read(root / "frozen.json")
    design = {
        k: frozen[k]
        for k in (
            "configuration",
            "settings",
            "arms",
            "training",
            "counts",
            "python",
            "platform",
            "package_version",
            "software",
        )
    }
    (target / "study-design.json").write_text(json.dumps(design, indent=2) + "\n")
    candidates = aggregate_candidates(root, read(root / "candidate-assessments.json"))
    (target / "candidate-groups.json").write_text(json.dumps(candidates, indent=2) + "\n")
    stability = aggregate_stability(read(root / "native-stability.json"))
    (target / "stability-groups.json").write_text(json.dumps(stability, indent=2) + "\n")
    groups = read(root / "all-groups.json")
    cases = (
        "morphology_only",
        "ecology_only",
        "four_complementary",
        "four_redundant",
        "gaussian_independent",
        "gaussian_correlated",
    )
    cells = [(c, n) for c in cases for n in (64, 256)]
    arms = [a + "__" + h for a in NATIVE for h in HEADS]
    counts = np.empty((9, 12), dtype=int)
    requested = counts.copy()
    failures = counts.copy()
    for i, arm in enumerate(arms):
        for j, (case, n) in enumerate(cells):
            r = next(r for r in groups if (r["case"], r["n"], r["arm"]) == (case, n, arm))
            counts[i, j] = r["endpoint_count"]
            requested[i, j] = r["requested"]
            failures[i, j] = r["failures"]
    rates = counts / requested
    mask = np.tile(np.arange(12) >= 8, (9, 1))
    fig, ax = plt.subplots(figsize=(17, 7.5))
    ax.imshow(np.ma.masked_where(mask, rates), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.imshow(np.ma.masked_where(~mask, rates), vmin=0, vmax=1, cmap="Oranges", aspect="auto")
    for i, j in np.ndindex(counts.shape):
        label = f"{counts[i, j]}/{requested[i, j]}"
        if failures[i, j]:
            label += f"\n?{failures[i, j]}"
        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="white" if rates[i, j] >= 0.75 else "black",
        )
    ax.set_yticks(
        range(9),
        [f"Seed {s} | {h}" for s in (0, 101, 202) for h in ("HDBSCAN", "Full GMM", "Shared GMM")],
    )
    names = (
        "Morphology\nonly",
        "Ecology\nonly",
        "Complementary",
        "Redundant",
        "Gaussian\nindependent",
        "Gaussian\ncorrelated",
    )
    ax.set_xticks(range(12), [f"{names[cases.index(c)]}\nN={n}" for c, n in cells], fontsize=9)
    for j in (1.5, 3.5, 5.5, 7.5, 9.5):
        ax.axvline(j, color="white", lw=2)
    ax.axvline(7.5, color="#a63b2e", lw=3)
    ax.set_title("Fresh separate encoders: group recovery and Gaussian control splitting", pad=14)
    fig.text(
        0.5,
        0.035,
        "Blue: selected K=4 and full ARI ≥0.8. Orange: K>1 Gaussian splits. "
        "Counts /requested datasets.\n? marks unavailable outcomes; "
        "intervals/failure bounds are retained. Every seed shown; no automatic winner.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.9, bottom=0.24)
    figures = repo / "docs/figures/separate-gmm"
    figures.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(figures / ("fresh-recovery-controls." + suffix), dpi=180)
    svg = figures / "fresh-recovery-controls.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    data = dict(
        arms=arms,
        cells=cells,
        counts=counts.tolist(),
        requested=requested.tolist(),
        failures=failures.tolist(),
    )
    (figures / "figure-data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                cell_groups=len(groups),
                primary_rates=len(read(root / "primary-rates.json")),
                paired_comparisons=len(read(root / "paired-comparisons.json")),
                candidate_groups=len(candidates),
                stability_groups=len(stability),
                figure_cells=counts.size,
                public_bytes=sum(p.stat().st_size for p in target.iterdir()),
            )
        )
    )


if __name__ == "__main__":
    main()
