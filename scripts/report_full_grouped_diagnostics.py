"""Export audited diagnostic tables and a static scientific figure without fitting."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot(audit, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = read(audit / "summary.json")
    arms = ("baseline", "pcs_2", "pcs_3", "starts_100", "shared_4", "source_4")
    names = (
        "Full / 4 PCs",
        "Full / 2 PCs",
        "Full / 3 PCs",
        "Full / 100 starts",
        "Shared / 4 PCs",
        "Source-row / full",
    )
    cohorts = ("archived", "fresh_64", "fresh_256")
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for index, (cohort, color) in enumerate(zip(cohorts, colors, strict=True)):
        x = np.arange(len(arms)) + (index - 1) * 0.2
        selected = [
            next(r for r in summary["summaries"] if r["cohort"] == cohort and r["arm"] == arm)
            for arm in arms
        ]
        rates = np.array(
            [
                r["selected_k_counts"]["4"] / r["available"] if r["available"] else np.nan
                for r in selected
            ]
        )
        intervals = np.array([r["selected_four_wilson_95"] or [np.nan, np.nan] for r in selected]).T
        axes[0, 0].errorbar(
            x,
            rates,
            yerr=np.maximum(0, [rates - intervals[0], intervals[1] - rates]),
            fmt="o",
            color=color,
            capsize=2,
            label=cohort.replace("_", " "),
        )
        axes[0, 1].errorbar(
            x,
            [r["ari"]["mean"] if r["ari"] else np.nan for r in selected],
            yerr=[r["ari"]["mean_mcse"] if r["ari"] else np.nan for r in selected],
            fmt="o",
            color=color,
            capsize=2,
        )
    for ax, title in zip(
        axes[0],
        ("Four-component selection (Wilson 95%)", "Generating-label agreement (mean ± MCSE)"),
        strict=True,
    ):
        ax.set_xticks(range(6), names, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(-0.07, 1.08)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    tails = [r for r in rows(audit / "retained_tails.csv") if r["case"] == "complementary"]
    x = np.arange(len(tails))
    axes[1, 0].bar(
        x,
        [float(r["observed_statistic"]) for r in tails],
        color=["#D55E00" if r["reject"] == "True" else "#0072B2" for r in tails],
    )
    axes[1, 0].plot(
        x,
        [float(r["null_max"]) for r in tails],
        "k_",
        label="Maximum of 99 retained null statistics",
    )
    axes[1, 0].set(
        title="Archived baseline: observed departure statistic",
        xlabel="Archived observation",
        ylabel="BIC improvement over K=1",
    )
    axes[1, 0].legend(fontsize=7)
    candidates = rows(audit / "candidates.csv")
    pairs = []
    for i in range(len(tails)):
        pair = [
            r
            for r in candidates
            if r["cohort"] == "archived"
            and r["arm"] == "baseline"
            and int(r["repetition"]) == i
            and int(r["k"]) in (1, 4)
        ]
        one, four = sorted(pair, key=lambda r: int(r["k"]))
        if one["bic"] and four["bic"]:
            pairs.append(
                (
                    i,
                    2
                    * (
                        float(four["log_likelihood_from_bic"])
                        - float(one["log_likelihood_from_bic"])
                    ),
                    float(four["penalty"]) - float(one["penalty"]),
                )
            )
    axes[1, 1].scatter(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        label="Twice fitted log-likelihood gain",
        color="#0072B2",
    )
    axes[1, 1].plot(
        [p[0] for p in pairs],
        [p[2] for p in pairs],
        color="#D55E00",
        label="Additional BIC penalty",
    )
    axes[1, 1].set(title="Archived baseline: K=4 versus K=1", xlabel="Archived observation")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(
        "Grouped full-covariance diagnostics · fixed settings, no recalibration", fontsize=13
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination / f"full-grouped-diagnostics.{suffix}", dpi=180)
    svg = destination / "full-grouped-diagnostics.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def mechanisms(root):
    import numpy as np

    def describe(values):
        return (
            {
                "n": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(min(values)),
                "max": float(max(values)),
            }
            if values
            else None
        )

    candidates = rows(root / "audit/candidates.csv")
    points = read(root / "study/points.json")
    oracles = rows(root / "audit/truth_initialized.csv")
    result = {"candidate_contrasts": [], "signal": [], "oracles": [], "paired_scores": []}
    for cohort in ("archived", "fresh_64", "fresh_256"):
        for arm in ("baseline", "pcs_2", "pcs_3", "starts_100", "shared_4", "source_4"):
            selected = [r for r in candidates if r["cohort"] == cohort and r["arm"] == arm]
            for lower_k in (1, 2):
                gains, costs, gaps = [], [], []
                for i in sorted({r["repetition"] for r in selected}):
                    one = next(
                        r for r in selected if r["repetition"] == i and int(r["k"]) == lower_k
                    )
                    four = next(r for r in selected if r["repetition"] == i and int(r["k"]) == 4)
                    if one["bic"] and four["bic"]:
                        gains.append(
                            2
                            * (
                                float(four["log_likelihood_from_bic"])
                                - float(one["log_likelihood_from_bic"])
                            )
                        )
                        costs.append(float(four["penalty"]) - float(one["penalty"]))
                        gaps.append(float(four["bic"]) - float(one["bic"]))
                result["candidate_contrasts"].append(
                    {
                        "cohort": cohort,
                        "arm": arm,
                        "lower_k": lower_k,
                        "twice_ll_gain": describe(gains),
                        "extra_penalty": describe(costs),
                        "bic_gap": describe(gaps),
                        "scope": (
                            "All converged logged candidates, including small components; "
                            "within one score space"
                        ),
                    }
                )
            fitted = [
                r
                for r in points
                if r["cohort"] == cohort and r["arm"] == arm and r["status"] == "complete"
            ]
            binary = [r for r in fitted if r["selected_k"] == 2]
            result["signal"].append(
                {
                    "cohort": cohort,
                    "arm": arm,
                    "k2_axis_homogeneity": {
                        axis: describe([r["metrics"]["axis_homogeneity"][axis] for r in binary])
                        for axis in ("a", "b")
                    },
                    "morphology_raw_a": describe(
                        [
                            r["metrics"]["upstream"]["morphological"]["raw_group_axis_r2"]["a"]
                            for r in fitted
                        ]
                    ),
                    "ecology_raw_b": describe(
                        [
                            r["metrics"]["upstream"]["ecological"]["raw_group_axis_r2"]["b"]
                            for r in fitted
                        ]
                    ),
                }
            )
        for name in ("four", "axis_a", "axis_b"):
            selected = [
                r
                for r in oracles
                if r["cohort"] == cohort and r["partition"] == name and r["status"] == "complete"
            ]
            result["oracles"].append(
                {
                    "cohort": cohort,
                    "partition": name,
                    "available": len(selected),
                    "eligible": sum(r["eligible"] == "True" for r in selected),
                    "bic_minus_selected": describe(
                        [float(r["bic_minus_selected"]) for r in selected]
                    ),
                    "bic_minus_same_k": describe(
                        [float(r["bic_minus_unsupervised_same_k"]) for r in selected]
                    ),
                    "scope": (
                        "Separate truth-based initialization; negative tiny gaps may "
                        "reflect numerical convergence tolerances"
                    ),
                }
            )
        for arm in ("shared_4", "starts_100"):
            tested, identical = 0, 0
            for directory in sorted((root / "study" / cohort).glob("dataset_*")):
                if any(
                    read(directory / name / "record.json")["status"] != "complete"
                    for name in ("baseline", arm)
                ):
                    continue
                with (
                    np.load(directory / "baseline/run/mixture.npz", allow_pickle=False) as a,
                    np.load(directory / arm / "run/mixture.npz", allow_pickle=False) as b,
                ):
                    np.testing.assert_array_equal(a["scores"], b["scores"])
                    identical += 1
                tested += 1
            result["paired_scores"].append(
                {"cohort": cohort, "arm": arm, "tested": tested, "exactly_identical": identical}
            )
    return result


def report(root, output, figures):
    output.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    audit = root / "audit"
    for name in (
        "summary.json",
        "points.csv",
        "candidates.csv",
        "truth_initialized.csv",
        "retained_tails.csv",
    ):
        shutil.copyfile(audit / name, output / name)
    for source, name in (
        ("study.json", "study.json"),
        ("study/datasets.json", "datasets.json"),
        ("study/source_study.json", "source_study.json"),
        ("study-execution.json", "execution.json"),
        ("runtime_sources.json", "runtime_sources.json"),
        ("runtime_versions.json", "runtime_versions.json"),
    ):
        shutil.copyfile(root / source, output / name)
    (output / "mechanisms.json").write_text(
        json.dumps(mechanisms(root), indent=2, sort_keys=True) + "\n"
    )
    plot(audit, figures)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    manifest = {
        "local_artifact_id": "full-grouped-diagnostics-20261023",
        "study_inventory_sha256": digest(root / "study/files.json"),
        "runtime_wheel": read(root / "runtime_package.json"),
        "published_artifacts": {p.name: digest(p) for p in sorted(output.iterdir())},
        "figure_files": {p.name: digest(p) for p in sorted(figures.iterdir())},
        "report_script_sha256": digest(Path(__file__)),
        "inference_scope": (
            "Exploratory paired point diagnostics. Historical baseline p-values only; "
            "no new calibration or stability draws, no species-boundary claim."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.artifact_root, args.output_dir, args.figure_dir)


if __name__ == "__main__":
    main()
