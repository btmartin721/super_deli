"""Publish audited boundary diagnostics with uncertainty across observations."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import statistics
from pathlib import Path

METRICS = (
    "statistic",
    "bic_statistic",
    "count_association",
    "lr_count_association",
    "covariance_min_over_regularizer",
    "morphology_retained_variance",
    "ecology_retained_variance",
)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(records, path):
    fields = list(dict.fromkeys(k for row in records for k in row))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def describe(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean_mcse": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
    }


def observation_means(root):
    pipelines = rows(root / "audit/pipelines.csv")
    covariance = {
        (r["cohort"], r["case"], r["repetition"]): r for r in rows(root / "audit/covariance.csv")
    }
    tails = {
        (r["cohort"], r["case"], r["repetition"], r["arm"], r["statistic_kind"]): r
        for r in rows(root / "audit/tails.csv")
    }
    grouped = {}
    for p in pipelines:
        if int(p["replicate"]) >= 0:
            grouped.setdefault((p["cohort"], p["case"], p["repetition"], p["arm"]), []).append(p)
    means = []
    for (cohort, case, repetition, arm), group in sorted(grouped.items()):
        complete = [p for p in group if p["status"] == "complete"]
        record = {
            "cohort": cohort,
            "case": case,
            "repetition": int(repetition),
            "arm": arm,
            "successful_null_fits": len(complete),
            "attempted_null_fits": len(group),
        }
        for key in METRICS:
            record["mean_null_" + key] = (
                statistics.mean(float(p[key]) for p in complete) if complete else None
            )
        record["eligible_four_fraction"] = (
            sum(4 in json.loads(p["eligible_ks"]) for p in complete) / len(complete)
            if complete
            else None
        )
        for kind in ("likelihood", "bic"):
            tail = tails[cohort, case, repetition, arm, kind]
            record[kind + "_p"] = float(tail["p_value"]) if tail["p_value"] else None
            record[kind + "_reject"] = tail["reject"] == "True" if tail["p_value"] else None
        cov = covariance[cohort, case, repetition]
        record.update(
            {
                key: float(cov[key])
                for key in (
                    "conditional_error",
                    "true_conditional_trace",
                    "fitted_conditional_trace",
                    "true_count_variance_ratio",
                    "fitted_count_variance_ratio",
                    "within_relative_error",
                )
            }
        )
        record["fitted_rank"] = int(cov["fitted_rank"])
        means.append(record)
    groups = []
    for cohort, case, arm in sorted({(r["cohort"], r["case"], r["arm"]) for r in means}):
        selected = [r for r in means if (r["cohort"], r["case"], r["arm"]) == (cohort, case, arm)]
        groups.append(
            {
                "cohort": cohort,
                "case": case,
                "arm": arm,
                "n_observations_with_null_fits": len(selected),
                "metrics": {
                    key: describe([r[key] for r in selected])
                    for key in selected[0]
                    if key.startswith("mean_null_")
                    or key
                    in {
                        "conditional_error",
                        "true_conditional_trace",
                        "fitted_conditional_trace",
                        "true_count_variance_ratio",
                        "fitted_count_variance_ratio",
                        "within_relative_error",
                        "eligible_four_fraction",
                    }
                },
            }
        )
    paired = []
    index = {(r["case"], r["repetition"], r["arm"]): r for r in means if r["cohort"] == "fresh"}
    for case, repetition in sorted(
        {(r["case"], r["repetition"]) for r in means if r["cohort"] == "fresh"}
    ):
        left, right = index[case, repetition, "fitted"], index[case, repetition, "known"]
        paired.append(
            {
                "case": case,
                "repetition": repetition,
                "p_fitted_minus_known": left["likelihood_p"] - right["likelihood_p"]
                if left["likelihood_p"] is not None and right["likelihood_p"] is not None
                else None,
                "mean_null_gain_fitted_minus_known": left["mean_null_statistic"]
                - right["mean_null_statistic"]
                if left["mean_null_statistic"] is not None
                and right["mean_null_statistic"] is not None
                else None,
                "conditional_trace_excess": left["fitted_conditional_trace"]
                - left["true_conditional_trace"],
                "predicted_count_variance_ratio_error": left["fitted_count_variance_ratio"]
                - left["true_count_variance_ratio"],
            }
        )
    return means, groups, paired


def raw_likelihoods(root, archived):
    import numpy as np
    from scipy.stats import multivariate_normal

    from super_deli.benchmarking.grouped_boundary_reference import known_model, load_model

    output = []
    for cohort, source in (("archived", archived), ("fresh", root / "study")):
        for case in read(root / "prespecified-design.json")["cases"]:
            for base in sorted((source / case).glob("dataset_*")):
                if not (base / "calibration/null_model.npz").exists():
                    continue
                fitted = load_model(base / "calibration")
                truth = known_model(fitted, case)

                def nll(model):
                    result = 0.0
                    for g in range(len(model.counts)):
                        x = model.morphology[model.targets == g]
                        n = len(x)
                        b, r = model.covariance, model.within
                        covariance = np.block(
                            [
                                [
                                    np.kron(np.ones((n, n)), b[:4, :4]) + np.kron(np.eye(n), r),
                                    np.tile(b[:4, 4:], (n, 1)),
                                ],
                                [np.tile(b[4:, :4], (1, n)), b[4:, 4:]],
                            ]
                        )
                        mean = np.r_[np.tile(model.mean[:4], n), model.mean[4:]]
                        observed = np.r_[x.ravel(), model.ecology[g]]
                        result -= float(
                            multivariate_normal.logpdf(observed, mean=mean, cov=covariance)
                        )
                    return result

                fitted_nll, known_nll = nll(fitted), nll(truth)
                np.testing.assert_allclose(
                    fitted_nll,
                    fitted.layout["diagnostics"]["negative_log_likelihood"],
                    rtol=1e-10,
                    atol=1e-8,
                )
                delta = known_nll - fitted_nll
                output.append(
                    {
                        "cohort": cohort,
                        "case": case,
                        "repetition": int(base.name.rsplit("_", 1)[1]),
                        "fitted_negative_log_likelihood": fitted_nll,
                        "known_negative_log_likelihood": known_nll,
                        "known_minus_fitted": delta,
                        "feasible_reference_better": delta < -1e-6,
                    }
                )
    return output


def plot(summary, groups, figures):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cases = [
        "independent_gaussian",
        "correlated_gaussian",
        "zero_conditional_gaussian",
        "rank_two_gaussian",
        "weak_conditional_gaussian",
    ]
    names = ["Independent", "Correlated", "Zero conditional", "Rank two", "Weak full rank"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for offset, cohort, arm, color, label in (
        (-0.18, "archived", "fitted", "#0072B2", "Archived fitted"),
        (0, "fresh", "fitted", "#D55E00", "Fresh fitted"),
        (0.18, "fresh", "known", "#009E73", "Fresh known-law"),
    ):
        selected = [
            next(
                r
                for r in summary["summaries"]
                if r["cohort"] == cohort
                and r["case"] == case
                and r["arm"] == arm
                and r["statistic_kind"] == "likelihood"
            )
            for case in cases
        ]
        y = np.array(
            [r["rejected"] / r["available"] if r["available"] else np.nan for r in selected]
        )
        ci = np.array([r["wilson_95"] or [np.nan, np.nan] for r in selected]).T
        axes[0].errorbar(
            np.arange(5) + offset,
            y,
            yerr=np.maximum(0, [y - ci[0], ci[1] - y]),
            fmt="o",
            capsize=2,
            color=color,
            label=label,
        )
    axes[0].axhline(0.05, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylim(-0.02, max(0.42, axes[0].get_ylim()[1]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    axes[0].set_ylabel("Rejection fraction")
    for ax, key, true_key in (
        (axes[1], "fitted_conditional_trace", "true_conditional_trace"),
        (axes[2], "fitted_count_variance_ratio", "true_count_variance_ratio"),
    ):
        selected = [
            next(
                r
                for r in groups
                if r["cohort"] == "fresh" and r["case"] == case and r["arm"] == "fitted"
            )
            for case in cases
        ]
        for offset, field, color, label in (
            (-0.08, key, "#D55E00", "Fitted"),
            (0.08, true_key, "#009E73", "Generating law"),
        ):
            values = [r["metrics"][field] for r in selected]
            ax.errorbar(
                np.arange(5) + offset,
                [v["mean"] for v in values],
                yerr=[v["mean_mcse"] or 0 for v in values],
                fmt="o",
                capsize=2,
                color=color,
                label=label,
            )
        ax.legend(fontsize=8)
    for ax, title in zip(
        axes,
        (
            "Likelihood rejection · Wilson 95%",
            "Conditional covariance trace · mean ± MCSE",
            "Variance ratio: n=1 / n=4 · mean ± MCSE",
        ),
        strict=True,
    ):
        ax.set_xticks(range(5), names, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_ylabel("Trace in true within-covariance units")
    axes[2].set_ylabel("Predicted conditional trace ratio")
    fig.suptitle(
        "Grouped Gaussian boundary diagnostic · 25 observations per law/cohort", fontsize=12
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"grouped-boundary-diagnostics.{suffix}", dpi=180)
    svg = figures / "grouped-boundary-diagnostics.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def report(root, output, figures, archived_study):
    output.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    summary = read(root / "audit/summary.json")
    # Pooled conditional draw descriptors are not population-level standard errors.
    for row in summary["summaries"]:
        for description in row["null_pipeline_metrics"].values():
            if description:
                description.pop("mean_mcse", None)
    summary["pooled_metric_scope"] = (
        "Descriptive draws; across-observation uncertainty uses observation-summary.json"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for name in ("tails.csv", "covariance.csv"):
        shutil.copyfile(root / "audit" / name, output / name)
    all_pipelines = rows(root / "audit/pipelines.csv")
    for case in sorted({r["case"] for r in all_pipelines}):
        selected = [r for r in all_pipelines if r["case"] == case]
        with (output / f"pipelines-{case}.csv.gz").open("xb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=list(selected[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(selected)
    means, groups, paired = observation_means(root)
    write_rows(means, output / "observations.csv")
    write_rows(paired, output / "paired-diagnostics.csv")
    write_rows(raw_likelihoods(root, archived_study), output / "raw-likelihood.csv")
    (output / "observation-summary.json").write_text(
        json.dumps(groups, indent=2, sort_keys=True) + "\n"
    )
    for name in (
        "study.json",
        "runtime_sources.json",
        "base_sources.json",
        "runtime_versions.json",
        "runtime_package.json",
        "source-changes.json",
        "reference-pins.json",
        "prespecified-design.json",
        "seed-domain-check.json",
        "study-execution.json",
        "independent-tails.json",
        "preservation-summary.json",
    ):
        shutil.copyfile(root / name, output / name)
    plot(summary, groups, figures)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    module_root = Path(__file__).resolve().parents[1]
    manifest = {
        "local_artifact_id": "grouped-boundary-diagnostics-20261023",
        "study_inventory_sha256": digest(root / "audit/files.json"),
        "runtime_wheel": read(root / "runtime_package.json"),
        "published_artifacts": {p.name: digest(p) for p in sorted(output.iterdir())},
        "figure_files": {p.name: digest(p) for p in sorted(figures.iterdir())},
        "report_script_sha256": digest(Path(__file__)),
        "independent_checker_sha256": digest(
            module_root / "examples/grouped-boundary-diagnostics/check_tails.py"
        ),
        "independent_candidate_equations_sha256": digest(
            module_root / "examples/grouped-likelihood/check_tails.py"
        ),
        "interpretation": (
            "Diagnostic paired references; known parameters are unavailable empirically. "
            "No correction, general size, significant K or species-boundary guarantee."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--archived-study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.artifact_root, args.output_dir, args.figure_dir, args.archived_study)


if __name__ == "__main__":
    main()
