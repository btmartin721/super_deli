"""Export aggregate-only tradeoffs and fixed primary-model figures from audited evidence."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def export(study, destination):
    independent = read(study / "report-qa.json")
    if not independent["passed"]:
        raise ValueError("Independent report audit required")
    destination.mkdir()
    for name in (
        "summary.json",
        "groups.json",
        "primary-rates.json",
        "paired-contrasts.json",
        "classwise.json",
    ):
        write(destination / name, read(study / name))
    frozen = read(study / "frozen.json")
    write(
        destination / "study-design.json",
        dict(
            version=frozen["package_version"],
            settings=frozen["settings"],
            training=frozen["training"],
            counts=frozen["counts"],
            software=frozen["software"],
            independent_datasets_per_cell=10,
            source_scope="Invented matched continuous measurements",
            seed_interpretation="Seeds, policies and evaluation rows repeat datasets; no selection",
        ),
    )
    write(
        destination / "independent-audit.json",
        dict(
            preflight={
                k: read(study / "preflight-audit.json")[k]
                for k in ("passed", "neural_fits", "verified_optimizer_updates")
            },
            main={
                k: read(study / "main-audit.json")[k]
                for k in ("passed", "neural_fits", "verified_optimizer_updates")
            },
            report=independent,
            raw_individual_exports=0,
        ),
    )


def save_figure(figure, directory, name):
    for suffix in ("png", "svg", "pdf"):
        path = directory / (name + "." + suffix)
        if path.exists():
            raise FileExistsError(path)
        metadata = {"Creator": "Super Deli"}
        if suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        if suffix == "svg":
            metadata["Date"] = None
        figure.savefig(path, dpi=170, bbox_inches="tight", metadata=metadata)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )


def figures(study, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "super-deli-reference-abstention-v1"})
    directory.mkdir()
    groups = read(study / "groups.json")
    cases = ("linear_weak", "linear_standard", "combined", "confounded_size")
    names = ("Weak linear", "Standard linear", "Allometry + saturation", "Confounded body size")
    models = ("logistic", "residual_0")
    policies = ("unscreened", "probability", "distance")
    colors = dict(unscreened="#777777", probability="#0072B2", distance="#D55E00")
    markers = dict(logistic="o", residual_0="s")
    points = []

    def row(case, n, arm, policy, cohort, key):
        group = next(g for g in groups if g["case"] == case and g["n"] == n)
        arm_row = next(a for a in group["arms"] if a["arm"] == arm and a["policy"] == policy)
        return arm_row["cohorts"][cohort].get(
            key, dict(available=0, mean=None, median=None, minimum=None, maximum=None)
        )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for r, n in enumerate((64, 256)):
        for col, (case, name) in enumerate(zip(cases, names, strict=True)):
            ax = axes[r, col]
            for model in models:
                for policy in policies:
                    x = row(case, n, model, policy, "iid", "accepted_fraction")
                    y = row(case, n, model, policy, "iid", "accepted_accuracy")
                    points.append(
                        dict(
                            figure="coverage-accuracy",
                            case=case,
                            n=n,
                            model=model,
                            policy=policy,
                            coverage=x,
                            accuracy=y,
                        )
                    )
                    if x["mean"] is not None and y["mean"] is not None:
                        ax.scatter(
                            x["mean"],
                            y["mean"],
                            marker=markers[model],
                            color=colors[policy],
                            s=55,
                            edgecolor="white",
                            linewidth=0.4,
                            zorder=3,
                        )
            ax.set_title(f"{name}\nN={n}")
            ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
            ax.axvline(0.8, color="#999999", linestyle=":", linewidth=0.8)
            ax.axhline(0.9, color="#999999", linestyle=":", linewidth=0.8)
            ax.grid(alpha=0.15)
            if r == 1:
                ax.set_xlabel("Known specimens accepted (fraction)")
            if col == 0:
                ax.set_ylabel("Accuracy among accepted known specimens")
    handles = [
        Line2D([], [], color=colors[p], marker="o", linestyle="", label=p.title()) for p in policies
    ]
    handles += [
        Line2D(
            [],
            [],
            color="black",
            marker=markers[m],
            linestyle="",
            label="Logistic" if m == "logistic" else "Residual seed 0",
        )
        for m in models
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.025), frameon=False
    )
    fig.suptitle("Reference assignment: coverage and accepted accuracy", y=1.075, fontsize=15)
    fig.tight_layout()
    save_figure(fig, directory, "coverage-accuracy")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharey=True)
    series = [(model, policy) for model in models for policy in ("probability", "distance")]
    for r, cohort in enumerate(("near", "far", "overlap")):
        for col, n in enumerate((64, 256)):
            ax = axes[r, col]
            for j, (model, policy) in enumerate(series):
                for i, case in enumerate(cases):
                    metric = row(case, n, model, policy, cohort, "accepted_fraction")
                    points.append(
                        dict(
                            figure="unfamiliar-acceptance",
                            case=case,
                            n=n,
                            model=model,
                            policy=policy,
                            cohort=cohort,
                            acceptance=metric,
                        )
                    )
                    if metric["mean"] is None:
                        continue
                    x = i + (j - 1.5) * 0.18
                    ax.bar(
                        x,
                        metric["mean"],
                        width=0.17,
                        color=colors[policy],
                        hatch="//" if model == "residual_0" else "",
                        edgecolor="#333333",
                        linewidth=0.3,
                        yerr=[
                            [metric["mean"] - metric["minimum"]],
                            [metric["maximum"] - metric["mean"]],
                        ],
                        capsize=2,
                        error_kw=dict(linewidth=0.65),
                    )
            title = {
                "near": "Nearby unfamiliar",
                "far": "Distant unfamiliar",
                "overlap": "Indistinguishable unfamiliar",
            }[cohort]
            ax.set_title(f"{title} · N={n}")
            ax.set_xticks(
                range(4), ["Weak linear", "Standard linear", "Combined", "Confounded size"]
            )
            ax.tick_params(axis="x", rotation=12)
            ax.set_ylim(0, 1.05)
            ax.axhline(0.1, color="#999999", linestyle=":", linewidth=0.8)
            ax.grid(axis="y", alpha=0.15)
            if col == 0:
                ax.set_ylabel("Unfamiliar specimens accepted (fraction)")
    handles = [
        Patch(
            facecolor=colors[p],
            hatch="//" if m == "residual_0" else "",
            edgecolor="#333333",
            label=("Logistic" if m == "logistic" else "Residual 0") + " / " + p.title(),
        )
        for m, p in series
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.025), frameon=False
    )
    fig.suptitle("Unfamiliar acceptance: dataset means and observed ranges", y=1.06, fontsize=15)
    fig.tight_layout()
    save_figure(fig, directory, "unfamiliar-acceptance")
    plt.close(fig)
    write(
        directory / "figure-data.json",
        dict(
            points=points,
            interpretation=(
                "Primary seed fixed in advance; ranges span ten independent training datasets, "
                "not confidence intervals; undefined accuracy is not plotted"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.study_root, args.output_dir)
    figures(args.study_root, args.figure_dir)
