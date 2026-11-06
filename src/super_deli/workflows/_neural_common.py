"""Source-bound neural clustering with explicit noise and native subsampling."""

from __future__ import annotations

import csv
import html
from importlib.metadata import version
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input
from super_deli.io.config import read_json, write_json
from super_deli.io.tabular import ingest_tabular


def _report(
    out,
    summary,
    point,
    units,
    *,
    title=None,
    view_details="",
    group_label="density groups",
    strength_label="Density strength",
    support_explanation=(
        "Conditional density strengths and subsampling sensitivity are distinct. "
        "Neither is a species probability or a significance level. P-values are not computed."
    ),
    plot_label="Neural representation colored by density group",
):
    colors = ("#1565c0", "#d84315", "#2e7d32", "#6a1b9a", "#00796b", "#ad1457")
    low, high = point.scores.min(axis=0), point.scores.max(axis=0)
    span = high - low
    dots, rows = [], []
    for unit, label, strength, score in zip(
        units, point.labels, point.strength, point.scores, strict=True
    ):
        x, y = 40 + 520 * (score - low) / span
        color = "#777777" if label < 0 else colors[int(label) % len(colors)]
        name = "noise/unassigned" if label < 0 else f"cluster_{int(label) + 1}"
        dots.append(
            f'<circle cx="{x:.5f}" cy="{600 - y:.5f}" r="3" fill="{color}">'
            f"<title>{html.escape(unit)}: {name}</title></circle>"
        )
        rows.append(f"<tr><td>{html.escape(unit)}</td><td>{name}</td><td>{strength:.5f}</td></tr>")
    content = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Neural tabular groups</title>'
        "<style>body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem}"
        "td,th{text-align:left;padding:.3rem .8rem;border-bottom:1px solid #ddd}"
        "svg{max-width:100%;height:auto}caption{text-align:left;font-weight:bold}</style>"
        + (
            f"<h1>{html.escape(title)}</h1>"
            if title
            else "<h1>Tabular denoising autoencoder and HDBSCAN</h1>"
            if "denoising" in summary["model"]
            else "<h1>Tabular autoencoder and HDBSCAN</h1>"
        )
        + view_details
        + f"<p>{len(units)} units; {summary['selected_k']} {html.escape(group_label)}; "
        f"{summary['noise_count']} noise/unassigned units.</p>"
        "<p>Two-dimensional latent representation. Colors identify local clusters. "
        "Gray observations are unassigned. Axes have no anatomical interpretation.</p>"
        f'<svg role="img" aria-label="{html.escape(plot_label)}" '
        'viewBox="0 0 600 600">' + "".join(dots) + "</svg>"
        f"<p>Initial/final training MSE: {point.losses[0]:.6g} / {point.losses[-1]:.6g}. "
        "This is transductive training reconstruction, not external validation.</p>"
        f"<p>Subsampling: {html.escape(summary['stability']['status'])}. "
        "Read summary.json and subsamples.json for inclusion denominators and failures.</p>"
        f"<p>{html.escape(support_explanation)}</p>"
        "<table><caption>Point assignments</caption><thead><tr><th>Unit</th><th>Group</th>"
        f"<th>{html.escape(strength_label)}</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>"
    )
    with (out / "report.html").open("x", encoding="utf-8") as f:
        f.write(content)


def run_neural_workflow(
    dataset,
    *,
    model_config,
    config_type,
    fit_function,
    training,
    base_dir,
    output_dir,
    modality_name=None,
):
    """Fit one declared neural seed. Always preserve sources and failed outcomes."""
    dataset = config_from_dict(dataset.to_dict())
    config = config_type.from_dict(model_config.to_dict())
    modalities = [
        m
        for m in dataset.modalities
        if m.kind in {"morphological", "ecological"}
        and (modality_name is None or m.name == modality_name)
    ]
    if len(modalities) != 1:
        raise ValueError("Select exactly one complete continuous morphology/ecology table")
    try:
        import numpy as np

        from super_deli.models.tabular_neural import point_arrays
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install super-deli[neural] for neural clustering") from exc
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    source = resolve_input(modalities[0].path, base_dir)
    try:
        before = fingerprint_file(source)
        loaded = ingest_tabular(
            dataset,
            base_dir=base_dir,
            output_dir=out / "ingestion",
            modality_names=[modalities[0].name],
        )
        table = loaded.tables[0]
        write_json(
            {
                "unit_ids": list(table.unit_ids),
                "observation_ids": list(table.observation_ids),
                "feature_names": list(config.feature_names),
            },
            out / "cohort.json",
        )
        write_json(config.to_dict(), out / "model-config.json")
        point, stability = fit_function(table, config=config)
        with (out / "point.npz").open("xb") as f:
            np.savez_compressed(f, **point_arrays(point))
        with (out / "stability.npz").open("xb") as f:
            np.savez_compressed(f, **stability.arrays)
        write_json(stability.records, out / "subsamples.json")
        write_json(point.diagnostics, out / "training.json")
        with (out / "assignments.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "observation_id",
                    "unit_id",
                    "cluster",
                    "density_membership_strength",
                    "latent_1",
                    "latent_2",
                ]
            )
            writer.writerows(
                [obs, unit, int(label), float(strength), *score]
                for obs, unit, label, strength, score in zip(
                    table.observation_ids,
                    table.unit_ids,
                    point.labels,
                    point.strength,
                    point.scores,
                    strict=True,
                )
            )
        configuration = {
            "dataset": dataset.to_dict(),
            "model": config.to_dict(),
            "modality": table.modality_name,
            "training": training,
        }
        summary = {
            "schema_version": 1,
            "status": "complete",
            "model": config.model_family,
            "dataset_id": dataset.dataset_id,
            "modality": table.modality_name,
            "level": table.level,
            "unit_ids": list(table.unit_ids),
            "observation_ids": list(table.observation_ids),
            "feature_names": list(config.feature_names),
            "selected_k": point.selected_k,
            "noise_count": int(np.sum(point.labels < 0)),
            "assigned_fraction": float(np.mean(point.labels >= 0)),
            "configuration": configuration,
            "configuration_sha256": configuration_hash(configuration),
            "source_sha256": before.sha256,
            "ingestion": loaded.summary,
            "model_sha256": fingerprint_file(out / "point.npz").sha256,
            "stability": stability.summary,
            "species_significance": {
                "status": "not_computed",
                "reason": "No neural null calibrated",
            },
            "software": {
                n: version(n)
                for n in ("super-deli", "numpy", "scipy", "scikit-learn", "torch", "threadpoolctl")
            },
            "interpretation": [
                "transductive unsupervised fit",
                "no best-seed selection",
                "noise is not a shared cluster",
                "no cluster prediction for unseen units",
                "density strength is not species probability",
                "no empirical species validation",
            ],
        }
        if fingerprint_file(source).sha256 != before.sha256:
            raise ValueError("Source changed during neural analysis")
        _report(out, summary, point, table.unit_ids)
        summary["artifact_sha256"] = {
            str(p.relative_to(out)): fingerprint_file(p).sha256
            for p in sorted(out.rglob("*"))
            if p.is_file()
        }
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        diagnostics = getattr(exc, "diagnostics", {})
        arrays = getattr(exc, "arrays", {})
        if arrays:
            with (out / "failed-training.npz").open("xb") as f:
                np.savez_compressed(f, **arrays)
        write_json(
            {"status": "failed", "error": str(exc), "diagnostics": diagnostics},
            out / "failure.json",
        )
        raise


def load_numeric_neural_model(directory, *, family):
    """Load a strictly shaped numeric model. Does not import PyTorch or predict groups."""
    import numpy as np

    from super_deli.models.tabular_neural import NeuralModel

    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    if summary.get("model") != family:
        raise ValueError("Not a supported neural model")
    if configuration_hash(summary["configuration"]) != summary["configuration_sha256"]:
        raise ValueError("Neural configuration fingerprint mismatch")
    if list(summary["configuration"]["model"]["feature_names"]) != summary["feature_names"]:
        raise ValueError("Neural feature-order mismatch")
    if fingerprint_file(directory / "point.npz").sha256 != summary["model_sha256"]:
        raise ValueError("Neural model fingerprint mismatch")
    with np.load(directory / "point.npz", allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in f.files}
    p = len(summary["feature_names"])
    ids = arrays["kept_indices"]
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or not 3 <= len(ids) <= p
        or not np.array_equal(ids, np.unique(ids))
        or np.any(ids < 0)
        or np.any(ids >= p)
    ):
        raise ValueError("Invalid kept feature indices")
    for key, shape in (
        ("means", (p,)),
        ("scales", (p,)),
        ("latent_means", (2,)),
        ("latent_scales", (2,)),
    ):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid neural array: {key}")
    if np.any(arrays["scales"] <= 0) or np.any(arrays["latent_scales"] <= 0):
        raise ValueError("Neural normalization scales must be positive")
    for side, widths in (("encoder", [len(ids), 16, 8, 2]), ("decoder", [2, 8, 16, len(ids)])):
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            for kind, shape in (("weight", (b, a)), ("bias", (b,))):
                value = arrays[f"{side}_{i}_{kind}"]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("Invalid neural weight array")
    return NeuralModel(tuple(summary["feature_names"]), arrays)
