"""Isolated observed-coordinate GMM workflow with raw-input provenance."""

import csv
import html
import shutil
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input
from super_deli.io.config import read_json, write_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.observed_config import ObservedGMMConfig
from super_deli.workflows.tabular_missing import cohort_rows


def run_observed_gmm(dataset, *, model_config, base_dir, output_dir, modality_name=None):
    dataset = config_from_dict(dataset.to_dict())
    model_config = ObservedGMMConfig.from_dict(model_config.to_dict())
    modalities = [
        m
        for m in dataset.modalities
        if m.kind in {"morphological", "ecological"}
        and (modality_name is None or m.name == modality_name)
    ]
    if len(modalities) != 1:
        raise ValueError("Select exactly one morphology/ecology table")
    try:
        import numpy as np

        from super_deli.models.observed_gmm import fit_observed_matrix
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install super-deli[clustering] for observed-data GMM") from exc
    out = Path(output_dir).absolute()
    out.mkdir(parents=True, exist_ok=False)
    source = resolve_input(modalities[0].path, base_dir)
    try:
        before = fingerprint_file(source)
        snapshot = out / ("input" + source.suffix)
        shutil.copyfile(source, snapshot)
        frozen = replace(
            dataset,
            modalities=tuple(
                replace(m, path=str(snapshot)) if m.name == modalities[0].name else m
                for m in dataset.modalities
            ),
        )
        loaded = ingest_tabular(
            frozen,
            base_dir=base_dir,
            output_dir=out / "ingestion",
            modality_names=[modalities[0].name],
        )
        table = loaded.tables[0]
        rows = cohort_rows(table, model_config.feature_names, "mean")
        write_json({"schema_version": 1, "rows": rows}, out / "cohort.json")
        raw = np.asarray(
            [[np.nan if v is None else v for v in r["values"]] for r in rows], dtype=float
        )
        # Persist the original mask even when an information guard rejects fitting.
        np.savez_compressed(out / "raw.npz", raw=raw, observed=~np.isnan(raw))
        fit = fit_observed_matrix(raw, config=model_config)
        np.savez_compressed(out / "model.npz", **fit.arrays)
        summary = {
            **fit.summary,
            "dataset_id": dataset.dataset_id,
            "modality": table.modality_name,
            "level": table.level,
            "kind": table.kind,
            "feature_order": list(model_config.feature_names),
            "unit_ids": list(table.unit_ids),
            "observation_ids": list(table.observation_ids),
            "source_sha256": before.sha256,
            "configuration": {"dataset": dataset.to_dict(), "model": model_config.to_dict()},
            "model_sha256": fingerprint_file(out / "model.npz").sha256,
            "software": {
                n: version(n)
                for n in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
            },
        }
        summary["configuration_sha256"] = configuration_hash(summary["configuration"])
        if fingerprint_file(source).sha256 != before.sha256 or (
            fingerprint_file(snapshot).sha256 != before.sha256
        ):
            raise ValueError("Source or frozen input changed during fitting")
        with (out / "membership.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            k = summary["selected_k"] or 0
            writer.writerow(
                [
                    "observation_id",
                    "unit_id",
                    "status",
                    "cluster",
                    *(f"membership_{i + 1}" for i in range(k)),
                ]
            )
            location = {int(source_index): i for i, source_index in enumerate(fit.kept_indices)}
            for i, row in enumerate(rows):
                prefix = [row["observation_id"], row["unit_id"]]
                if not row["eligible"] or fit.parameters is None:
                    writer.writerow(
                        prefix
                        + [
                            "excluded_all_missing"
                            if not row["eligible"]
                            else "unresolved_failed_search",
                            "",
                        ]
                        + [""] * k
                    )
                else:
                    j = location[i]
                    writer.writerow(
                        prefix
                        + [
                            "assigned",
                            int(fit.arrays["labels"][j]) + 1,
                            *fit.arrays["membership"][j],
                        ]
                    )
        write_json(summary, out / "summary.json")
        (out / "report.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Observed-data GMM</title>'
            "<h1>Observed-data Gaussian mixture</h1><p>Point estimates in raw feature space; "
            "no PCA, bootstrap, significance test or species probabilities.</p>"
            "<p>Assumes independent units, Gaussian components and ignorable missingness. "
            "Pair counts do not prove identifiability; inspect convergence and active bounds.</p>"
            "<pre>" + html.escape(__import__("json").dumps(summary, indent=2)) + "</pre>",
            encoding="utf-8",
        )
        write_json(
            {
                p.relative_to(out).as_posix(): fingerprint_file(p).sha256
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
            out / "checksums.json",
        )
        return summary
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, out / "failure.json")
        raise


def load_observed_model(directory):
    """Load a checksummed local point fit; prediction never refits preprocessing."""
    import numpy as np

    from super_deli.models.observed_gmm import Normalization, ObservedFit
    from super_deli.models.observed_likelihood import Parameters

    root = Path(directory)
    summary = read_json(root / "summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("model") != "observed_coordinate_full_gmm"
        or fingerprint_file(root / "model.npz").sha256 != summary.get("model_sha256")
    ):
        raise ValueError("A complete, checksummed observed-data model is required")
    config = ObservedGMMConfig.from_dict(summary["config"])
    with np.load(root / "model.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    p = Parameters(*(arrays[n] for n in ("weights", "means", "covariances")))
    k, d = summary["selected_k"], len(config.feature_names)
    if (
        p.weights.shape != (k,)
        or p.means.shape != (k, d)
        or p.covariances.shape != (k, d, d)
        or not np.isfinite(p.means).all()
        or not np.isfinite(p.covariances).all()
        or np.any(p.weights <= 0)
        or not np.isclose(p.weights.sum(), 1)
        or np.linalg.eigvalsh(p.covariances).min() < config.covariance_bound * (1 - 1e-7)
    ):
        raise ValueError("Invalid saved mixture parameters")
    return ObservedFit(
        summary,
        arrays,
        Normalization(arrays["normalization_means"], arrays["scales"]),
        p,
        arrays["kept_indices"],
    )
