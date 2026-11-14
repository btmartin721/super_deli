"""Opt-in SNPio-backed complete-genotype discovery and paired PCA comparisons."""

from dataclasses import replace
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np

from super_deli.core.partition_comparison import compare_partitions
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.io.genomic_neural import prepare_genotypes, verify_genomic_sources
from super_deli.models.genomic_neural import (
    fit_genomic_legacy_point,
    fit_genomic_neural_point,
    fit_genomic_pca_shared_point,
)
from super_deli.models.genomic_neural_config import (
    TRAINING,
    GenomicComparisonConfig,
    GenomicNeuralConfig,
)
from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples
from super_deli.workflows.tabular_analysis import _csv

LIMITATIONS = [
    "Candidate genetic clusters are not species boundaries; no new p-values.",
    (
        "Complete unrelated diploid biallelic calls and a fixed panel are assumed; "
        "population labels never enter training."
    ),
    (
        "Patterson scaling and locus filtering are fitted to each training subset; "
        "the original SNPio allele coding/panel stays fixed."
    ),
    (
        "Neural reconstruction probabilities and GMM memberships are conditional "
        "fitted quantities, not genotype uncertainty or species probabilities."
    ),
    (
        "Specimen subsampling conditions on the locus panel; it is distinct from the "
        "existing locus-block bootstrap and does not address linkage sensitivity."
    ),
    "Linkage, continuous structure, admixture, relatedness and ascertainment may induce clusters.",
    (
        "All fixed neural seeds are retained; no automatic winner or consensus. "
        "Compare BIC only within a fitted representation."
    ),
    (
        "Frozen numeric projection requires the identical source-locus order and "
        "allele coding; file-relative locus keys are not cross-file homology claims."
    ),
]


def _point(family, values, model, keys):
    function = {
        "categorical_ae": fit_genomic_neural_point,
        "pca_shared": fit_genomic_pca_shared_point,
        "pca_diagonal": fit_genomic_legacy_point,
    }[family]
    return function(values, config=model, locus_keys=keys)


def _save_point(point, out, ids):
    out.mkdir(exist_ok=False)
    np.savez_compressed(out / "model.npz", **point.arrays)
    metadata = dict(
        schema_version=1,
        family=point.model.family,
        locus_keys=point.model.locus_keys,
        model_sha256=fingerprint_file(out / "model.npz").sha256,
        candidates=point.candidates,
        diagnostics=point.diagnostics,
    )
    write_json(metadata, out / "model.json")
    _csv(
        out / "assignments.csv",
        [
            dict(
                unit_id=u,
                candidate_cluster=int(point.arrays["labels"][i]) + 1,
                maximum_conditional_membership=float(point.arrays["membership"][i].max()),
            )
            for i, u in enumerate(ids)
        ],
    )


def _run(dataset, model, arms, base_dir, output_dir, modality_name):
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        values, cohort, sources = prepare_genotypes(
            dataset, model, base_dir, out / "inputs", modality_name
        )
        ids = cohort["unit_ids"]
        keys = cohort["locus_keys"]
        manifest = dict(
            schema_version=1,
            dataset=dataset.to_dict(),
            model=model.to_dict(),
            arms=[dict(name=n, family=f, seed=s) for n, f, s in arms],
            training=TRAINING,
            software={
                name: version(name)
                for name in ("super-deli", "snpio", "numpy", "scipy", "scikit-learn", "torch")
            },
            cohort=cohort,
            sources=sources,
        )
        digest = configuration_hash(manifest)
        write_json(manifest, out / "manifest.json")
        points, rows = {}, []
        (out / "arms").mkdir()
        for name, family, seed in arms:
            child = out / "arms" / name
            row = dict(
                name=name, family=family, seed=seed, status="failed", selected_k=None, error=None
            )
            try:
                point = _point(family, values, replace(model, random_seed=seed), keys)
                _save_point(point, child, ids)
                points[name] = point
                row.update(
                    status="complete",
                    selected_k=point.selected_k,
                    at_search_limit=point.selected_k == model.max_k,
                    candidates=point.candidates,
                    diagnostics=point.diagnostics,
                )
            except Exception as exc:
                child.mkdir(exist_ok=True)
                row["error"] = f"{type(exc).__name__}: {exc}"
                if getattr(exc, "arrays", None):
                    np.savez_compressed(child / "failed-model.npz", **exc.arrays)
                write_json(
                    dict(error=row["error"], diagnostics=getattr(exc, "diagnostics", {})),
                    child / "failure.json",
                )
            rows.append(row)
        records = {name: [] for name, _, _ in arms}
        draws = subsample_indices(len(values), model.n_subsamples, 0.8, model.subsampling_seed)
        if model.n_subsamples:
            (out / "subsamples").mkdir()
        for draw, selected in enumerate(draws):
            draw_dir = out / "subsamples" / f"draw-{draw:04d}"
            draw_dir.mkdir()
            for name, family, seed in arms:
                record = dict(
                    draw=draw,
                    training_indices=selected.tolist(),
                    unit_ids=[ids[i] for i in selected],
                    status="failed",
                    labels=None,
                    selected_k=None,
                    error=None,
                )
                try:
                    point = _point(family, values[selected], replace(model, random_seed=seed), keys)
                    _save_point(point, draw_dir / name, record["unit_ids"])
                    record.update(
                        status="complete",
                        labels=point.arrays["labels"].tolist(),
                        selected_k=point.selected_k,
                    )
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    if getattr(exc, "arrays", None):
                        np.savez_compressed(draw_dir / (name + "-failed.npz"), **exc.arrays)
                records[name].append(record)
            write_json(
                {name: record[-1] for name, record in records.items()}, draw_dir / "records.json"
            )
        for row in rows:
            stability = summarize_subsamples(len(values), records[row["name"]])
            native, arrays = stability.summary, stability.arrays
            row["stability"] = native
            row["stability"]["conditioning"] = (
                "80% specimens without replacement; fixed source panel/coding and optimizer seed"
            )
            if arrays:
                np.savez_compressed(out / "arms" / row["name"] / "stability.npz", **arrays)
            write_json(records[row["name"]], out / "arms" / row["name"] / "subsamples.json")
        comparisons = []
        for left, right in combinations(points, 2):
            a, b = points[left].arrays["labels"], points[right].arrays["labels"]
            comparisons.append(
                dict(
                    left=left,
                    right=right,
                    **compare_partitions(a.tolist(), b.tolist(), [True] * len(a), [True] * len(b)),
                )
            )
        _csv(
            out / "assignments.csv",
            [
                dict(
                    unit_id=u,
                    sample_id=cohort["sample_ids"][i],
                    **{
                        name + "__cluster": int(p.arrays["labels"][i]) + 1
                        for name, p in points.items()
                    },
                )
                for i, u in enumerate(ids)
            ],
        )
        summary = dict(
            schema_version=1,
            status="complete"
            if len(points) == len(arms) and not any(row["stability"]["failed"] for row in rows)
            else "partial"
            if points
            else "failed",
            dataset_id=dataset.dataset_id,
            unit_ids=ids,
            n_units=len(ids),
            n_source_loci=values.shape[1],
            models=rows,
            comparisons=comparisons,
            configuration_sha256=digest,
            limitations=LIMITATIONS,
            inference=dict(status="not_computed", p_value=None),
        )
        verify_genomic_sources(out / "inputs", sources)
        from super_deli.workflows.genomic_neural_report import write_report

        write_report(summary, points, out / "report.html")
        summary["artifacts"] = {
            p.relative_to(out).as_posix(): fingerprint_file(p).sha256
            for p in sorted(out.rglob("*"))
            if p.is_file()
        }
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        write_json(
            dict(status="failed", error=f"{type(exc).__name__}: {exc}"), out / "failure.json"
        )
        raise


def run_genomic_neural(dataset, *, model_config, base_dir, output_dir, modality_name=None):
    model = GenomicNeuralConfig.from_dict(model_config.to_dict())
    dataset = config_from_dict(dataset.to_dict())
    return _run(
        dataset,
        model,
        [("neural", "categorical_ae", model.random_seed)],
        base_dir,
        output_dir,
        modality_name,
    )


def run_genomic_comparison(dataset, *, analysis_config, base_dir, output_dir, modality_name=None):
    config = GenomicComparisonConfig.from_dict(analysis_config.to_dict())
    dataset = config_from_dict(dataset.to_dict())
    arms = [
        ("pca_diagonal", "pca_diagonal", config.model.encoding_seed),
        ("pca_shared", "pca_shared", config.model.encoding_seed),
    ]
    arms += [(f"neural_seed_{seed}", "categorical_ae", seed) for seed in config.neural_seeds]
    return _run(dataset, config.model, arms, base_dir, output_dir, modality_name)
