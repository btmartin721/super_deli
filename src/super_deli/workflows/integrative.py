"""Read selected modalities and integrate only explicitly linked target profiles."""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path

from super_deli.core.data import DatasetConfig
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import validate_dataset
from super_deli.io.config import write_json
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.workflows.cluster_artifacts import candidate_result, write_cluster_artifacts


def _target_ids(dataset, source_level, source_ids, view, target_level):
    if source_level == target_level:
        if view.aggregation != "none":
            raise ValueError("Same-level views require aggregation='none'")
        return tuple(source_ids)
    if not (
        source_level == "specimen"
        and target_level in {"population", "locality"}
        and view.aggregation == "mean"
    ):
        raise ValueError(
            "No downward broadcast; cross-level views require specimen mean aggregation"
        )
    specimens = {s.specimen_id: s for s in dataset.specimens}
    targets = tuple(getattr(specimens[i], f"{target_level}_id") for i in source_ids)
    if any(t is None for t in targets):
        raise ValueError(f"Every aggregated specimen requires an explicit {target_level}_id")
    return targets


def _load_view(dataset, source, view, config, base_dir, directory):
    import numpy as np

    from super_deli.models.integrative import IntegrationInput

    exclusions, names, types, landmarks = [], (), (), ()
    if source.kind == "genomic":
        from super_deli.io.genomic import ingest_genomic
        from super_deli.io.genomic_encoding import encode_diploid

        loaded = ingest_genomic(
            dataset,
            base_dir=base_dir,
            output_dir=directory / "ingestion",
            modality_name=source.name,
        )
        values, encoding = encode_diploid(
            loaded, output_prefix=directory / "encoding", random_seed=config.random_seed
        )
        write_json(encoding, directory / "encoding.json")
        observation_ids, units = loaded.sample_ids, loaded.specimen_ids
    elif source.kind in {"morphological", "ecological"}:
        from super_deli.io.tabular import ingest_tabular
        from super_deli.models.tabular_gmm import _matrix

        loaded = ingest_tabular(
            dataset,
            base_dir=base_dir,
            output_dir=directory / "ingestion",
            modality_names=[source.name],
        )
        table = loaded.tables[0]
        values, names, types = _matrix(table, view.tabular_config(config.max_working_bytes))
        observation_ids, units = table.observation_ids, table.unit_ids
    else:
        from super_deli.io.morphometric import ingest_morphometric

        loaded = ingest_morphometric(
            dataset,
            base_dir=base_dir,
            output_dir=directory / "ingestion",
            modality_names=[source.name],
        )
        collection = loaded.collections[0]
        records = []
        for record in collection.records:
            if any(any(point) for point in record.missing_mask):
                if view.shape.missing_policy == "error":
                    raise ValueError(f"{record.observation_id}: incomplete shape landmarks")
                exclusions.append(
                    {
                        "modality": source.name,
                        "observation_id": record.observation_id,
                        "unit_id": record.unit_id,
                        "reason": "incomplete_landmark_coordinates",
                    }
                )
            else:
                records.append(record)
        values = np.array([r.coordinates for r in records], dtype=float)
        observation_ids, units = (
            tuple(r.observation_id for r in records),
            tuple(r.unit_id for r in records),
        )
        landmarks = collection.landmark_ids
    targets = _target_ids(dataset, source.level, units, view, config.target_level)
    data = IntegrationInput(
        source.name,
        source.kind,
        source.level,
        values,
        tuple(observation_ids),
        tuple(units),
        targets,
        names,
        types,
        landmarks,
    )
    with (directory / "source_values.npy").open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
    write_json(
        {
            "schema_version": 1,
            "observation_ids": data.observation_ids,
            "unit_ids": data.unit_ids,
            "target_ids": data.target_ids,
            "feature_names": names,
            "feature_types": types,
            "landmark_ids": landmarks,
            "source_level": source.level,
            "array_sha256": fingerprint_file(directory / "source_values.npy").sha256,
        },
        directory / "source_rows.json",
    )
    return data, loaded.summary["provenance"], exclusions


def run_integrative(
    dataset: DatasetConfig,
    *,
    model_config: IntegrativeConfig,
    base_dir: str | Path,
    output_dir: str | Path,
):
    dataset = config_from_dict(dataset.to_dict())
    model_config = IntegrativeConfig.from_dict(model_config.to_dict())
    report = validate_dataset(dataset, base_dir=base_dir, check_files=False)
    if not report.valid:
        raise ValueError("; ".join(i.message for i in report.issues if i.severity == "error"))
    by_name = {m.name: m for m in dataset.modalities}
    selected = []
    for view in model_config.views:
        if view.modality not in by_name or by_name[view.modality].kind != view.kind:
            raise ValueError("Every selected view must match a declared modality and kind")
        source = by_name[view.modality]
        if source.kind == "morphometric" and source.format not in {"tps", "landmarks_csv"}:
            raise ValueError(
                "Shape integration requires reviewed TPS/coordinate data, not raw images"
            )
        # Check linkage before creating output directories or reading numerical data.
        _target_ids(
            dataset,
            source.level,
            [o.unit_id for o in source.observations],
            view,
            model_config.target_level,
        )
        selected.append(source)
    try:
        from super_deli.io.integrative_artifacts import write_integration_parameters
        from super_deli.models.integrative import fit_integrative
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Install 'super-deli[clustering]' and add [genomics] for genomic views"
        ) from exc
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    try:
        inputs, provenance, observation_exclusions = [], [], []
        for i, (view, source) in enumerate(zip(model_config.views, selected, strict=True)):
            directory = destination / f"view_{i:03d}"
            directory.mkdir()
            data, evidence, excluded = _load_view(
                dataset, source, view, model_config, base_dir, directory
            )
            inputs.append(data)
            provenance.append(evidence)
            observation_exclusions.extend(excluded)
        coverage = {data.name: sorted(set(data.target_ids)) for data in inputs}
        declared = (
            {s.specimen_id for s in dataset.specimens}
            if model_config.target_level == "specimen"
            else set(
                dataset.populations
                if model_config.target_level == "population"
                else dataset.localities
            )
        )
        unobserved = sorted(declared - set().union(*(set(ids) for ids in coverage.values())))
        write_json(
            {
                "schema_version": 1,
                "target_level": model_config.target_level,
                "target_coverage": coverage,
                "unobserved_target_units": unobserved,
                "excluded_observations": observation_exclusions,
            },
            destination / "input_coverage.json",
        )
        fit = fit_integrative(inputs, config=model_config)
        configuration = {"dataset": dataset.to_dict(), "model": model_config.to_dict()}
        digest = configuration_hash(configuration)
        result = candidate_result(
            dataset,
            fit,
            digest,
            model_config.target_level,
            model_name=f"integrative_weighted_pca_{model_config.covariance_type}_gmm",
            space="weighted integrative PCA",
            pipeline="view preprocessing/profile fusion/PCA/GMM",
        )
        write_cluster_artifacts(
            fit,
            result,
            scores=fit.pipeline.scores,
            observation_ids=fit.unit_ids,
            directory=destination,
        )
        views = write_integration_parameters(fit, destination)
        with (destination / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("unit_id", *(f"PC{i + 1}" for i in range(model_config.n_pcs))))
            writer.writerows(
                (unit, *scores)
                for unit, scores in zip(fit.unit_ids, fit.pipeline.scores, strict=True)
            )
        write_json(
            {
                "schema_version": 1,
                "target_level": model_config.target_level,
                "unit_ids": fit.unit_ids,
                "excluded_units": fit.exclusions,
                "excluded_observations": observation_exclusions,
                "views": [
                    {
                        "modality": v["modality"],
                        "source_level": v["source_level"],
                        "aggregation": v["aggregation"],
                        "members": v["members"],
                    }
                    for v in views
                ],
            },
            destination / "alignment.json",
        )
        stability = fit.stability
        summary = {
            "schema_version": 1,
            "status": "complete",
            "dataset_id": dataset.dataset_id,
            "model": result.model,
            "level": model_config.target_level,
            "unit_ids": list(fit.unit_ids),
            "row_kind": "integrated target-unit profile",
            "unobserved_target_units": unobserved,
            "excluded_units": fit.exclusions,
            "excluded_observations": observation_exclusions,
            "selected_k": fit.point.selected_k,
            "covariance_type": model_config.covariance_type,
            "n_pcs": model_config.n_pcs,
            "views": views,
            "point_seed": fit.point_seed,
            "fusion": (
                "Center target PC profiles; divide by sqrt(total score variance); "
                "multiply by sqrt(normalized weight); concatenate and fit joint PCA"
            ),
            "bootstrap": {
                "status": stability.status,
                "requested": stability.requested,
                "successful": stability.successful,
                "failed": len(stability.failures),
                "selected_k_counts": stability.selected_k_counts,
                "resampling_unit": "Paired independent target units across all views",
                "pipeline": "Refit preprocessing/PCA, profiles, block scaling, joint PCA and GMM",
                "conditioning": "Fixed group members and source genomic loci/eligibility",
                "oob": "Pair-specific requested/successful counts when both targets were absent",
                "intervals": "Failed-outcome identification bounds, not confidence intervals",
            },
            "interpretation": [
                "Candidate integrated structure, not species probabilities or formal species tests",
                "Paired target resampling; unmatched individuals require explicit group links",
                "More source observations give a group more influence on its view's raw PCA fit",
                "Equal source weights within group profiles; equal target-profile GMM weights",
                "No ecological down-broadcast, categorical encoding or missing-modality imputation",
                "Correlated views can emphasize shared signals; weights are not probabilities",
                "Stability conditions on observed members, locus eligibility and feature schemas",
                "Independent targets are assumed; spatial or related sampling can violate this",
                "In-bag stability can be optimistic; OOB denominators can be sparse",
            ],
            "provenance": {
                "configuration": configuration,
                "configuration_sha256": digest,
                "views": provenance,
                "fusion_sha256": fingerprint_file(destination / "fusion.npz").sha256,
                "mixture_sha256": fingerprint_file(destination / "mixture.npz").sha256,
                "software": {
                    name: version(name)
                    for name in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
                },
            },
        }
        write_json(summary, destination / "summary.json")
        return summary
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
