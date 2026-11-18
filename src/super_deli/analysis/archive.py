"""Read completed matched-multimodal runs into a local, fit-free evidence report."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

from super_deli.core.partition_comparison import compare_partitions
from super_deli.core.provenance import configuration_hash
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import validate_dataset
from super_deli.io.config import write_json
from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig

from .archive_io import (
    ArchiveError,
    SealedRun,
    id_index,
    read_genomic_labels,
    read_membership_matrix,
    require,
)
from .boundary import boundary_agreement, evidence, groups_by_label, recurrence
from .report_html import render_html


@dataclass(frozen=True)
class ArchiveReport:
    json_path: Path
    html_path: Path


def _ids(ids, expected, context):
    require(
        isinstance(ids, list) and len(ids) == len(expected) and set(ids) == set(expected),
        f"{context}: inconsistent cohort IDs",
    )


def _verify_cohort(archive, manifest, analysis, cohort):
    dataset = config_from_dict(manifest["dataset"])
    frozen = config_from_dict(manifest["frozen_dataset"])
    for configuration in (dataset, frozen):
        require(
            validate_dataset(configuration, base_dir=archive.run, check_files=False).valid,
            "Invalid archived dataset declaration",
        )
    ids = cohort["unit_ids"]
    _ids(ids, {s.specimen_id for s in dataset.specimens}, "Declared cohort")
    _ids(ids, {s.specimen_id for s in frozen.specimens}, "Frozen cohort")
    require(
        bool(ids)
        and dataset.dataset_id == frozen.dataset_id
        and dataset.random_seed == frozen.random_seed
        and {s.specimen_id: s for s in dataset.specimens}
        == {s.specimen_id: s for s in frozen.specimens}
        and set(dataset.populations) == set(frozen.populations)
        and set(dataset.localities) == set(frozen.localities),
        "Archive dataset identity mismatch",
    )
    require(
        cohort["ecology_measurement_scope"] == analysis.ecology_measurement_scope,
        "Ecology measurement scope differs from the analysis configuration",
    )
    views = id_index(cohort["views"], "modality", "Cohort views")
    modalities = {m.name: m for m in dataset.modalities}
    snapshots = {m.name: m for m in frozen.modalities}
    singleton = {a.modalities[0]: a for a in analysis.arms if a.family != "integrative"}
    kinds = {
        v.modality: v.kind
        for a in analysis.arms
        if a.family == "integrative"
        for v in a.model.views
    }
    require(views.keys() == snapshots.keys() == singleton.keys(), "Archive view identity mismatch")
    links = {}
    report_views = []
    for name, view in views.items():
        source, snapshot = modalities[name], snapshots[name]
        require(
            source.level == snapshot.level == "specimen"
            and source.kind == snapshot.kind == view["kind"] == kinds[name],
            f"{name}: cohort view kind or level mismatch",
        )
        require(
            replace(
                snapshot,
                path=source.path,
                popmapfile=source.popmapfile,
                observations=source.observations,
            )
            == source,
            f"{name}: frozen view declaration differs from the source declaration",
        )
        observations = view["observation_ids"]
        links[name] = dict(zip(ids, observations, strict=True))
        for modality in (source, snapshot):
            require(
                len(modality.observations) == len(ids)
                and {o.unit_id: o.observation_id for o in modality.observations} == links[name],
                f"{name}: inconsistent observation links",
            )
        features = view["feature_names"]
        require(
            features == list(getattr(singleton[name].model, "feature_names", ())),
            f"{name}: inconsistent modeled features",
        )
        report_views.append(
            {
                **{k: v for k, v in view.items() if k != "observation_ids"},
                "measurement_level": source.level,
                "role": "analytical",
                "declared_features": [
                    {
                        "name": f.name,
                        "role": f.role,
                        "unit": f.unit,
                        "type": f.dtype,
                        "included": f.name in features,
                    }
                    for f in source.features
                ],
            }
        )
    return dataset, links, report_views


def _probabilities(values, context):
    require(
        all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        and math.isclose(sum(values), 1, rel_tol=1e-10, abs_tol=1e-12),
        f"{context}: invalid saved membership vector",
    )


def _assignment(label, candidate, reason, maximum, probabilities, zero_based, k):
    membership_status = (
        evidence("unavailable", "Full genomic memberships were not archived.")
        if probabilities is None
        else evidence("available")
    )
    if k == 1:
        alternative = evidence("not_computed", "The selected partition has one component.")
    elif probabilities is None:
        alternative = membership_status
    else:
        other = max((key for key in probabilities if key != candidate), key=probabilities.get)
        alternative = evidence("available", label=other, membership=probabilities[other])
    return {
        "assignment": label,
        "candidate_cluster": candidate,
        "comparison_label": zero_based,
        "status": "assigned" if label is not None else "abstained",
        "reason": reason,
        "maximum_conditional_membership": maximum,
        "conditional_memberships": probabilities,
        "membership_vector_evidence": membership_status,
        "strongest_alternative": alternative,
    }


def _read_point(archive, arm, native, comparison_rows, links, cohort_ids, k):
    name = arm.name
    prefix = f"arms/{name}"
    genomic = arm.family == "genomic"
    combined = arm.family == "integrative"
    order = native["specimen_ids" if genomic else "unit_ids"]
    _ids(order, cohort_ids, prefix)
    if combined:
        bound = id_index(native["views"], "modality", prefix)
        require(bound.keys() == set(arm.modalities), f"{prefix}: inconsistent combined views")
        for modality, row in bound.items():
            members = id_index(row["members"], "target_id", prefix)
            require(
                {u: v["observation_ids"] for u, v in members.items()}
                == {u: [obs] for u, obs in links[modality].items()},
                f"{prefix}: inconsistent combined observation links",
            )
        expected_obs = {unit: unit for unit in order}
    else:
        expected_obs = links[arm.modalities[0]]
        saved = native["sample_ids" if genomic else "observation_ids"]
        require(
            dict(zip(order, saved, strict=True)) == expected_obs,
            f"{prefix}: inconsistent native observation links",
        )
    labels = [f"cluster_{i + 1}" for i in range(k)]
    native_rows, vectors, genomic_labels = None, {}, {}
    if genomic:
        require(native["cluster_labels"] == labels, f"{prefix}: invalid native cluster labels")
        label_data = archive.read(f"{prefix}/labels.npy")
        genomic_labels = dict(zip(order, read_genomic_labels(label_data, len(order)), strict=True))
        data = archive.read(f"{prefix}/membership.npy", optional=True)
        if data is not None:
            vectors = dict(zip(order, read_membership_matrix(data, len(order), k), strict=True))
    else:
        native_rows = archive.csv(
            f"{prefix}/assignments.csv",
            {
                "unit_id",
                "observation_id",
                "candidate_cluster",
                "assignment",
                "reason",
                *(f"membership_{label}" for label in labels),
            },
        )
        require(native_rows.keys() == cohort_ids, f"{prefix}: inconsistent assignment IDs")
    result = archive.json(f"{prefix}/result.json", optional=True)
    if result is not None:
        require(
            result["schema_version"] == 1
            and result["level"] == "specimen"
            and result["dataset_id"] == native["dataset_id"]
            and result["configuration_sha256"] == native["provenance"]["configuration_sha256"],
            f"{prefix}: inconsistent native result identity",
        )
        result_assignments = id_index(result["assignments"], "unit_id", prefix)
        require(result_assignments.keys() == cohort_ids, f"{prefix}: inconsistent result cohort")
    assignments = {}
    for unit in order:
        top = comparison_rows[unit]
        zero_based = int(top[f"{name}__cluster"])
        require(0 <= zero_based < k, f"{prefix}: out-of-range comparison label for {unit}")
        maximum = float(top[f"{name}__maximum_conditional_membership"])
        require(0 <= maximum <= 1, f"{prefix}: invalid maximum membership for {unit}")
        candidate = labels[zero_based]
        label, reason = candidate, None
        context = f"{prefix} specimen {unit}"
        if genomic:
            require(genomic_labels[unit] == zero_based, f"{context}: native genomic label mismatch")
        values = vectors.get(unit)
        if native_rows is not None:
            row = native_rows[unit]
            require(
                row["observation_id"] == expected_obs[unit]
                and row["candidate_cluster"] == candidate,
                f"{context}: native assignment ID or candidate mismatch",
            )
            label, reason = row["assignment"] or None, row["reason"] or None
            values = tuple(float(row[f"membership_{key}"]) for key in labels)
        if result is not None:
            row = result_assignments[unit]
            if genomic:
                label, reason = row["label"], row["reason"]
            else:
                require(
                    (label, reason) == (row["label"], row["reason"]),
                    f"{context}: native result disagrees with assignment CSV",
                )
        require(
            label == candidate or (label is None and bool(reason)),
            f"{context}: invalid native assignment",
        )
        if values is not None:
            _probabilities(values, context)
            require(
                values[zero_based] == maximum == max(values),
                f"{context}: inconsistent conditional membership",
            )
        assignments[unit] = _assignment(
            label,
            candidate,
            reason,
            maximum,
            dict(zip(labels, values, strict=True)) if values is not None else None,
            zero_based,
            k,
        )
    p_values = [row for row in result["evidence"] if row["kind"] == "p_value"] if result else []
    return assignments, p_values, result is not None, order


def _read_arm(archive, arm, declared, saved, cohort_ids, links, comparison_rows, manifest):
    name = arm.name
    status = saved["status"]
    require(
        saved["family"] == arm.family
        and saved["modalities"] == declared["modalities"]
        and saved["configuration"] == declared["model"]
        and status in {"complete", "partial", "failed"},
        f"{name}: inconsistent arm declaration",
    )
    k = saved["selected_k"]
    partition = {
        "name": name,
        "family": arm.family,
        "modalities": declared["modalities"],
        "status": status,
        "selected_k": k,
        "search_boundary": saved["search_boundary"],
        "error": saved["error"],
        "declared_count": len(cohort_ids),
        "resolved_count": 0,
        "groups": {},
    }
    if status == "failed":
        require(
            k is None
            and saved["search_boundary"] is None
            and all(
                row[f"{name}__cluster"] == row[f"{name}__maximum_conditional_membership"] == ""
                for row in comparison_rows.values()
            ),
            f"{name}: failed arm contains point results",
        )
        return partition, {}, []
    require(
        type(k) is int
        and 1 <= k <= arm.model.max_k
        and saved["search_boundary"] is (k == arm.model.max_k),
        f"{name}: inconsistent selected K or search limit",
    )
    prefix = f"arms/{name}"
    native = archive.json(f"{prefix}/summary.json")
    provenance = native["provenance"]
    configuration = {"dataset": manifest["frozen_dataset"], "model": declared["model"]}
    if arm.family != "integrative":
        configuration["modality"] = arm.modalities[0]
    if arm.family == "genomic":
        configuration["blocks"] = (
            None
            if arm.block_ids is None
            else {"schema_version": 1, "block_ids": list(arm.block_ids)}
        )
    require(
        native["schema_version"] == 1
        and native["status"] == "complete"
        and native["dataset_id"] == manifest["dataset"]["dataset_id"]
        and native["selected_k"] == k
        and provenance["configuration"] == configuration
        and provenance["configuration_sha256"] == configuration_hash(configuration)
        and (arm.family == "genomic" or native["level"] == "specimen"),
        f"{prefix}: native run identity or configuration mismatch",
    )
    stability = native["stability" if arm.family == "genomic" else "bootstrap"]
    require(stability == saved["stability"], f"{prefix}: inconsistent stability metadata")
    assignments, p_values, has_result, order = _read_point(
        archive, arm, native, comparison_rows, links, cohort_ids, k
    )
    groups = groups_by_label((unit, row["assignment"]) for unit, row in assignments.items())
    partition.update(
        resolved_count=sum(map(len, groups.values())),
        groups={label: sorted(members) for label, members in sorted(groups.items())},
        stability=stability,
        evidence_type=saved["evidence_type"],
        native_p_values=p_values,
        native_p_value_records=evidence(
            "available" if has_result else "unavailable",
            None if has_result else "The native result evidence file was not archived.",
            record_count=len(p_values) if has_result else None,
        ),
    )
    if arm.family == "integrative":
        unit, row = min(
            assignments.items(),
            key=lambda item: (item[1]["maximum_conditional_membership"], item[0]),
        )
        partition["lowest_maximum_membership"] = {
            "unit_id": unit,
            "membership": row["maximum_conditional_membership"],
        }
    return partition, assignments, order


def _focal_boundary(archive, partitions, assignments, orders, groups, arm_name, specimen_pair):
    candidates = [
        name
        for name, model in partitions.items()
        if model["family"] == "integrative" and len(model["groups"]) >= 2
    ]
    if arm_name is None and candidates:
        arm_name = max(candidates, key=lambda name: len(partitions[name]["modalities"]))
    if arm_name is None and groups is None and specimen_pair is None:
        return evidence("unavailable", "No saved combined partition resolves two groups.")
    require(arm_name in candidates, "Boundary arm must be a resolved combined partition")
    if specimen_pair is not None:
        require(len(specimen_pair) == 2, "Select two specimen IDs")
    point_groups = {
        label: set(members) for label, members in partitions[arm_name]["groups"].items()
    }
    candidate_groups = groups_by_label(
        (unit, row["candidate_cluster"]) for unit, row in assignments[arm_name].items()
    )
    by_unit = {unit: row["assignment"] for unit, row in assignments[arm_name].items()}
    if groups is None:
        if specimen_pair is None:
            units = sorted(unit for unit, label in by_unit.items() if label is not None)
            first = units[0]
            specimen_pair = (first, next(unit for unit in units if by_unit[unit] != by_unit[first]))
        groups = tuple(by_unit.get(unit) for unit in specimen_pair)
    require(
        len(groups) == 2 and groups[0] != groups[1] and all(g in point_groups for g in groups),
        "Boundary needs two distinct resolved native group labels",
    )
    left, right = (point_groups[g] for g in groups)
    if specimen_pair is None:
        specimen_pair = (min(left), min(right))
    require(
        len(specimen_pair) == 2
        and (
            (specimen_pair[0] in left and specimen_pair[1] in right)
            or (specimen_pair[1] in left and specimen_pair[0] in right)
        ),
        "The specimen pair must span the focal groups",
    )
    cross_arms = {}
    for name, rows in assignments.items():
        if rows:
            cross_arms[name] = boundary_agreement(
                {unit: row["assignment"] for unit, row in rows.items()}, left, right
            )
        else:
            cross_arms[name] = evidence("unavailable", "The arm has no point partition.")
    return evidence(
        "available",
        originating_arm=arm_name,
        groups=[{"label": group, "unit_ids": sorted(point_groups[group])} for group in groups],
        specimen_pair=list(specimen_pair),
        descriptive_agreement=evidence(
            "available", "Same-cohort comparison of saved point partitions.", arms=cross_arms
        ),
        stability=recurrence(
            archive,
            arm_name,
            partitions[arm_name]["stability"],
            orders[arm_name],
            candidate_groups,
            {group: point_groups[group] for group in groups},
            specimen_pair,
        ),
        fitted_split_vs_merge=evidence(
            "not_computed", "The native K search did not fit this specific merged hypothesis."
        ),
        independent_validation=evidence("not_computed", "The arms share the discovery cohort."),
    )


def _partition_comparisons(records, partitions, assignments, ids):
    pairs = list(combinations(partitions, 2))
    expected_pairs = set(pairs)
    saved = {}
    require(isinstance(records, list), "Expected a list of native partition comparisons")
    for record in records:
        pair = record["left"], record["right"]
        require(pair in expected_pairs, f"Unexpected native partition comparison: {pair!r}")
        require(pair not in saved, f"Repeated native partition comparison: {pair!r}")
        saved[pair] = record

    comparisons = []
    for left, right in pairs:
        context = f"Native partition comparison {left!r} vs {right!r}"
        require((left, right) in saved, f"{context}: missing from summary")
        record = saved[left, right]
        if not assignments[left] or not assignments[right]:
            expected = {
                "status": "unavailable",
                "reason": "A native point/cohort verification failed",
            }
            agreement = evidence(
                "unavailable",
                "An arm has no point partition.",
                n_units=len(ids),
                left_resolved=partitions[left]["resolved_count"],
                right_resolved=partitions[right]["resolved_count"],
                joint_resolved=0,
                joint_pairs=0,
                discordant_pairs=None,
            )
        else:
            a, b = ([assignments[name][unit] for unit in ids] for name in (left, right))
            expected = compare_partitions(
                [row["comparison_label"] for row in a],
                [row["comparison_label"] for row in b],
                [row["assignment"] is not None for row in a],
                [row["assignment"] is not None for row in b],
            )
            agreement = expected.copy()
            if expected["status"] == "undefined":
                agreement.update(
                    status="unavailable", reason="Fewer than two jointly resolved units"
                )
        for field, value in expected.items():
            require(field in record, f"{context}: missing {field} in summary")
            archived = record[field]
            require(
                type(archived) is type(value) and archived == value,
                f"{context}: {field} saved {archived!r}, recomputed {value!r}",
            )
        require(
            record.keys() == expected.keys() | {"left", "right"},
            f"{context}: unexpected fields in summary",
        )
        comparisons.append({"left": left, "right": right, **agreement})
    return comparisons


def _read_run(archive):
    summary = archive.json("summary.json")
    require(
        summary["schema_version"] == 1 and summary["status"] in {"complete", "partial"},
        "Unsupported or unfinished matched-multimodal run",
    )
    require(
        isinstance(summary["artifacts"], dict) and bool(summary["artifacts"]),
        "Missing or invalid native run digest ledger",
    )
    archive.native_digests = summary["artifacts"]
    manifest = archive.json("manifest.json")
    cohort = archive.json("inputs/cohort.json")
    frozen_dataset = archive.json("inputs/dataset.json")
    require(
        configuration_hash(frozen_dataset) == configuration_hash(manifest["frozen_dataset"]),
        "Saved frozen dataset differs from the run manifest",
    )
    require(
        manifest["schema_version"] == 1 and manifest["cohort"] == cohort,
        "Unsupported run manifest or inconsistent cohort declaration",
    )
    digest = configuration_hash(manifest["analysis"])
    require(
        digest == manifest["configuration_sha256"] == summary["configuration_sha256"],
        "Analysis configuration digest mismatch",
    )
    analysis = MultimodalAnalysisConfig.from_dict(manifest["analysis"])
    dataset, links, views = _verify_cohort(archive, manifest, analysis, cohort)
    ids = cohort["unit_ids"]
    _ids(summary["unit_ids"], set(ids), "Run summary")
    require(
        summary["dataset_id"] == dataset.dataset_id and summary["n_units"] == len(ids),
        "Run dataset or cohort size mismatch",
    )
    declared_arms = id_index(manifest["analysis"]["arms"], "name", "Declared arms")
    saved_arms = id_index(summary["models"], "name", "Saved arms")
    require(declared_arms.keys() == saved_arms.keys(), "Saved comparison arm inventory mismatch")
    statuses = {arm["status"] for arm in saved_arms.values()}
    require(
        bool(statuses & {"complete", "partial"})
        and (summary["status"] == "complete") == (statuses == {"complete"}),
        "Run status disagrees with arm availability",
    )
    rows = archive.csv(
        "assignments.csv",
        {
            "unit_id",
            *(f"{m}__observation_id" for m in links),
            *(
                f"{a}__{suffix}"
                for a in saved_arms
                for suffix in ("cluster", "maximum_conditional_membership")
            ),
        },
    )
    require(rows.keys() == set(ids), "Comparison assignment cohort mismatch")
    partitions, assignments, orders = {}, {}, {}
    for arm in analysis.arms:
        partitions[arm.name], assignments[arm.name], orders[arm.name] = _read_arm(
            archive,
            arm,
            declared_arms[arm.name],
            saved_arms[arm.name],
            rows.keys(),
            links,
            rows,
            manifest,
        )
    specimens = []
    for unit in ids:
        source_ids = {name: view[unit] for name, view in links.items()}
        require(
            all(rows[unit][f"{m}__observation_id"] == obs for m, obs in source_ids.items()),
            f"{unit}: comparison CSV disagrees with observation links",
        )
        specimens.append(
            {
                "unit_id": unit,
                "source_observation_ids": source_ids,
                "unmeasured_views": [],
                "arms": {
                    name: assignments[name].get(unit)
                    or evidence("unavailable", partitions[name]["error"] or "No saved point result")
                    for name in partitions
                },
            }
        )
    comparisons = _partition_comparisons(summary["comparisons"], partitions, assignments, ids)
    document = {
        "schema_version": 1,
        "run": {
            "workflow": "analyze-matched-multimodal",
            "status": summary["status"],
            "dataset_id": dataset.dataset_id,
            "dataset_configuration_sha256": configuration_hash(manifest["dataset"]),
            "analysis_configuration_sha256": digest,
            "study_path": archive.run.relative_to(archive.root).as_posix(),
        },
        "cohort": {
            "level": "specimen",
            "unit_ids": ids,
            "declared_count": len(ids),
            "exclusions": [],
            "independence": cohort["independence"],
            "ecology_measurement_scope": cohort["ecology_measurement_scope"],
            "views": views,
        },
        "partitions": list(partitions.values()),
        "partition_comparisons": comparisons,
        "specimens": specimens,
        "inference": {
            "native_run_inference": summary["inference"],
            "species_boundary_inference": evidence(
                "not_computed", "The native matched analysis defines no species-boundary test."
            ),
            "held_out_gain": evidence(
                "not_computed",
                "This matched archive has no dataset-level held-out diagnostic.",
                experimental=True,
                mean_log_predictive_density_gain=None,
            ),
        },
        "limitations": summary["limitations"],
        "provenance": {
            "study_manifest_path": str(archive.manifest),
            "study_manifest_sha256": archive.manifest_sha256,
            "verified_run_artifacts": archive.verified,
        },
    }

    return document, assignments, orders, manifest


def _report(archive, boundary_arm, boundary_groups, specimen_pair):
    document, assignments, orders, _ = _read_run(archive)
    partitions = {arm["name"]: arm for arm in document["partitions"]}
    document["focal_boundary"] = _focal_boundary(
        archive,
        partitions,
        assignments,
        orders,
        boundary_groups,
        boundary_arm,
        specimen_pair,
    )
    return document


def _fresh_output(output_dir):
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Report destination already exists: {output}")
    return output


def _write_report(document, html, output, filename):
    result = ArchiveReport(output / f"{filename}.json", output / f"{filename}.html")
    output.mkdir(mode=0o700, exist_ok=False)
    try:
        write_json(document, result.json_path)
        with result.html_path.open("x", encoding="utf-8") as handle:
            handle.write(html)
    except BaseException:
        result.json_path.unlink(missing_ok=True)
        result.html_path.unlink(missing_ok=True)
        output.rmdir()
        raise
    return result


def report_from_archive(
    run_dir: str | Path,
    *,
    study_manifest: str | Path,
    output_dir: str | Path,
    boundary_arm: str | None = None,
    boundary_groups: tuple[str, str] | None = None,
    specimen_pair: tuple[str, str] | None = None,
) -> ArchiveReport:
    """Verify sealed inputs and write a local JSON document and offline HTML report.

    By default, use the widest saved combined arm that resolves two groups.
    Within that arm, start with the first specimen in ID order and the first
    specimen assigned to a different group. Callers may choose the arm, the
    two native group labels, and a specimen pair spanning those groups.
    """
    output = _fresh_output(output_dir)
    archive = SealedRun(Path(run_dir), Path(study_manifest))
    require(
        not output.resolve().is_relative_to(archive.run),
        "Report destination must be outside the sealed run directory",
    )
    try:
        document = _report(archive, boundary_arm, boundary_groups, specimen_pair)
        html = render_html(document)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ArchiveError(f"Malformed matched-multimodal archive: {exc}") from exc
    return _write_report(document, html, output, "report")
