"""Audit specimen sources without inference, alignment, clustering or calibration."""

import hashlib
import itertools
from collections import Counter
from dataclasses import replace
from pathlib import Path

from super_deli.core.data import (
    DatasetConfig,
    Feature,
    Modality,
    MorphometricReaderOptions,
    Observation,
    Specimen,
    TabularReaderOptions,
)
from super_deli.core.provenance import configuration_hash
from super_deli.core.replacement_audit import ReplacementAuditPlan
from super_deli.core.validation import validate_dataset
from super_deli.io.config import read_json, write_json
from super_deli.io.landmark_csv import read_landmark_csv
from super_deli.io.specimen_audit import (
    COLOUR,
    genbank_references,
    inspect_tps,
    source_tables,
    supplement_membership,
)
from super_deli.io.tabular import _read_table as read_table
from super_deli.io.tps import read_tps
from super_deli.workflows.empirical_curation import write_rows

ARCHIVE_NAMES = dict(
    metadata="metadata_full.csv",
    colour="banding_data.csv",
    full_disc="FBODY.tps",
    anterior="ADM.tps",
    rostrum="RS.tps",
    author_code="main_script.R",
    readme="README.md",
)


def verify_sources(plan, directory):
    root = Path(directory)
    paths, hashes = {}, {}
    for s in plan.sources:
        p = root / s.filename
        if p.is_symlink():
            raise ValueError("Source locks cannot follow symbolic links")
        with p.open("rb") as h:
            raw = h.read(s.size_bytes + 1)
        if len(raw) != s.size_bytes or hashlib.sha256(raw).hexdigest() != s.sha256:
            raise ValueError("Source size/hash mismatch for " + s.role)
        paths[s.role], hashes[s.role] = p, s.sha256
    if sum(s.size_bytes for s in plan.sources) > 16 * 1024**2:
        raise ValueError("Source bundle exceeds 16 MiB")
    return paths, hashes


def archive_audit(plan, paths):
    doc = read_json(paths["archive_metadata"])
    container = read_json(paths["archive_files"])
    if not isinstance(doc, dict) or not isinstance(container, dict):
        raise ValueError("Structured archive metadata/inventory required")
    embedded = container.get("_embedded")
    inventory = embedded.get("stash:files") if isinstance(embedded, dict) else None
    if not isinstance(inventory, list) or any(
        not isinstance(e, dict)
        or not {"path", "size", "digestType", "digest"} <= e.keys()
        or not isinstance(e["path"], str)
        for e in inventory
    ):
        raise ValueError("Explicit archive file inventory required")
    entries = {x["path"]: x for x in inventory}
    if len(entries) != len(inventory) or set(entries) != set(ARCHIVE_NAMES.values()):
        raise ValueError("Unexpected archive file inventory")
    for source in plan.sources:
        if source.role in ARCHIVE_NAMES:
            entry = entries[ARCHIVE_NAMES[source.role]]
            if (
                entry["size"] != source.size_bytes
                or entry["digestType"] != "sha-256"
                or entry["digest"] != source.sha256
            ):
                raise ValueError("Archive inventory does not bind source bytes")
    if doc.get("storageSize") != sum(e["size"] for e in entries.values()):
        raise ValueError("Archive storage size differs from locked files")
    if doc.get("license") != "https://spdx.org/licenses/CC0-1.0.html":
        raise ValueError("Archive license must be explicitly CC0")
    return {
        k: doc.get(k)
        for k in (
            "identifier",
            "versionNumber",
            "versionStatus",
            "curationStatus",
            "publicationDate",
            "visibility",
            "license",
        )
    }


def reference_links(metadata, references, plan):
    """An observed prefix match remains a candidate, never an authoritative join."""
    taxa = dict(plan.taxon_names)
    rows = []
    for unit, row in sorted(metadata.items()):
        exact = [r for r in references if r["voucher"] == unit]
        candidates = [
            r
            for r in references
            if any(
                r["voucher"].startswith(p) and r["voucher"][len(p) :] == unit
                for p in plan.voucher_prefixes
            )
        ]
        matches = exact if exact else candidates
        status = (
            "ambiguous"
            if len(matches) > 1
            else "literal_voucher"
            if exact
            else "candidate_namespace"
            if candidates
            else "unmatched"
        )
        rows.append(
            dict(
                source_id=unit,
                status=status,
                accession_versions=[r["version"] for r in matches],
                vouchers=[r["voucher"] for r in matches],
                organism_agreement=(
                    len(matches) == 1 and taxa.get(row["species"]) == matches[0]["organism"]
                ),
                mapping_authoritative=(len(exact) == 1),
            )
        )
    return rows


def colour_accounting(colours):
    discrepancies = []
    for unit, row in sorted(colours.items()):
        body, band = float(row["area_body"]), float(row["area_band"])
        if row["per_band"] != "-":
            error = float(row["per_band"]) - 100 * band / body
            if abs(error) > 1e-8:
                discrepancies.append(
                    dict(source_id=unit, relationship="100*area_band/area_body", difference=error)
                )
        if all(row[n] != "-" for n in ("mean_grey_body", "mean_grey_band", "per_dif_grey")):
            grey = float(row["mean_grey_body"])
            if grey == 0:
                discrepancies.append(
                    dict(source_id=unit, relationship="grey_percentage", reason="zero_denominator")
                )
            else:
                error = (
                    float(row["per_dif_grey"]) - 100 * (grey - float(row["mean_grey_band"])) / grey
                )
                if abs(error) > 1e-8:
                    discrepancies.append(
                        dict(
                            source_id=unit,
                            relationship="100*(body_grey-band_grey)/body_grey",
                            difference=error,
                        )
                    )
    return dict(
        rows=len(colours),
        missing_by_column={n: sum(r[n] == "-" for r in colours.values()) for n in COLOUR},
        zero_band_area=sum(float(r["area_band"]) == 0 for r in colours.values()),
        zero_band_area_with_missing_gray=sum(
            float(r["area_band"]) == 0 and r["mean_grey_band"] == "-" for r in colours.values()
        ),
        positive_band_area_with_missing_gray=sum(
            float(r["area_band"]) > 0 and r["mean_grey_band"] == "-" for r in colours.values()
        ),
        complete_rows=sum(all(r[n] != "-" for n in COLOUR) for r in colours.values()),
        derived_relationships=["100*area_band/area_body", "100*(body_grey-band_grey)/body_grey"],
        relationship_tolerance=1e-8,
        discrepancies=discrepancies,
        transformation="none; zeroes, missing tokens and redundant features retained",
    )


def membership_audit(metadata, colours, geometry, supplementary):
    views = {"full_disc": {r["image"] for r in geometry["full_disc"]}, "colour": set(colours)}
    details = {}
    for name, rows in geometry.items():
        ids = {r["image"] for r in rows}
        # Diagnostic stem matching is segregated from literal source correspondence.
        stems = {Path(i).stem for i in ids}
        details[name] = dict(
            records=len(rows),
            literal_metadata_matches=len(ids & metadata.keys()),
            unmatched_literal_ids=sorted(ids - metadata.keys()),
            candidate_image_stem_matches=len(stems & metadata.keys()),
            fixed_landmarks_per_record=len(rows[0]["fixed"]),
            curve_points_per_record=sum(len(c) for c in rows[0]["curves"]),
            absent_scales=sum("SCALE" not in r["fields"] for r in rows),
        )
    flags = {
        row[0]: dict(zip(("FD", "ADM", "HR", "VC"), row[4:], strict=True)) for row in supplementary
    }
    comparisons = {}
    for flag, name in (
        ("FD", "full_disc"),
        ("ADM", "anterior"),
        ("HR", "rostrum"),
        ("VC", "colour"),
    ):
        expected = {unit for unit, fields in flags.items() if fields[flag] == "Y"}
        actual = (
            set(colours)
            if name == "colour"
            else {
                r["image"] if name == "full_disc" else Path(r["image"]).stem for r in geometry[name]
            }
        )
        comparisons[name] = dict(
            expected_yes=len(expected),
            observed_records=len(actual),
            expected_missing=sorted(expected - actual),
            observed_not_expected=sorted(actual - expected),
            join_role="literal" if name in ("full_disc", "colour") else "candidate_image_stem_only",
        )
    return dict(
        metadata_records=len(metadata),
        colour_records=len(colours),
        geometry=details,
        colour_unmatched_ids=sorted(set(colours) - metadata.keys()),
        literal_shape_colour_overlap=len(views["full_disc"] & views["colour"]),
        complete_colour_shape_overlap=len(
            views["full_disc"] & {s for s, r in colours.items() if all(r[n] != "-" for n in COLOUR)}
        ),
        specimens_without_full_disc_or_colour=sorted(metadata.keys() - set.union(*views.values())),
        all_pairwise_view_overlap={
            a + "__" + b: len(views[a] & views[b]) for a, b in itertools.combinations(views, 2)
        },
        supplement_rows=len(supplementary),
        supplement_membership=comparisons,
    )


def write_manifests(out, metadata, colours, geometry, audit_id):
    """Keep all declared specimens; absent observations do not become zero rows."""
    analysis = out / "analysis"
    analysis.mkdir()
    splits = analysis / "full_disc_tps"
    splits.mkdir()
    units = {
        s: "unit_" + hashlib.sha256((audit_id + "\0" + s).encode()).hexdigest()[:24]
        for s in metadata
    }
    if len(set(units.values())) != len(units):
        raise ValueError("Anonymous specimen identity collision")
    shape = sorted(
        (r for r in geometry["full_disc"] if r["image"] in metadata),
        key=lambda r: units[r["image"]],
    )
    colour = [(unit, row) for unit, row in sorted(colours.items()) if unit in metadata]
    landmark_ids = tuple(f"fixed_{i:02d}" for i in range(1, 10))
    options = MorphometricReaderOptions(
        dimensions=2, landmark_ids=landmark_ids, coordinate_unit="pixel", tps_id_field="IMAGE"
    )
    coordinates = []
    manifest = []
    for i, record in enumerate(shape):
        unit = units[record["image"]]
        observation = "shape_" + unit
        for name, point in zip(landmark_ids, record["fixed"], strict=True):
            coordinates.append(
                (
                    observation,
                    unit,
                    name,
                    *point,
                    record["image"],
                    record["fields"].get("SCALE", ""),
                )
            )
        filename = f"full_disc_tps/record_{i:04d}.tps"
        text = (
            "LM=9\n"
            + "\n".join(" ".join(p) for p in record["fixed"])
            + "\nIMAGE="
            + record["image"]
            + "\n"
        )
        if "SCALE" in record["fields"]:
            text += "SCALE=" + record["fields"]["SCALE"] + "\n"
        (analysis / filename).write_text(text)
        modality = Modality(
            name="full_disc",
            kind="morphometric",
            level="specimen",
            format="tps",
            path=filename,
            observations=(Observation(record["image"], unit),),
            morphometric_options=options,
        )
        (parsed,) = read_tps(analysis / filename, modality)
        if parsed.coordinates != tuple(tuple(map(float, p)) for p in record["fixed"]):
            raise ValueError("Fixed TPS derivative differs from raw coordinates")
        manifest.append((observation, unit, filename, record["image"]))
    write_rows(
        analysis / "landmarks.csv",
        ["observation_id", "unit_id", "landmark_id", "x", "y", "image", "scale_factor"],
        coordinates,
    )
    write_rows(
        analysis / "tps_manifest.csv",
        ["landmark_csv_observation_id", "unit_id", "path", "tps_image_observation_id"],
        manifest,
    )
    write_rows(
        analysis / "colour.csv",
        ["observation_id", *COLOUR],
        [("colour_" + units[s], *(row[n] for n in COLOUR)) for s, row in colour],
    )
    modalities = []
    if shape:
        m = Modality(
            name="full_disc",
            kind="morphometric",
            level="specimen",
            format="landmarks_csv",
            path="landmarks.csv",
            observations=tuple(
                Observation("shape_" + units[r["image"]], units[r["image"]]) for r in shape
            ),
            morphometric_options=replace(options, tps_id_field="ID"),
        )
        read_landmark_csv(analysis / m.path, m)
        modalities.append(m)
    if colour:
        m = Modality(
            name="colour",
            kind="morphological",
            level="specimen",
            format="csv",
            path="colour.csv",
            observations=tuple(Observation("colour_" + units[s], units[s]) for s, _ in colour),
            features=tuple(
                Feature(
                    n,
                    "continuous",
                    "source_pixel_area"
                    if n.startswith("area_")
                    else "percent"
                    if n.startswith("per_")
                    else "greyscale_0_255",
                )
                for n in COLOUR
            ),
            tabular_options=TabularReaderOptions(missing_values=("-",)),
        )
        read_table(analysis / m.path, m)
        modalities.append(m)
    if not modalities:
        raise ValueError("No literally matched observations for an audit manifest")
    dataset = DatasetConfig(
        dataset_id=audit_id,
        modalities=tuple(modalities),
        specimens=tuple(Specimen(units[s]) for s in sorted(metadata)),
    )
    report = validate_dataset(dataset, base_dir=analysis)
    write_json(report.to_dict(), out / "manifest-validation.json")
    if not report.valid:
        raise ValueError(
            "Existing-format audit manifests failed validation: "
            + "; ".join(i.message for i in report.issues if i.severity == "error")
        )
    write_json(dataset.to_dict(), analysis / "dataset.json")
    evaluation = out / "evaluation"
    evaluation.mkdir()
    write_rows(
        evaluation / "source_labels.csv",
        ["specimen_id", "source_species"],
        [(units[s], row["species"]) for s, row in sorted(metadata.items())],
    )
    write_rows(
        evaluation / "source_covariates.csv",
        ["specimen_id", "length_cm", "sex"],
        [(units[s], row["length"], row["sex"]) for s, row in sorted(metadata.items())],
    )
    write_rows(
        out / "source_identity.csv",
        ["specimen_id", "source_id"],
        [(units[s], s) for s in sorted(metadata)],
    )
    return dict(
        status="audit_only",
        declared_specimens=len(metadata),
        full_disc_observations=len(shape),
        colour_observations=len(colour),
        geometry_tokens=sum(len(r["fixed"]) * 2 for r in shape),
        colour_tokens=sum(row[n] != "-" for _, row in colour for n in COLOUR),
        scale_unit=None,
        scale_values_rescaled=False,
        variables_payload="retained_in_geometry_audit_only",
        validation=report.to_dict(),
    )


def audit_replacement(plan, *, source_dir, output_dir):
    plan = ReplacementAuditPlan.from_dict(plan.to_dict())
    paths, before = verify_sources(plan, source_dir)
    out = Path(output_dir).absolute()
    source_root = Path(source_dir).resolve()
    if out.resolve().is_relative_to(source_root) or source_root.is_relative_to(out.resolve()):
        raise ValueError("Audit outputs must be separate from sources")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_json(plan.to_dict(), out / "specification.json")
        archive = archive_audit(plan, paths)
        metadata, colours = source_tables(paths["metadata"], paths["colour"])
        geometry = {
            name: inspect_tps(paths[name], landmarks=n, curves=c)
            for name, n, c in (("full_disc", 9, 0), ("anterior", 5, 4), ("rostrum", 7, 2))
        }
        refs = genbank_references(paths["references"])
        supplement = supplement_membership(paths["supplement"])
        links = reference_links(metadata, refs, plan)
        membership = membership_audit(metadata, colours, geometry, supplement)
        write_json(dict(metadata=metadata, colour=colours), out / "source-tables.json")
        write_json(geometry, out / "geometry-audit.json")
        write_json(refs, out / "molecular-records.json")
        write_json(links, out / "reference-links.json")
        write_json(membership, out / "membership-audit.json")
        write_json(colour_accounting(colours), out / "colour-audit.json")
        manifests = write_manifests(out, metadata, colours, geometry, plan.audit_id)
        write_json(manifests, out / "manifest-audit.json")
        counts = dict(Counter(x["status"] for x in links))
        summary = dict(
            schema_version=1,
            status="audited",
            phase=plan.phase,
            audit_id=plan.audit_id,
            specification_sha256=configuration_hash(plan.to_dict()),
            archive=archive,
            metadata_records=len(metadata),
            reference_records=len(refs),
            source_label_counts=dict(
                sorted(Counter(r["species"] for r in metadata.values()).items())
            ),
            reference_link_counts=counts,
            candidate_organism_agreements=sum(
                x["status"] == "candidate_namespace" and x["organism_agreement"] for x in links
            ),
            literal_authoritative_references=sum(x["mapping_authoritative"] for x in links),
            view_reference_linkage={
                name: dict(Counter(x["status"] for x in links if x["source_id"] in ids))
                for name, ids in {
                    "full_disc": {g["image"] for g in geometry["full_disc"]},
                    "colour": set(colours),
                }.items()
            },
            full_disc_records=len(geometry["full_disc"]),
            colour_records=len(colours),
            shape_colour_overlap=membership["literal_shape_colour_overlap"],
            readiness=dict(
                typed_ingestion="available_audit_manifests",
                published_label_description="available_development_only",
                independent_reference_concordance="conditional_voucher_crosswalk_required",
                independent_unit_fitting="blocked_no_approved_sampling_design",
                grouped_resampling="blocked_no_authoritative_specimen_haul_links",
                independent_holdout="blocked_haul_species_confounding",
                fixed_shape=(
                    "available_representation_only"
                    if manifests["full_disc_observations"]
                    else "blocked_no_literal_metadata_join"
                ),
                physical_scale="blocked_unit_unverified",
                curve_shape="inapplicable_fixed_landmark_pipeline",
                significance="not_authorized",
            ),
            requirements=[
                "authoritative voucher-to-source-ID crosswalk",
                "specimen-to-haul mapping and justified dependence-aware analysis design",
                "verified physical units for SCALE before physical measurements",
                "separate frozen point-evaluation protocol",
            ],
            interpretation=(
                "Molecular reference provenance is separate from specimen-link "
                "verification and biological species truth"
            ),
            empirical_fits=0,
            pvalues_generated=0,
            stability_draws=0,
            significance_draws=0,
        )
        _, after = verify_sources(plan, source_dir)
        if after != before:
            raise ValueError("Source bundle changed during audit")
        pending = out / "summary.pending.json"
        write_json(summary, pending)
        hashes = {
            p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file() and p != pending
        }
        hashes["summary.json"] = hashlib.sha256(pending.read_bytes()).hexdigest()
        write_json(hashes, out / "checksums.json")
        pending.rename(out / "summary.json")
        return summary
    except Exception as exc:
        write_json(dict(status="failed", error=str(exc)), out / "failure.json")
        raise
