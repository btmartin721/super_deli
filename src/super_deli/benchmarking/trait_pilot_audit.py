"""Independent fit-free accounting from retained source cells and reference ledgers.

Does not import the pilot protocol, selection or reporting implementation.
"""

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def _read(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    return json.loads(path.read_text(encoding="utf-8"))


def audit_trait_pilot(directory):
    if not __debug__:
        raise RuntimeError("Independent audit requires Python assertion checks")
    root = Path(directory)
    manifest = _read(root / "checksums.json")
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }
    expected = {k: v for k, v in manifest.items() if not k.endswith("checksums.json")}
    if files != expected or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Pilot evidence differs from complete artifact inventory")
    # Nested inventories themselves must also match their listed hashes.
    for name, digest in manifest.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Pilot artifact hash mismatch")
    audit = root / "evidence-reconstruction/prior-reconstruction"
    cells = [r for r in _read(audit / "cells.jsonl.gz") if r["sheet"] == "Data"]
    header = {k: v["value"] for k, v in cells[0]["cells"].items()}
    raw = {r["row"]: {header[k]: v for k, v in r["cells"].items()} for r in cells[1:]}
    source = _read(audit / "source-records.jsonl.gz")
    references = _read(root / "reference-evidence.jsonl.gz")
    source_ids = [r["source_row"] for r in source]
    assert len(raw) == len(source) == len(references) == len(set(source_ids))
    assert source_ids == [r["source_row"] for r in references]
    ref_index = {r["source_row"]: r for r in references}
    missing = {"", "NA", "unknown"}
    core = (
        "body_l antenna_I antenna_II gnI_carp gnI_pl gnI_pw gnI_pd cx_II "
        "gnII_carp gnII_pl gnII_pw gnII_pd cx_III ppV_bw ppV_l ppVI_bw "
        "ppVI_l ppVII_bw ppVII_l"
    ).split()
    panels = {
        "core19": core,
        "somatic24": core
        + (
            "upI_basis upI_exopodite upI_endopodite upIII_exopodite_prox upIII_exopodite_dist"
        ).split(),
    }
    reasons = [
        "ambiguous_identity",
        "outside_family",
        "outside_sex_stage",
        "formula_measurement",
        "uncertain_measurement",
        "missing_measurement",
        "nonpositive_measurement",
    ]
    design = _read(root / "frozen-design.json")
    assert design["panels"] == panels and design["exclusion_order"] == reasons
    counts = Counter(
        r["values"]["voucher"] for r in source if r["values"]["voucher"] not in missing
    )
    summary = _read(root / "summary.json")
    independently_checked = 0
    for panel, features in panels.items():
        for sex in ("male", "female"):
            key = panel + "_" + sex
            ledger = _read(root / (key + ".jsonl.gz"))
            assert [r["source_row"] for r in ledger] == source_ids
            chosen = []
            expected_flags = []
            numeric = {}
            for original, observed in zip(source, ledger, strict=True):
                number = original["source_row"]
                row = raw[number]

                def get(name, fields=row):
                    return fields.get(name, {}).get("value", "")

                voucher = get("voucher")
                # The unchanged source audit owns identifier syntax validation.
                # Independently check unique voucher counts and stable hashes.
                unique = original["identity_status"] == "unique"
                if unique:
                    assert counts[voucher] == 1 and voucher not in missing
                    assert (
                        original["unit_id"]
                        == "v_" + hashlib.sha256(voucher.encode()).hexdigest()[:20]
                    )
                flags = {r: [] for r in reasons}
                if not unique:
                    flags["ambiguous_identity"] = [original["identity_status"]]
                if get("family") != "Niphargidae":
                    flags["outside_family"] = [get("family")]
                if get("sex") != sex:
                    flags["outside_sex_stage"] = [get("sex")]
                values = {}
                for name in features:
                    cell = row.get(name, {})
                    token = get(name)
                    if cell.get("formula") is not None:
                        flags["formula_measurement"].append(name)
                    invalid = False
                    value = None
                    if cell.get("formula") is not None and not cell.get("cache_present", False):
                        invalid = True
                    elif token not in missing:
                        try:
                            if token != token.strip() or cell.get("type") in ("e", "b", "d"):
                                raise ValueError("nonmeasurement")
                            value = float(token)
                            if not math.isfinite(value):
                                raise ValueError("nonfinite")
                        except ValueError:
                            invalid = True
                    if invalid:
                        flags["uncertain_measurement"].append(name)
                    elif value is None:
                        flags["missing_measurement"].append(name)
                    elif value <= 0:
                        flags["nonpositive_measurement"].append(name)
                    values[name] = value
                active = [r for r in reasons if flags[r]]
                assert observed == dict(
                    source_row=number,
                    unit_id=original["unit_id"],
                    included=not active,
                    reasons=active,
                    reason_details=flags,
                    primary_exclusion=active[0] if active else None,
                    raw_selected_values={f: get(f) for f in features},
                )
                expected_flags.append(flags)
                if not active:
                    chosen.append(original)
                    numeric[number] = values
            s = summary["rosters"][key]
            n = len(chosen)
            refs = [ref_index[r["source_row"]] for r in chosen]
            vectors = Counter(tuple(numeric[r["source_row"]][f] for f in features) for r in chosen)
            partitions = {}
            for r in refs:
                if r["molecular_partition_link"]:
                    for method, labels in r["raw_molecular_partitions"].items():
                        for label in labels if isinstance(labels, list) else [labels]:
                            counter = partitions.setdefault(method, Counter())
                            counter[label] += 1

            def count(values):
                return dict(sorted(Counter(values).items()))

            calculated = dict(
                source_rows=len(source),
                included=n,
                excluded=len(source) - n,
                first_exclusion_counts=count(
                    next(k for k in reasons if fs[k]) for fs in expected_flags if any(fs.values())
                ),
                overlapping_reason_counts={
                    k: sum(bool(fs[k]) for fs in expected_flags) for k in reasons
                },
                per_trait_exclusions={
                    f: {k: sum(f in fs[k] for fs in expected_flags) for k in reasons[3:]}
                    for f in features
                },
                constant_traits=[
                    f
                    for f in features
                    if n and len({numeric[r["source_row"]][f] for r in chosen}) == 1
                ],
                duplicate_numeric_vector_groups=sum(v > 1 for v in vectors.values()),
                duplicate_numeric_vector_rows=sum(v for v in vectors.values() if v > 1),
                nominal_pca_rank_upper_bound=min(max(n - 1, 0), len(features)),
                size_five_group_count_upper_bound=n // 5,
                bounds_are_not_power_or_effective_sample_size=True,
                locality_token_rows=sum(
                    r["values"]["locality_name"] not in missing for r in chosen
                ),
                date_token_rows=sum(r["values"]["survey_date"] not in missing for r in chosen),
                locality_tokens=count(r["values"]["locality_name"] for r in chosen),
                historical_motu_tokens=count(r["historical_motu"] for r in refs),
                correspondence_counts=count(r["correspondence_status"] for r in refs),
                documentary_specimen_links=sum(r["documented_specimen_link"] for r in refs),
                molecular_partition_links=sum(r["molecular_partition_link"] for r in refs),
                raw_partition_label_counts={
                    k: dict(sorted(v.items())) for k, v in sorted(partitions.items())
                },
                reference_cautioned_rows=sum(bool(r["source_cautions"]) for r in refs),
                conflicted_reference_rows=sum(
                    bool(r["genbank_conflict"] or r["contradictory_voucher_records"]) for r in refs
                ),
                source_only_morphology_roster="available" if n else "empty_under_fixed_policy",
                descriptive_concordance_feasibility=(
                    "documented_links_available_future_design_only"
                    if any(r["molecular_partition_link"] for r in refs)
                    else "no_linked_partition_rows_under_fixed_policy"
                ),
            )
            assert s == calculated, key
            independently_checked += len(ledger)
    assert summary["outcome"] == "descriptive_rosters_assessed_inferential_pilot_no_go"
    for key in (
        "independent_unit_fitting",
        "grouped_predictive_screen",
        "resampling",
        "significance",
        "species_boundary_validation",
    ):
        assert summary["readiness"][key]["status"] == "blocked"
    for key in (
        "fitting_authorized",
        "sampling_independence_asserted",
        "reference_species_truth_asserted",
        "contact_performed",
        "executable_analysis_exported",
    ):
        assert summary[key] is False
    for key in ("empirical_fits", "bootstrap_draws", "pvalues_generated", "source_downloads"):
        assert summary[key] == 0
    assert summary["source_rows"] == len(source)
    return dict(
        status="passed",
        source_rows=len(source),
        roster_row_decisions=independently_checked,
        all_roster_summary_fields_independently_recomputed=True,
        raw_cell_provenance_checked=True,
        inferential_gates_closed=True,
        new_selection_or_workflow_imported=False,
        empirical_fits=0,
    )
