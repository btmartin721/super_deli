"""Label-blind raw-measurement selection and descriptive feasibility accounting."""

import math
from collections import Counter

from super_deli.core.trait_pilot import PANELS, REASONS, SEXES
from super_deli.io.trait_source import MISSING


def select_rosters(rows, formulas):
    """Select using identity, family, sex/stage and selected raw measurements only."""
    result = {}
    for panel, features in PANELS.items():
        for sex in SEXES:
            records = []
            for row in rows:
                selected = {name: row["values"][name] for name in features}
                flags = {reason: [] for reason in REASONS}
                if row["identity_status"] != "unique" or not row["unit_id"]:
                    flags["ambiguous_identity"] = [row["identity_status"]]
                if row["values"]["family"] != "Niphargidae":
                    flags["outside_family"] = [row["values"]["family"]]
                if row["values"]["sex"] != sex:
                    flags["outside_sex_stage"] = [row["values"]["sex"]]
                for name in features:
                    value = row["numeric_values"][name]
                    if (row["source_row"], name) in formulas:
                        flags["formula_measurement"].append(name)
                    if name in row["numeric_issues"]:
                        flags["uncertain_measurement"].append(name)
                    elif value is None:
                        flags["missing_measurement"].append(name)
                    elif not math.isfinite(value):
                        raise ValueError("Reconstructed numeric values must be finite")
                    elif value <= 0:
                        flags["nonpositive_measurement"].append(name)
                reasons = [reason for reason in REASONS if flags[reason]]
                records.append(
                    dict(
                        source_row=row["source_row"],
                        unit_id=row["unit_id"],
                        included=not reasons,
                        reasons=reasons,
                        reason_details=flags,
                        primary_exclusion=reasons[0] if reasons else None,
                        raw_selected_values=selected,
                    )
                )
            result[panel + "_" + sex] = records
    return result


def _counts(values):
    return dict(sorted(Counter(values).items()))


def describe_roster(records, rows, evidence, features):
    """Attach evidence after selection; counts are not species-accuracy metrics."""
    chosen = [r for r in records if r["included"]]
    source = {r["source_row"]: r for r in rows}
    linked = {r["source_row"]: r for r in evidence}
    selected = [source[r["source_row"]] for r in chosen]
    refs = [linked[r["source_row"]] for r in chosen]
    vectors = Counter(tuple(r["numeric_values"][f] for f in features) for r in selected)
    partitions = {}
    for ref in refs:
        if not ref["molecular_partition_link"]:
            continue
        for method, label in ref["raw_molecular_partitions"].items():
            # Retain all documented labels, including multiple labels for a method.
            values = label if isinstance(label, list) else [label]
            partitions.setdefault(method, Counter()).update(values)
    n = len(chosen)
    return dict(
        source_rows=len(records),
        included=n,
        excluded=len(records) - n,
        first_exclusion_counts=_counts(
            r["primary_exclusion"] for r in records if not r["included"]
        ),
        overlapping_reason_counts={
            reason: sum(reason in r["reasons"] for r in records) for reason in REASONS
        },
        per_trait_exclusions={
            f: {
                reason: sum(f in r["reason_details"][reason] for r in records)
                for reason in REASONS[3:]
            }
            for f in features
        },
        constant_traits=[
            f for f in features if n and len({r["numeric_values"][f] for r in selected}) == 1
        ],
        duplicate_numeric_vector_groups=sum(v > 1 for v in vectors.values()),
        duplicate_numeric_vector_rows=sum(v for v in vectors.values() if v > 1),
        nominal_pca_rank_upper_bound=min(max(n - 1, 0), len(features)),
        size_five_group_count_upper_bound=n // 5,
        bounds_are_not_power_or_effective_sample_size=True,
        locality_token_rows=sum(r["values"]["locality_name"] not in MISSING for r in selected),
        date_token_rows=sum(r["values"]["survey_date"] not in MISSING for r in selected),
        locality_tokens=_counts(r["values"]["locality_name"] for r in selected),
        historical_motu_tokens=_counts(r["historical_motu"] for r in refs),
        correspondence_counts=_counts(r["correspondence_status"] for r in refs),
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
