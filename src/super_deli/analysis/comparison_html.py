"""Render the comparison document for local, offline inspection."""

import json

from .report_html import _counts, _status, _table, _text


def _details(label, content):
    return f"<details><summary>{_text(label)}</summary>{content}</details>"


def _json(value):
    return f"<pre>{_text(json.dumps(value, indent=2, ensure_ascii=False))}</pre>"


def _candidate(candidate):
    coverage = candidate["coverage"]
    groups = _table(
        ("Group", "Size", "Members"),
        (
            (group["label"], group["size"], ", ".join(group["unit_ids"]))
            for group in coverage["groups"]
        ),
    )
    exclusions = _table(
        ("Status", "Reason", "Count", "Excluded IDs"),
        (
            (row["status"], row["reason"], row["count"], ", ".join(row["unit_ids"]))
            for row in coverage["exclusions"]
        ),
    )
    return (
        f"<h3>{_text(candidate['id'])}</h3>"
        f"<p>{coverage['resolved_units']}/{coverage['declared_units']} specimens resolved "
        f"in {coverage['resolved_groups']} groups.</p>{groups}"
        + (_details("Excluded assignments", exclusions) if coverage["exclusions"] else "")
        + _details("Source and provenance", _json(candidate["source"]))
    )


def _stability(stability):
    summary = []
    measures = []
    for arm, row in stability.get("arms", {}).items():
        summary.append(
            (
                arm,
                _status(row),
                row.get("resampling_unit"),
                row.get("requested"),
                row.get("successful"),
                row.get("failed"),
                row.get("missing"),
                row.get("covered"),
            )
        )
        if "exact_group_recurrence" not in row:
            continue
        items = [
            *(
                (f"Exact recurrence of {label}", counts)
                for label, counts in row["exact_group_recurrence"].items()
            ),
            ("Both exact groups recur", row["joint_exact_group_recurrence"]),
            (
                "Specimen pair separated across all projected draws",
                row["specimen_pair_separation"]["all_projected_draws"],
            ),
            (
                "Specimen pair separated when both were omitted",
                row["specimen_pair_separation"]["both_specimens_omitted"],
            ),
        ]
        measures.extend((arm, label, _counts(counts), _status(counts)) for label, counts in items)
    return (
        f"<p>{_text(_status(stability))}</p>"
        + (
            _table(
                (
                    "Arm",
                    "Status",
                    "Resampling unit",
                    "Requested",
                    "Successful",
                    "Failed",
                    "Missing records",
                    "Full roster covered",
                ),
                summary,
            )
            if summary
            else ""
        )
        + (
            _table(("Arm", "Saved measure", "Count / covered draws", "Status"), measures)
            if measures
            else ""
        )
    )


def _focal(boundary):
    if boundary["status"] != "available":
        return f"<p>{_text(_status(boundary))}</p>"
    groups = boundary["groups"]
    members = _table(
        ("Focal group", "Specimens"),
        ((row["label"], ", ".join(row["unit_ids"])) for row in groups),
    )
    arms = _table(
        (
            "Arm",
            "Cross-group pairs separated",
            *(f"{row['label']} pairs kept" for row in groups),
        ),
        (
            (
                arm,
                _counts(row["cross_group_separation"]),
                *(_counts(counts) for counts in row["within_group_cohesion"]),
            )
            for arm, row in boundary["arms"].items()
        ),
    )
    return (
        f"<p>Split candidate {_text(boundary['split_hypothesis'])}. "
        f"Merged candidate {_text(boundary['merged_hypothesis'])}. "
        f"Originating archive arm {_text(boundary['originating_arm'])}.</p>"
        f"<p>Example pair {_text(' and '.join(boundary['specimen_pair']))}. "
        f"{_text(boundary['specimen_pair_selection'])}</p>{members}{arms}"
    )


def _contrast(row):
    coverage = row["coverage"]
    metrics = row["partition_comparison"]
    relation = metrics["relation"].replace("_", " ")
    overlap = _table(
        ("Left group", "Right group", "Joint members"),
        (
            (cell["left_group"], cell["right_group"], cell["units"])
            for cell in metrics["member_overlap"]
        ),
    )
    agreement = _table(
        (
            "Arm",
            "Together only on left, comparable",
            "Together only on right, comparable",
            "Agree with left",
            "Agree with right",
            "Interpretation",
        ),
        (
            (
                arm,
                value["comparable_left_only"],
                value["comparable_right_only"],
                _counts(value["supporting_left"]),
                _counts(value["supporting_right"]),
                value["reason"] or value["interpretation"],
            )
            for arm, value in row["same_cohort_agreement"]["arms"].items()
        ),
    )
    return (
        f"<h3>{_text(row['id'])} · {_text(row['left'])} versus {_text(row['right'])}</h3>"
        f"<p>{coverage['declared_units']} declared specimens and {coverage['declared_pairs']} "
        f"possible roster pairs. {coverage['jointly_resolved_units']} specimens and "
        f"{coverage['jointly_resolved_pairs']} pairs jointly resolved.</p>"
        f"<p>{metrics['together_only_left']} pairs together only on the left. "
        f"{metrics['together_only_right']} pairs together only on the right. "
        f"{metrics['unchanged_pairs']} pairs unchanged.</p>"
        f"<p>Relation on jointly resolved specimens · {_text(relation)}. "
        f"Coassignment agreement · {_text(metrics['coassignment_agreement'])}. "
        f"Adjusted Rand index · {_text(metrics['adjusted_rand'])}. "
        f"Degenerate · {_text(metrics['degenerate'])}.</p>"
        + (f"<p>{_text(metrics['reason'])}</p>" if metrics.get("reason") else "")
        + _details("Member overlap", overlap)
        + f"<h4>Agreement on discriminating pairs</h4><p>"
        f"{coverage['possible_discriminating_pairs']} possible discriminating pairs on the "
        f"jointly resolved candidate cohort. "
        f"{_text(row['same_cohort_agreement']['reason'])}</p>{agreement}"
        f"<h4>Focal cohesion and separation</h4>{_focal(row['focal_boundary'])}"
        f"<h4>Archived stability</h4>{_stability(row['stability'])}"
        f"<h4>Confirmatory evidence</h4>"
        f"<p>Fitted comparison · {_text(_status(row['fitted_split_vs_merge']))}</p>"
        f"<p>Independent validation · {_text(_status(row['independent_validation']))}</p>"
    )


def render_comparison(document):
    run = document["run"]
    candidates = "".join(_candidate(row) for row in document["hypotheses"])
    contrasts = "".join(_contrast(row) for row in document["contrasts"])
    arms = _table(
        (
            "Arm",
            "Shared measurements by view",
            "Status",
            "Selected K",
            "Resolved / declared",
        ),
        (
            (
                row["name"],
                ", ".join(row["modalities"]),
                row["status"],
                row["selected_k"],
                f"{row['resolved_count']}/{row['declared_count']}",
            )
            for row in document["archive_arms"]
        ),
    )
    native = "".join(
        _details(
            row["name"],
            _json(
                {
                    "source": row["source"],
                    "stability": row.get("stability"),
                    "p_value_records": row.get("native_p_value_records", {"status": "unavailable"}),
                    "native_p_values": row.get("native_p_values", []),
                }
            ),
        )
        for row in document["archive_arms"]
    )
    limitations = "".join(f"<li>{_text(message)}</li>" for message in document["limitations"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Super Deli hypothesis comparison</title>
<style>
:root {{ font: 16px/1.55 system-ui, sans-serif; background: #f7f9f8; color: #18292e; }}
body {{ max-width: 82rem; margin: auto; padding: 2rem clamp(1rem, 4vw, 4rem); }}
h1, h2, h3, h4 {{ line-height: 1.2; color: #173f40; }}
nav a {{ display: inline-block; color: #176569; margin-right: 1rem; }}
section {{ margin-block: 2.5rem; }}
.table {{ overflow-x: auto; margin-block: .8rem 1.3rem; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th, td {{ text-align: left; vertical-align: top; padding: .5rem .65rem;
          border-bottom: 1px solid #d9e3e0; }}
th {{ background: #e7f0ee; }}
summary {{ cursor: pointer; font-weight: 600; }}
code, td, pre {{ overflow-wrap: anywhere; }}
pre {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<header>
<h1>Super Deli hypothesis comparison</h1>
<p>{_text(run["dataset_id"])} · {_text(run["study_path"])} · Schema {document["schema_version"]}</p>
<nav aria-label="Report sections">
<a href="#candidates">Candidates</a><a href="#contrasts">Contrasts</a>
<a href="#sources">Evidence sources</a><a href="comparison.json">Comparison JSON</a>
</nav>
</header>
<main>
<section id="candidates"><h2>Candidate partitions</h2>{candidates}</section>
<section id="contrasts"><h2>Declared contrasts</h2>{contrasts}</section>
<section id="sources"><h2>Evidence sources</h2>{arms}
<p>Native p-values and conditional memberships retain the scope of their original runs.
Specimen-level fitted assignments and memberships are available in the local JSON.</p>
{native}
{
        _details(
            "Run inference and experimental dataset-level held-out gain",
            _json(document["native_inference"]),
        )
    }
{_details("Verified provenance", _json(document["provenance"]))}
<ul>{limitations}</ul></section>
</main>
</body>
</html>
"""
