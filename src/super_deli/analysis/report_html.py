"""Render a verified report with escaped text and no external dependencies."""

from html import escape


def _text(value):
    return escape("—" if value is None else str(value))


def _table(headers, rows):
    head = "".join(f'<th scope="col">{_text(value)}</th>' for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return (
        '<div class="table"><table>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _status(row):
    return row["status"].replace("_", " ") + (f". {row['reason']}" if row.get("reason") else "")


def _counts(row):
    return f"{row['count'] if row['count'] is not None else '—'}/{row['comparable']}" + (
        f" ({row['possible']} possible)" if row["comparable"] != row["possible"] else ""
    )


def _boundary(boundary):
    if boundary["status"] != "available":
        return f"<p>{_text(_status(boundary))}</p>"
    group_labels = [group["label"] for group in boundary["groups"]]
    arm_rows = []
    for arm, row in boundary["descriptive_agreement"]["arms"].items():
        if "cross_group_separation" in row:
            counts = [row["cross_group_separation"], *row["within_group_cohesion"]]
            arm_rows.append((arm, *map(_counts, counts)))
        else:
            arm_rows.append((arm, _status(row), None, None))
    agreement = _table(
        ("Arm", "Cross-group pairs separated", *(f"{label} pairs kept" for label in group_labels)),
        arm_rows,
    )
    members = _table(
        ("Group", "Members"),
        ((group["label"], ", ".join(group["unit_ids"])) for group in boundary["groups"]),
    )
    stability = boundary["stability"]
    stability_text = f"<p>{_text(_status(stability))}</p>"
    if "requested" in stability:
        stability_text += (
            f"<p>{_text(stability['resampling_unit'])}. "
            f"{stability['requested']} requested, {stability['successful']} successful, "
            f"{stability['failed']} failed.</p>"
        )
    if "joint_exact_group_recurrence" in stability:
        pairs = stability["specimen_pair_separation"]
        stability_text += _table(
            ("Saved projection measure", "Count / comparable draws"),
            (
                *(
                    (f"Exact recurrence of {label}", _counts(row))
                    for label, row in stability["exact_group_recurrence"].items()
                ),
                ("Both exact groups recur", _counts(stability["joint_exact_group_recurrence"])),
                (
                    "Specimen pair separated across all projected draws",
                    _counts(pairs["all_projected_draws"]),
                ),
                (
                    "Specimen pair separated when both were omitted",
                    _counts(pairs["both_specimens_omitted"]),
                ),
            ),
        )
    return (
        f"<p>Originating arm {_text(boundary['originating_arm'])}. "
        f"Focal groups {_text(' and '.join(group_labels))}. "
        f"Focal specimens {_text(' and '.join(boundary['specimen_pair']))}.</p>"
        f"<details><summary>Focal group members</summary>{members}</details>"
        f"<h3>Descriptive agreement on the discovery cohort</h3>{agreement}"
        f"<h3>Saved bootstrap evidence</h3>{stability_text}"
        f"<p>Fitted split versus merge · {_text(_status(boundary['fitted_split_vs_merge']))}</p>"
        f"<p>Independent validation · {_text(_status(boundary['independent_validation']))}</p>"
    )


def _arm_assignment(row, *, comparison_label=False):
    if row["status"] == "unavailable":
        return _status(row)
    label = row["assignment"]
    membership = row["maximum_conditional_membership"]
    if label is None:
        return f"Abstained. {row['reason']}"
    saved_label = f", saved index {row['comparison_label']}" if comparison_label else ""
    return f"{label} ({membership}{saved_label})"


def _alternative(row):
    alt = row.get("strongest_alternative", row)
    if alt["status"] == "available":
        return f"{alt['label']} ({alt['membership']})"
    return _status(alt)


def _specimen_highlights(boundary, partitions, specimens, arms):
    if boundary["status"] != "available":
        return ""
    origin = boundary["originating_arm"]
    lowest = next(
        p["lowest_maximum_membership"]["unit_id"] for p in partitions if p["name"] == origin
    )
    by_id = {row["unit_id"]: row["arms"] for row in specimens}
    reasons = dict.fromkeys(boundary["specimen_pair"], "Focal pair")
    reasons[lowest] = (
        "Focal pair and lowest maximum membership"
        if lowest in reasons
        else "Lowest maximum membership"
    )
    return _table(
        ("Specimen", f"Role in {origin}", *arms, f"Strongest alternative in {origin}"),
        (
            (
                unit,
                reason,
                *(_arm_assignment(by_id[unit][name], comparison_label=True) for name in arms),
                _alternative(by_id[unit][origin]),
            )
            for unit, reason in reasons.items()
        ),
    )


def _specimens(document):
    partitions = document["partitions"]
    arms = [arm["name"] for arm in partitions]
    specimens = document["specimens"]
    highlights = _specimen_highlights(document["focal_boundary"], partitions, specimens, arms)
    assignments = _table(
        ("Specimen", "Source observations", *arms),
        (
            (
                row["unit_id"],
                ", ".join(f"{view} = {obs}" for view, obs in row["source_observation_ids"].items()),
                *(_arm_assignment(row["arms"][name]) for name in arms),
            )
            for row in specimens
        ),
    )
    alternatives = _table(
        ("Specimen", *arms),
        (
            (row["unit_id"], *(_alternative(row["arms"][name]) for name in arms))
            for row in specimens
        ),
    )
    return (
        "<p>Memberships are conditional on the selected model for each arm.</p>"
        f"{highlights}<details><summary>All specimen assignments and sources</summary>"
        f"{assignments}</details><details><summary>Strongest alternatives</summary>{alternatives}"
        "</details>"
    )


def _arm_stability(partitions):
    rows = []
    for partition in partitions:
        stability = partition.get("stability")
        if stability is None:
            rows.append((partition["name"], "Unavailable", None, None, None, None))
            continue
        failed = len(stability["failures"]) if "failures" in stability else stability["failed"]
        status = _status(stability) if stability["requested"] else "Not computed"
        rows.append(
            (
                partition["name"],
                status,
                stability["requested"],
                stability["successful"],
                failed,
                stability.get("resampling_unit", partition["evidence_type"].replace("_", " ")),
            )
        )
    return _table(("Arm", "Status", "Requested", "Successful", "Failed", "Scope"), rows)


def render_html(document: dict) -> str:
    run, cohort, provenance = (document[k] for k in ("run", "cohort", "provenance"))
    views = _table(
        ("View", "Kind", "Measurement level", "Role", "Modeled tabular columns"),
        (
            (
                v["modality"],
                v["kind"],
                v["measurement_level"],
                v["role"],
                ", ".join(v["feature_names"]) or None,
            )
            for v in cohort["views"]
        ),
    )
    partitions = _table(
        ("Arm", "Views", "Status", "Resolved / declared", "Selected K", "At search maximum"),
        (
            (
                p["name"],
                ", ".join(p["modalities"]),
                _status(p) if not p["error"] else p["error"],
                f"{p['resolved_count']}/{p['declared_count']}",
                p["selected_k"],
                p["search_boundary"],
            )
            for p in document["partitions"]
        ),
    )
    comparison = _table(
        ("Left arm", "Right arm", "Status", "Jointly resolved", "Discordant / comparable pairs"),
        (
            (
                r["left"],
                r["right"],
                _status(r),
                r.get("joint_resolved"),
                f"{r['discordant_pairs']}/{r['joint_pairs']}"
                if r.get("discordant_pairs") is not None
                else None,
            )
            for r in document["partition_comparisons"]
        ),
    )
    p_values = _table(
        ("Arm", "Target", "Status", "Value", "Original null hypothesis", "Method"),
        (
            (p["name"], e["target"], _status(e), e["value"], e["null_hypothesis"], e["method"])
            for p in document["partitions"]
            for e in p.get("native_p_values", [])
        ),
    )
    p_value_coverage = _table(
        ("Arm", "Record availability", "Archived p-value records"),
        (
            (p["name"], _status(record), record.get("record_count"))
            for p in document["partitions"]
            for record in (p.get("native_p_value_records") or {"status": "unavailable"},)
        ),
    )
    sources = _table(
        ("Artifact", "Bytes", "SHA-256"),
        (
            (v["study_path"], v["bytes"], v["sha256"])
            for v in provenance["verified_run_artifacts"].values()
        ),
    )
    stability = _arm_stability(document["partitions"])
    boundary_inference = _status(document["inference"]["species_boundary_inference"])
    limitations = "".join(f"<li>{_text(item)}</li>" for item in document["limitations"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Super Deli archived analysis</title>
<style>
:root {{ color-scheme: light; font: 16px/1.5 system-ui, sans-serif; color: #18292e;
        background: #f8faf9; }}
body {{ max-width: 85rem; margin: auto; padding: 2rem clamp(1rem, 4vw, 4rem); }}
h1, h2, h3 {{ line-height: 1.2; color: #173f40; }}
nav a {{ display: inline-block; margin: 0 .9rem .25rem 0; color: #176569; }}
section {{ margin-block: 2.6rem; }}
.table {{ overflow-x: auto; margin-block: .8rem 1.3rem; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; }}
th, td {{ text-align: left; vertical-align: top; padding: .55rem .65rem;
          border-bottom: 1px solid #d9e3e0; }}
th {{ background: #e7f0ee; }}
code, td {{ overflow-wrap: anywhere; }}
summary {{ cursor: pointer; font-weight: 600; }}
</style>
</head>
<body>
<header>
<h1>Super Deli archived analysis</h1>
<p><strong>{_text(run["dataset_id"])}</strong> · {_text(run["workflow"])}
 · {_text(run["status"])} · {cohort["declared_count"]} specimens</p>
<nav aria-label="Report sections">
<a href="#cohort">Cohort</a><a href="#partitions">Partitions</a>
<a href="#boundary">Focal boundary</a><a href="#specimens">Specimens</a>
<a href="#evidence">Evidence and sources</a><a href="report.json">Report JSON</a>
</nav>
</header>
<main>
<section id="cohort"><h2>Comparison cohort</h2>
<p>Independence · {_text(cohort["independence"].replace("_", " "))}.
Excluded specimens · {len(cohort["exclusions"])}.</p>{views}</section>
<section id="partitions"><h2>Saved partitions</h2>{partitions}
<details><summary>Same-cohort partition comparisons</summary>{comparison}</details></section>
<section id="boundary"><h2>Focal boundary</h2>{_boundary(document["focal_boundary"])}</section>
<section id="specimens"><h2>Specimen assignments</h2>{_specimens(document)}</section>
<section id="evidence"><h2>Evidence and sources</h2>
<p>Species-boundary inference · {_text(boundary_inference)}</p>
<p>Experimental held-out gain · {_text(_status(document["inference"]["held_out_gain"]))}</p>
<details><summary>Saved stability by arm</summary>{stability}</details>
<details><summary>Native p-value records</summary>{p_value_coverage}{p_values}</details>
<ul>{limitations}</ul>
<p>Run location <code>{_text(run["study_path"])}</code></p>
<p>Dataset configuration SHA-256 <code>{_text(run["dataset_configuration_sha256"])}</code></p>
<p>Analysis configuration SHA-256 <code>{_text(run["analysis_configuration_sha256"])}</code></p>
<p>Study manifest SHA-256 <code>{_text(provenance["study_manifest_sha256"])}</code></p>
<details><summary>Verified source artifacts</summary>{sources}</details></section>
</main>
</body>
</html>
"""
