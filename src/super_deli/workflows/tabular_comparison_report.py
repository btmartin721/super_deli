"""Offline HTML with escaped source text and explicitly different support measures."""

from __future__ import annotations

import html
import json
from pathlib import Path


def _escape(value):
    return html.escape(str(value), quote=True)


def _table(rows, columns):
    head = "".join(f"<th>{_escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(r.get(c))}</td>" for c in columns) + "</tr>" for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _plot(scores, labels, resolved):
    x = [float(row[0]) for row in scores]
    y = [float(row[1]) if len(row) > 1 else 0.0 for row in scores]

    def scale(v, lo, hi):
        return 30 + 340 * (v - lo) / (hi - lo or 1)

    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    points = []
    for a, b, label, ok in zip(x, y, labels, resolved, strict=True):
        color = palette[label % len(palette)] if label >= 0 and ok else "#888888"
        points.append(
            f'<circle cx="{scale(a, min(x), max(x)):.3f}" '
            f'cy="{400 - scale(b, min(y), max(y)):.3f}" r="3" fill="{color}"/>'
        )
    second = "PC2" if len(scores[0]) > 1 else "Constant display axis (one PC)"
    return (
        '<svg viewBox="0 0 400 420" role="img" aria-label="PCA assignments">'
        '<rect width="400" height="420" fill="white"/>'
        + "".join(points)
        + f'<text x="170" y="415">PC1; {_escape(second)}</text></svg>'
    )


def write_comparison_report(summary, points, output):
    """Write one self-contained report; native artifacts remain authoritative."""
    rows = summary["models"]
    sections = []
    for row in rows:
        name = row["name"]
        plot = _plot(**points[name]) if name in points else "<p>Point fit unavailable.</p>"
        summary_link = row.get("native_summary", f"arms/{name}/summary.json")
        assignment_link = row.get("native_assignments", f"arms/{name}/assignments.csv")
        links = (
            (
                f'<p><a href="{_escape(summary_link)}">Native summary</a>; '
                f'<a href="{_escape(assignment_link)}">Native assignments</a></p>'
            )
            if name in points
            else f"<p>{_escape(row.get('error'))}</p>"
        )
        if row.get("ordinary_summary"):
            links += (
                f'<p><a href="{_escape(row["ordinary_summary"])}">Retained ordinary fit</a></p>'
            )
        details = ""
        for key, title in (
            ("selection", "Native screening decisions"),
            ("preprocessing", "Retained preprocessing evidence"),
        ):
            if key in row:
                details += f"<details><summary>{title}</summary><pre>"
                details += _escape(json.dumps(row[key], indent=2)) + "</pre></details>"
        sections.append(
            f"<section><h2>{_escape(name)}</h2>{plot}"
            + links
            + "<h3>Prespecified settings</h3><pre>"
            + _escape(json.dumps(row["configuration"], indent=2))
            + "</pre><h3>Family-specific resampling</h3><pre>"
            + _escape(json.dumps(row.get("stability"), indent=2))
            + "</pre>"
            + details
            + "</section>"
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Super Deli tabular comparison</title><style>"
        "body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;color:#18222c}"
        "table{border-collapse:collapse;display:block;overflow-x:auto}td,th{padding:.5rem;"
        "border:1px solid #b8c2cc;text-align:left}svg{max-width:420px;width:100%}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;padding:1rem}"
        "</style><h1>Super Deli tabular comparison</h1>"
        f"<p>Dataset: {_escape(summary['dataset_id'])}. Status: {_escape(summary['status'])}.</p>"
        "<p>These partitions provide candidate groups for biological assessment. "
        "No winning model, consensus species "
        "partition or new p-value is inferred. Gray points are noise or unresolved. "
        "Colors are local to each model, not matched labels.</p>"
        "<h2>Model summaries</h2>"
        + _table(
            rows,
            (
                "name",
                "family",
                "status",
                "selected_k",
                "resolved_count",
                "noise_count",
                "unresolved_count",
                "stability_status",
                *summary.get("report_columns", []),
            ),
        )
        + "<h2>Partition agreement</h2>"
        "<p>Agreement uses only the intersection of resolved, non-noise units. Read "
        "coverage and degeneracy alongside agreement. It does not measure biological accuracy.</p>"
        + _table(
            summary["comparisons"],
            (
                "left",
                "right",
                "status",
                "joint_resolved",
                "joint_coverage",
                "joint_pairs",
                "adjusted_rand",
                "degenerate",
            ),
        )
        + "<h2>Evidence boundaries</h2><p>"
        + _escape(" ".join(summary["limitations"]))
        + "</p>"
        + (
            "<h2>Explicit target sampling</h2>"
            + _table(summary["groups"], ("target_id", "morphology_count", "ecology_count"))
            if "groups" in summary
            else ""
        )
        + "".join(sections)
        + "</html>\n"
    )
    with Path(output).open("x", encoding="utf-8") as handle:
        handle.write(document)
