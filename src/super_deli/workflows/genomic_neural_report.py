"""Small offline report; no remote assets or biological truth in displays."""

import json
from html import escape


def write_report(summary, points, path):
    sections = []
    colors = ("#2166ac", "#b2182b", "#238b45", "#984ea3", "#c96b00", "#555555")
    for row in summary["models"]:
        name = escape(row["name"])
        body = (
            f"<h2>{name}</h2><p>Status: {escape(row['status'])}; "
            f"selected K: {row['selected_k']}</p>"
        )
        if row["name"] in points:
            a = points[row["name"]].arrays
            xy = a["scores"]
            span = xy.max(0) - xy.min(0)
            scaled = (xy - xy.min(0)) / (span + 1e-15)
            dots = []
            for i, (x, y) in enumerate(scaled):
                title = escape(summary["unit_ids"][i])
                color = colors[int(a["labels"][i]) % len(colors)]
                dots.append(
                    f'<circle cx="{35 + 500 * x:.3f}" cy="{325 - 290 * y:.3f}" '
                    f'r="3" fill="{color}"><title>{title}</title></circle>'
                )
            body += (
                '<svg viewBox="0 0 570 360" role="img" '
                'aria-label="Fitted two-coordinate candidate clusters">' + "".join(dots) + "</svg>"
            )
        else:
            body += "<p>" + escape(row["error"] or "Fit unavailable") + "</p>"
        body += (
            "<details><summary>Native evidence and specimen stability</summary><pre>"
            + escape(json.dumps(row, indent=2))
            + "</pre></details>"
        )
        sections.append(body)
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Genomic model comparison</title>"
        "<style>body{font:16px system-ui;max-width:1000px;margin:2em auto;padding:1em}"
        "svg{max-width:640px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>"
        "<h1>Genomic candidate clusters</h1><p>Colors are local to each arm. "
        "No winning model or species partition is selected.</p>"
        + "".join(sections)
        + "<h2>Interpretation</h2><ul>"
        + "".join("<li>" + escape(s) + "</li>" for s in summary["limitations"])
        + "</ul></html>"
    )
