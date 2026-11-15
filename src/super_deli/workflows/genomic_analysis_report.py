"""Self-contained local comparison report; no remote scripts or assets."""

import json
from pathlib import Path

from super_deli.workflows.matched_analysis_report import _plot
from super_deli.workflows.tabular_comparison_report import _escape, _table


def write_genomic_report(summary, points, output):
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Super Deli genomic–phenotypic comparison</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}",
        "table{border-collapse:collapse;display:block;overflow-x:auto}th,t"
        "d{border:1px solid #aaa;padding:.5rem}",
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{max-width:420px}</style>",
        "<body><h1>Matched genomic–phenotypic discovery</h1>",
        "<p>Dataset: "
        + _escape(summary["dataset_id"])
        + ". Status: "
        + _escape(summary["status"])
        + ".</p>",
        "<p>Candidate partitions; no model winner, consensus or species claim.</p>",
        _table(summary["models"], ("name", "family", "status", "selected_k", "search_boundary")),
        "<h2>Label-invariant agreement</h2><p>Agreement between models is "
        "not biological accuracy.</p>",
        _table(
            summary["comparisons"],
            ("left", "right", "status", "joint_coverage", "adjusted_rand", "degenerate"),
        ),
        "<h2>Evidence meanings and limitations</h2>",
    ]
    parts.extend("<p>" + _escape(s) + "</p>" for s in summary["limitations"])
    for row in summary["models"]:
        name = row["name"]
        parts.append("<h2>" + _escape(name) + "</h2>")
        if name in points:
            parts.extend(
                [
                    _plot(points[name], "pca_gmm").replace("Joint PC", "Native PC"),
                    "<p>Axes are PC1/PC2 of this native arm. All fitted PCs enter clustering.</p>",
                    '<p><a href="arms/'
                    + name
                    + '/summary.json">Native summary and stability evidence</a></p>',
                ]
            )
        else:
            parts.append("<p>Unavailable: " + _escape(row["error"]) + "</p>")
        parts.append(
            "<details><summary>Native configuration and evidence</summary><pre>"
            + _escape(json.dumps(row, indent=2))
            + "</pre></details>"
        )
    parts.append("</body></html>")
    with Path(output).open("x", encoding="utf-8") as handle:
        handle.write("\n".join(parts) + "\n")
