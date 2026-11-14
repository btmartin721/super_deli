"""Offline matched-model report with representation-specific axes and evidence."""

import json
from pathlib import Path

from super_deli.workflows.tabular_comparison_report import _escape, _table

HEADER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Super Deli matched discovery comparison</title>
<style>
body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;color:#18222c}
table{border-collapse:collapse;display:block;overflow-x:auto}
td,th{padding:.5rem;border:1px solid #b8c2cc;text-align:left}
svg{max-width:420px;width:100%}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;padding:1rem}
</style></head><body><h1>Matched morphology–ecology discovery comparison</h1>
"""


def _plot(point, family):
    scores, labels = point["scores"], point["labels"]
    x = [r[0] for r in scores]
    y = [r[1] if len(r) > 1 else 0 for r in scores]

    def scale(v, lo, hi):
        return 35 + 330 * (v - lo) / (hi - lo or 1)

    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
    dots = "".join(
        f'<circle cx="{scale(a, xmin, xmax):.3f}" cy="{390 - scale(b, ymin, ymax):.3f}" '
        f'r="3" fill="{palette[int(c) % len(palette)]}"/>'
        for a, b, c in zip(x, y, labels, strict=True)
    )
    axes = (
        ("Joint PC1 / PC2" if len(scores[0]) > 1 else "Joint PC1 / constant display axis")
        if family == "pca_gmm"
        else "Morphology latent 1 / 2"
    )
    legend = " ".join(
        f'<span style="color:{palette[int(c) % len(palette)]}">Native cluster {int(c)}</span>'
        for c in sorted(set(labels))
    )
    return (
        '<svg viewBox="0 0 400 420" role="img" aria-label="' + _escape(axes) + '">'
        '<rect width="400" height="420" fill="white"/>'
        + dots
        + '<text x="15" y="410">'
        + _escape(axes)
        + "</text></svg><p>Display only: clustering uses all "
        + str(len(scores[0]))
        + " coordinates. Colors identify components within this arm only.</p><p>"
        + legend
        + "</p>"
    )


def write_matched_report(summary, points, output):
    sections = []
    for row in summary["models"]:
        name = row["name"]
        if name not in points:
            body = "<p>Point fit unavailable: " + _escape(row["error"]) + "</p>"
        else:
            body = _plot(points[name], row["family"])
            body += (
                f'<p><a href="arms/{name}/summary.json">Native summary</a>; '
                f'<a href="arms/{name}/assignments.csv">Native assignments</a></p>'
            )
        for key, title in (
            ("configuration", "Prespecified native configuration"),
            ("stability", "Native resampling evidence"),
            ("preprocessing", "Native preprocessing"),
            ("candidates", "Within-arm component search"),
        ):
            if key in row:
                body += (
                    "<details><summary>"
                    + title
                    + "</summary><pre>"
                    + _escape(json.dumps(row[key], indent=2))
                    + "</pre></details>"
                )
        sections.append("<section><h2>" + _escape(name) + "</h2>" + body + "</section>")
    document = (
        HEADER
        + "<p>Dataset: "
        + _escape(summary["dataset_id"])
        + ". Status: "
        + _escape(summary["status"])
        + ".</p>"
        + (
            "<p>Explicit specimen IDs align complete individual measurements. "
            "Independence and individual ecology are declared assumptions, not "
            "verified facts.</p>"
        )
        + (
            "<p>These are candidate partitions. No winner, consensus species or "
            "biological accuracy is inferred. Absolute BIC values are not compared "
            "across fitted spaces.</p>"
        )
        + "<h2>Models</h2>"
        + _table(
            summary["models"],
            (
                "name",
                "family",
                "status",
                "selected_k",
                "search_boundary",
                "resolved_count",
                "stability_status",
            ),
        )
        + (
            "<h2>Partition agreement</h2><p>Label-invariant agreement is between "
            "fitted partitions, not a test of biological "
            "truth.</p>"
        )
        + _table(
            summary["comparisons"],
            (
                "left",
                "right",
                "status",
                "joint_resolved",
                "joint_coverage",
                "adjusted_rand",
                "degenerate",
            ),
        )
        + "<h2>Evidence meanings</h2><p>"
        + _escape(" ".join(summary["limitations"]))
        + "</p>"
        + (
            "<p>Conditional memberships describe the selected fitted mixture. PCA "
            "bootstrap and neural 80% subsampling have different inclusion and OOB "
            "denominators; read their native evidence "
            "separately.</p>"
        )
        + (
            "<p>Gaussian-departure significance: not computed by this analysis. "
            "Existing standalone calibration commands remain separate; a departure "
            "p-value would not certify K or species "
            "boundaries.</p>"
        )
        + "".join(sections)
        + "</body></html>\n"
    )
    with Path(output).open("x", encoding="utf-8") as f:
        f.write(document)
