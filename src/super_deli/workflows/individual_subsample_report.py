"""Offline report separating point structure and sampling sensitivity."""

from html import escape


def write_report(summary, support, path):
    s = summary
    rows = []
    for r in support:
        similarity = (
            "unavailable"
            if r["best_jaccard_mean"] is None
            else format(r["best_jaccard_mean"], ".3f")
        )
        low = (
            "unavailable"
            if r["best_jaccard_lower"] is None
            else format(r["best_jaccard_lower"], ".3f")
        )
        high = (
            "unavailable"
            if r["best_jaccard_upper"] is None
            else format(r["best_jaccard_upper"], ".3f")
        )
        rows.append(
            f"<tr><td>{escape(r['point_cluster'])}</td><td>{r['point_individuals']}</td>"
            f"<td>{r['minimum_included']}–{r['maximum_included']}</td>"
            f"<td>{r['below_declared_minimum']}/{s['requested']}</td>"
            f"<td>{similarity}</td><td>{low}–{high}</td></tr>"
        )
    frequencies = "".join(
        f"<tr><td>{r['k']}</td><td>{r['count']}</td><td>{r['lower']:.3f}–{r['upper']:.3f}</td></tr>"
        for r in s["selected_k_frequencies"]
    )
    eligibility = "".join(
        f"<tr><td>{r['k']}</td><td>{r['eligible']}</td><td>{r['small_component']}</td>"
        f"<td>{r['insufficient_samples']}</td><td>{r['unavailable']}</td></tr>"
        for r in s["candidate_eligibility"]
    )
    table_rows = "".join(rows)
    body = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Whole-individual subsampling</title>
<style>
body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
table{{border-collapse:collapse}}th,td{{border:1px solid #aaa;padding:.5rem;text-align:left}}
.note{{background:#eef3f7;padding:1rem}}
</style>
<h1>Whole-individual subsampling sensitivity</h1>
<p>The original point analysis selected {s["source_point_k"]} clusters from
{s["individuals"]} individuals. Its assignments and existing bootstrap are preserved.</p>
<p>Each refit includes {s["subsample_individuals"]} distinct individuals (80%),
with their data paired across views. The complete analysis pipeline is refitted.
{s["successful"]}/{s["requested"]} refits succeeded; {s["failed"]} failed.</p>
<p class="note">These results describe sensitivity to leaving out individuals.
Frequencies and membership values are not species probabilities or significance tests.
Failed-result ranges are identification bounds, not confidence intervals.
Existing bootstrap evidence and subsampling describe different perturbations.</p>
<p>Support and coassignment concern candidate partitions. The source run retains
its original conditional-membership threshold and resolved/unresolved assignments.</p>
<h2>Candidate cluster counts</h2>
<table><tr><th>Clusters</th><th>Successful refits</th>
<th>Attempt-based frequency bounds</th></tr>{frequencies}</table>
<h2>Candidate eligibility across refits</h2>
<table><tr><th>Clusters</th><th>Eligible</th><th>Fitted cluster too small</th>
<th>Too few distinct training individuals</th><th>Unavailable after fit failure</th>
</tr>{eligibility}</table>
<h2>Representation of point clusters</h2>
<p>The minimum remains {s["native_minimum_distinct_individuals"]} distinct individuals
per fitted cluster. Point-cluster representation is a support diagnostic;
it does not determine the fitted partition.</p>
<table><tr><th>Point cluster</th><th>Original individuals</th><th>Included range</th>
<th>Below minimum</th><th>Mean included-set similarity</th><th>Failed-outcome range</th>
</tr>{table_rows}</table>
<h2>Included and omitted individuals</h2>
<p>Included-pair coverage: {s["included_pair_coverage"]:.1%}.
Omitted-pair coverage: {s["omitted_pair_coverage"]:.1%}.
Included coassignment uses pairs present in a refit. Omitted coassignment uses
predictions only when both individuals were left out; pairs with no such evaluation
remain unavailable. These denominators are stored separately.</p>
<p>Whole clusters can disappear when their represented sample is too small.
Stability of an observed partition does not establish species boundaries.
Detailed support, membership, seed and failure records accompany this report.</p></html>"""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(body)
