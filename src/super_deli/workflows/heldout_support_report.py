"""Offline reporting of prediction, sampling feasibility and unavailable evidence."""

from html import escape


def display(value):
    return "unavailable" if value is None else format(value, ".5g")


def write_report(summary, groups, path):
    s = summary
    rows = "".join(
        f"<tr><td>{escape(g['point_cluster'])}</td><td>{g['point_individuals']}</td>"
        f"<td>{g['minimum_included']}&ndash;{g['maximum_included']}</td>"
        f"<td>{g['below_declared_minimum']}/{s['requested_draws']}</td>"
        f"<td>{g['covered_individuals']}/{g['point_individuals']}</td>"
        f"<td>{display(g['covered_individual_mean'])}</td></tr>"
        for g in groups
    )
    eligibility = "".join(
        f"<tr><td>{r['k']}</td><td>{r['eligible']}</td><td>{r['small_component']}</td>"
        f"<td>{r['insufficient_samples']}</td><td>{r['unavailable']}</td><td>{r['selected']}</td></tr>"
        for r in s["candidate_eligibility"]
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Held-out predictive support</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
table{{border-collapse:collapse}}th,td{{border:1px solid #aaa;padding:.5rem;text-align:left}}
.note{{background:#eef3f7;padding:1rem}}</style>
<h1>Omitted-individual predictive support</h1>
<p>The original point fit has {s["source_point_k"]} candidate clusters. Its partition,
assignment threshold, existing stability and p-values are preserved.</p>
<p>Coverage: {s["covered_individuals"]}/{s["individuals"]} individuals ({s["coverage"]:.1%}).
Successful omitted comparisons: {s["successful_comparisons"]}/{s["requested_comparisons"]}.
Unavailable requested comparisons: {s["unavailable_requested_comparisons"]}.</p>
<p>Covered-individual mean gain: <strong>{display(s["covered_individual_mean"])}</strong>.
Complete full-cohort mean: {display(s["full_cohort_mean"])}.
Coverage/completeness qualification: {s["qualified"]}; direction: {s["direction"]}.</p>
<p class="note">Gain compares the training-selected model with its native one-group
score model on individuals omitted from training. Each individual's omitted gains
are averaged first, then covered individuals receive equal weight. Repeated
omissions are not independent observations. The 95% coverage rule is a reporting
policy, not a confidence level or significance threshold. Unavailable or incomplete
evidence is never a one-group conclusion. Unbounded missing gains have no finite
failed-outcome range; available-case values remain explicitly partial.</p>
<h2>Eligibility in the unchanged subsample searches</h2>
<table><tr><th>K</th><th>Eligible</th><th>Fitted cluster too small</th>
<th>Insufficient training support</th><th>Native search unavailable</th>
<th>Selected by native BIC</th></tr>{eligibility}</table>
<h2>Representation of original point clusters</h2>
<table><tr><th>Point cluster</th><th>Individuals</th><th>Included range</th>
<th>Below native minimum {s["native_minimum_distinct_individuals"]}</th>
<th>Omitted coverage</th><th>Covered mean gain</th></tr>{rows}</table>
<p>Cluster-specific values are descriptive annotations from same-sample point
labels. Insufficient representation can prevent rare-group recovery; positive
global prediction does not validate a particular cluster count or every extra
group. No observed advantage does not establish one species. No automatic
filter, new partition, species probability or p-value is introduced.</p>
<p>Saved ingested calls/loci and extracted landmarks are fixed. Native draw
transformations were fitted on training individuals only. Scores compare models
within the same frozen projected space, not raw-data likelihood. Source genotype,
measurement and missing-value uncertainty are not propagated.</p></html>
""",
        encoding="utf-8",
    )
