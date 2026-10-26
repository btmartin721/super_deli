"""Reproduce compact summaries from the immutable, locally audited 300-dataset study.

Truth is evaluated after fitting; no estimator, configuration or p-value changes.
Approximate secondary intervals are descriptive and unadjusted.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t

parser = argparse.ArgumentParser(description="Export small paired density-study summaries")
parser.add_argument("--artifact-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
r = args.artifact_root.resolve()
study = r / "study"
audit = json.loads((r / "audit/summary.json").read_text())
assert audit["status"] == "passed"
rows = [json.loads(p.read_text()) for p in sorted(study.glob("*/summary.json"))]
assert len(rows) == 300
out = args.output_dir.resolve()
out.mkdir(exist_ok=False)


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stats(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    n = len(a)
    if not n:
        return dict(
            datasets=0, mean=None, se=None, approximate_95_lower=None, approximate_95_upper=None
        )
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    d = float(t.ppf(0.975, n - 1) * se) if se is not None else None
    return dict(
        datasets=n,
        mean=mean,
        se=se,
        approximate_95_lower=mean - d if d is not None else None,
        approximate_95_upper=mean + d if d is not None else None,
    )


paired = []
stability = []
independent_rates = []
for case in dict.fromkeys(x["case"] for x in rows):
    for n in (64, 256):
        cell = [x for x in rows if x["case"] == case and x["n"] == n]
        for field in ["correct_k", "ari", "assigned_fraction"]:
            diffs = [
                int(x["families"]["hdbscan"]["metrics"][field])
                - int(x["families"]["gmm"]["metrics"][field])
                if field == "correct_k"
                else x["families"]["hdbscan"]["metrics"][field]
                - x["families"]["gmm"]["metrics"][field]
                for x in cell
                if all(
                    x["families"][f]["status"] == "complete"
                    and x["families"][f]["metrics"].get(field) is not None
                    for f in ("hdbscan", "gmm")
                )
            ]
            paired.append(
                dict(case=case, n=n, metric=field, direction="HDBSCAN minus GMM", **stats(diffs))
            )
        for family in ("hdbscan", "gmm"):
            good = [x for x in cell if x["families"][family]["status"] == "complete"]
            for field in [
                "ari",
                "assigned_fraction",
                "core_assigned_fraction",
                "assigned_ari",
                "background_recall",
                "noise_precision",
            ]:
                independent_rates.append(
                    dict(
                        case=case,
                        n=n,
                        family=family,
                        metric=field,
                        **stats([x["families"][family]["metrics"].get(field) for x in good]),
                    )
                )
            datasetstats = []
            for row in cell:
                p = study / row["key"]
                records = json.loads((p / f"{family}.json").read_text())
                if not records["draws"]:
                    continue
                with np.load(p / f"{family}.npz", allow_pickle=False) as a:
                    labels = a["point_labels"]
                    pco = a["coassignment"]
                    up = np.triu(np.ones((n, n), dtype=bool), 1)
                    pos = (labels[:, None] >= 0) & (labels[None, :] >= 0)
                    within = up & pos & (labels[:, None] == labels[None, :])
                    between = up & pos & (labels[:, None] != labels[None, :])
                    completed = [x for x in records["draws"] if x["status"] == "complete"]
                    k = len(set(labels) - {-1})
                    stable = (
                        sum(x["selected_k"] == k for x in completed) / len(completed)
                        if completed
                        else None
                    )

                    def conditional(mask, coassignment=pco):
                        vals = coassignment[mask]
                        return float(np.nanmean(vals)) if np.isfinite(vals).any() else None

                    datasetstats.append(
                        dict(
                            k_match=stable,
                            within=conditional(within),
                            between=conditional(between),
                            unit_coverage=float(np.mean(a["unit_successful"] > 0)),
                            pair_coverage=float(np.mean(a["pair_successful"][up] > 0)),
                            noise_frequency=float(np.nanmean(a["noise_frequency"])),
                        )
                    )
            if datasetstats:
                stability.append(
                    dict(
                        case=case,
                        n=n,
                        family=family,
                        independent_datasets=len(datasetstats),
                        **{
                            field: stats([x[field] for x in datasetstats])
                            for field in datasetstats[0]
                        },
                    )
                )
# Independent exact-interval crosscheck uses a different scipy API.
for rate in audit["primary_rates"]:
    assert rate["failed"] == 0
    ci = binomtest(rate["correct_k"], rate["datasets"]).proportion_ci(
        confidence_level=1 - 0.05 / 20, method="exact"
    )
    assert math.isclose(ci.low, rate["simultaneous_95_lower"], abs_tol=1e-10)
    assert math.isclose(ci.high, rate["simultaneous_95_upper"], abs_tol=1e-10)
write("primary-rates.json", audit["primary_rates"])
write("descriptive-metrics.json", audit["descriptive_metrics"])
write("paired-differences.json", paired)
write("metric-uncertainty.json", independent_rates)
write("stability.json", stability)
fields = list(audit["primary_rates"][0])
with (out / "primary-rates.csv").open("x", newline="") as f:
    c = csv.DictWriter(f, fieldnames=fields)
    c.writeheader()
    c.writerows(audit["primary_rates"])
counts = json.loads((study / "summary.json").read_text())
pre = json.loads((r / "preflight/summary.json").read_text())
gate = json.loads((r / "storage-gate.json").read_text())
small = dict(
    schema_version=1,
    version="0.48.0.dev0",
    main=counts,
    preflight=pre,
    independent_numerical_audit={
        "status": audit["status"],
        "numerically_audited_fits": audit["numerically_audited_fits"],
    },
    storage={
        "projected_bytes_with_allowance": gate["projected_bytes_with_25pct_allowance"],
        "cap_bytes": 1073741824,
    },
    simultaneous_primary_rates=20,
    uncertainty=(
        "Primary rates: simultaneous exact 95%; secondary means and paired differences: "
        "descriptive, approximate t intervals over independent datasets, unadjusted. "
        "Five stability datasets/cell, not 125 independent draws."
    ),
    limitations=[
        "Fixed two-dimensional synthetic laws and frozen settings",
        "No empirical fitting, author contact, species-significance or new p-values",
        "Noise/unassigned coverage must accompany group counts and label agreement",
        "Underlying HDBSCAN algorithm not independently reimplemented",
    ],
)
write("summary.json", small)
write(
    "checksums.json",
    {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file()
    },
)
print(json.dumps(dict(main=counts, preflight=pre, public_files=len(list(out.iterdir())))))
