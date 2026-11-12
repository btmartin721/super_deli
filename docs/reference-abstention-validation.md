# Reference-abstention validation protocol

The bounded, simulation-only design freezes eighty fresh training datasets:
ten per cell across weak linear groups, standard linear groups, combined
allometric/saturating measurements and confounded body size, at N=64/256.
All use the preceding four morphology/eight ecology loading/noise laws. The
illustrative generating groups are not empirical species truth.

Every dataset receives four unchanged supervised fits: logistic regression and
the residual classifier at seeds 0,101,202. Logistic uses C=1, lbfgs, 2,000 maximum
iterations and tolerance 1e-9. The residual architecture and 500-update CPU float64
Adam schedule are unchanged. Seed zero is primary; the other seeds remain
sensitivity measurements, not independent datasets or candidates for selection.

Separate cohorts supply 160 labelled known-class calibration specimens (forty
per class), 2,048 known-class evaluation specimens, and 512 each from nearby,
distant and observationally overlapping unfamiliar groups. Confounded-size cases
also receive 2,048 known-class specimens with the size/class association removed.
All roles have separate identities and innovations. Unknown labels enter only
evaluation. Training alone fits transformations and raw support; calibration
alone supplies thresholds. No evaluation outcomes select a policy or threshold.

Known training axes have four corner means +/-a, where a=1.5 for weak/confounded
laws and 3 otherwise. Nearby unfamiliar means are (+a,0), distant means (+6,+6).
The overlapping unfamiliar control exactly duplicates known class 3's conditional
measurement law, with an evaluation-only unfamiliar identity. Allometry and
saturation use the preceding frozen maps. The confounded near/far/overlap groups
retain positive-axis size association, with the overlap control matching class 3.

Compare unscreened predictions and both fixed support policies without retraining
per policy. Main training budget is 320 classifier fits: 240 residual and eighty
logistic, plus eight read-only archived prediction replays. The independently
audited preflight has one fresh dataset per law/N cell, or 32 classifier fits.
Combined: 352 fresh fits, 264 neural trainings and 132,000 audited optimizer updates;
no new GMM fits, significance draws or stability resamples. Freeze source/runtime,
all seed paths and settings before scientific generation. Retain failures without
replacement. Engineering fixtures and replays never enter performance estimates.

Report coverage, accepted accuracy and balanced accuracy, incorrect accepted
fractions of all known specimens, compatible-set inclusion/size, every withheld
reason, unfamiliar accepted fractions, classwise outcomes and seed sensitivity.
Empty accepted denominators remain undefined with coverage zero. Keep changed
conditions and the indistinguishable control separate from exchangeable known-class
inclusion claims. Seeds, policies and evaluation rows are repeated measurements.

The prespecified practical joint target requires >=80% known IID acceptance,
>=90% accuracy among accepted known specimens, and <=10% acceptance for both nearby
and distant unfamiliar groups. These targets are benchmark desiderata, not
guarantees. Withholding everything fails. Thirty-two primary dataset-level target
rates (eight cells, logistic/residual seed zero, two support policies) receive
exact marginal and separate Bonferroni simultaneous intervals, requested
denominators and unavailable bounds. Sixteen paired policy comparisons retain
discordant counts and descriptive differences without superiority tests. Ten
datasets/cell support descriptive conclusions. Promote no policy or default.

Detailed data, labels, scores, predictions and weights stay local under two GiB,
with eight GiB free, at most two single-thread workers, a twenty-minute serial
preflight cap and ninety-minute projected serial main/audit cap. The measured
gate includes audit/controller work, staging twice, a 25% allowance and 128 MiB
reserve, without parallel speed credit. Preserve guard failures; amendments are
required before changing frozen guards/settings. Only code, invented examples,
documentation, figures and small aggregates are public. No empirical analysis
or author contact is part of this checkpoint.

Run the explicitly invented installation example with:

```bash
python examples/reference-abstention/smoke.py --output-dir invented-abstention
```

The separately source-bound research lifecycle is `benchmark-reference-abstention`
with `bind`, `freeze`, `run`, `audit`, `gate` and `report` actions. It requires the
completed preceding archive and its verified model/runtime provenance. It cannot
silently treat an unrelated or invented archive as completed research evidence.
