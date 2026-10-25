# Empirical benchmark foundation

This checkpoint supplies fit-free curation, audited input manifests and a frozen
future evaluation protocol. It performs no clustering, PCA, stability resampling
or significance calibration. Existing estimators, assumptions, defaults and
historical p-values remain unchanged. See the [curation report](empirical-benchmark-curation-results.md).

## Selected source and analytical roles

The initial development case uses four continuous body measurements and two blood
isotope measurements from Pygoscelis captures. Isotopes are individual foraging-
ecology proxies, rather than temperature or spatial climate layers. Each source
row links its own two views. The key is the exact pair `studyName`/`Individual ID`;
there is no fuzzy matching, inferred specimen correspondence or label-derived key.

The source is the authors' public, CC0-licensed mirror at commit
`8957207b78d6ccd1b4654a9dd9c9041b657478ab`. The [profile](../examples/empirical-foundation/profile.json)
pins its URL, exact byte count, SHA-256 and independently verified Git blob SHA-1.
The full feature/identity profile is also bound to the separate
[protocol](../examples/empirical-foundation/protocol.json) by a canonical hash.
Direct EDI access returned HTTP 403. The mirror recipe cites older data-package
versions than its README; their byte equivalence was not verified. The published
[catalog](../examples/empirical-foundation/catalog.json) retains both citation sets.
A mirror is not represented as a directly verified upstream download.

Numeric tokens are preserved exactly. Eligibility requires all six prespecified
features; labels, sex, island, year and comments never select the complete cohort.
Nonfinite or malformed nonmissing measurements, nonpositive physical dimensions,
missing/duplicate keys and undeclared columns fail explicitly. Isotope values can
be negative. No imputation, outlier removal, normalization, unit conversion,
sex/allometry adjustment or aggregation occurs.

## Output boundary

The command creates a new directory containing:

- `analysis/dataset.json`, `morphology.csv` and `ecology.csv`: anonymous capture IDs,
  explicit same-record observation links and continuous features only.
- `evaluation/references.csv`: published reference assignments joined by anonymous
  IDs. These are development hypotheses, with independence unestablished.
- `evaluation/metadata.csv`: nonanalytical covariates, kept separate from features.
- `audit/`: source index, every inclusion/exclusion reason, full profile/protocol,
  and the existing tabular reader's typed ingestion/provenance artifacts.
- `summary.json` and `checksums.json`: completeness, integrity and applicability.

All full empirical inputs/outputs and source keys remain local. GitHub contains
source locks, the frozen plan, compact counts and entirely invented engineering
fixtures. Free-text source comments are not exported into the analytical or
metadata tables. Existing output directories are never overwritten. Failures
after directory creation leave `failure.json`; failed integrity checks do not
create a misleading completed output. The final summary is renamed into place only
after its checksum manifest is written.

## Curation is not statistical readiness

The 344 rows are capture records, not certified unrelated biological individuals.
The source publication sampled mated adults, and individual codes recur across
seasons. No verified mapping to independent nests or persistent biological IDs
is supplied here. The curator preserves this uncertainty rather than inferring
such links from code patterns.

Islands contain multiple reference species, so averaging to islands would mix
taxa. Assigning a separate artificial locality to every capture would also
misrepresent the sampling design. The current grouped predictive-screen and
scaled-fold stability workflows are therefore inapplicable to these manifests.

Existing clustering configurations require an explicit independence assertion.
This checkpoint makes none, emits no runnable clustering configuration, and marks
statistical fitting unavailable. The four future point comparisons are
morphology-only, isotope-only, full-covariance integration and shared-covariance
integration, each with BIC. The protocol retains two PCs/view and jointly,
equal weights, standard scaling, complete features, K=1..4, minimum size five,
ten starts, one thread and the existing numerical settings. Statistical pipeline
semantics are pinned to commit `54fe5675b0308079f52c340cf801c59d8e3b7ea3`.

Execution, bootstrap and significance draws are disabled. A later checkpoint
must resolve or explicitly justify sampling identity/independence, document
reference evidence and authorize its analysis design. It must retain the source
of any assumption and separate descriptive reference agreement from evidence for
species boundaries. No absolute BIC comparison across different input score
spaces or automatic family selection is planned. Labels never tune PCs, weights,
K limits, eligibility or optimizer settings. Sex/year/island summaries are
reserved for evaluation, not silent adjustment or training.

This familiar dataset is a development/compatibility case. It does not supply
blinded validation, a genomic benchmark, shape validation, a near-boundary
speciation test or calibrated species probabilities. The two deferred studies
need separate format/identity/reference audits; their compatibility is not claimed.

## Reproduce the fit-free curation

From the checkout, create a fresh local directory outside the repository:

```bash
empirical_root="$(mktemp -d "${TMPDIR:-/tmp}/super-deli-empirical.XXXXXX")"
python examples/empirical-foundation/download.py \
  examples/empirical-foundation/profile.json \
  --output-dir "$empirical_root/source"
super-deli curate-empirical examples/empirical-foundation/profile.json \
  --source "$empirical_root/source/source_001.csv" \
  --protocol examples/empirical-foundation/protocol.json \
  --output-dir "$empirical_root/curated"
```

The source download is explicitly bounded and rejects unexpected redirects,
truncation, excess bytes and fingerprint changes. Network access is needed only
for that optional download step. Curation and all CI tests run offline. The
synthetic smoke is:

```bash
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/super-deli-smoke.XXXXXX")"
python examples/empirical-foundation/smoke.py --output-dir "$smoke_root/result"
```

`core.empirical` defines immutable source/column roles;
`core.empirical_protocol` enforces the frozen non-executable plan;
`workflows.empirical_curation` orchestrates existing structural and typed readers;
`benchmarking.empirical_cli` is a lazy command adapter. No estimator implementation
is duplicated or altered. Python support remains 3.11 and 3.12.

## Source credits

Gorman, Williams and Fraser (2014), [Ecological Sexual Dimorphism and Environmental
Variability within a Community of Antarctic Penguins](https://doi.org/10.1371/journal.pone.0090081).
Data were made available by the Palmer Station LTER program and the investigators.
The author-maintained [palmerpenguins mirror](https://github.com/allisonhorst/palmerpenguins/tree/8957207b78d6ccd1b4654a9dd9c9041b657478ab)
provides the locked copy and CC0 license. All credits and source citations are
retained; no personal filenames or data are added to the repository.
