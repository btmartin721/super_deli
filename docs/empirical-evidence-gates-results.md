# Empirical readiness audit results

The assessment completed for all 344 locked source captures and the unchanged
330-row complete cohort. It checked all 1,980 retained numeric tokens and every
analytical/reference/metadata/source-index link. There were no empirical fits,
bootstrap draws, significance draws, new cohorts or changes to historical
p-values. See the [method and CLI](empirical-evidence-gates.md),
[reviewed claims](../examples/empirical-gates/evidence.json),
[readiness report](../examples/empirical-gates/results/readiness.json) and
[aggregate source facts](../examples/empirical-gates/results/source-facts.json).

## Sampling and identities

The `studyName`/`Individual ID` composite identifies 344 distinct source records.
`Individual ID` alone has only 190 distinct tokens. This establishes capture
identity rather than persistent biological identity. The [original field
methods](https://doi.org/10.1371/journal.pone.0090081) describe sampling both adults
at marked study nests over three seasons. Their reported nest/adult totals are
172/344. Neither paired sampling nor matching totals proves that file suffixes
encode an authoritative nest crosswalk.

The [pinned data dictionary](https://github.com/allisonhorst/palmerpenguins/blob/8957207b78d6ccd1b4654a9dd9c9041b657478ab/man/penguins_raw.Rd)
calls the individual field unique, which conflicts with its literal source
values unless a scope is supplied. An [existing identifier
report](https://github.com/allisonhorst/palmerpenguins/issues/102) has no comments
in the retained snapshot. This flags a discrepancy; the issue is not treated as
an authoritative explanation of the numbering. Reused codes cannot establish
recaptures, distinct birds or nest membership. No mapping is invented.

## References and access limits

Published species assignments cover all 330 retained captures: 141 Adélie,
67 chinstrap and 122 gentoo. They are admitted for descriptive development
reporting. The field paper documents molecular sexing, which does not itself
independently adjudicate species identity for each source record. Its citation
of an earlier molecular phylogeny establishes taxonomic context, without a
verified specimen crosswalk or blinded validation.

Five content downloads and two HTTP-403 archive failures were retained. Of the
five downloaded bodies, four were usable: field paper, pinned dictionary, issue
and empty comment list. The earlier-phylogeny PubMed request returned HTTP 200
with a browser/cookie challenge; it is explicitly unavailable as article content.
Only its citation in the accessible field paper is verified here. No access
restriction was bypassed and no taxonomic revision was silently adjudicated.

Direct EDI metadata access remained unavailable for both checked versions.
Equivalence between the actual pinned author mirror and cited upstream versions
therefore remains unestablished. The mirror lock and all previous citations are
preserved.

## Decision and next checkpoint

Ingestion and descriptive reporting of field assignments are available.
Independent-unit fitting, resampling, significance and independent
species-boundary validation remain blocked for specific evidence requirements;
the existing grouped predictive workflow is inapplicable to this capture design.
The readiness audit is complete even though those scientific prerequisites are
not established by the available public records. No gate was opened merely to
produce an empirical benchmark.

Next obtain an authoritative nest/persistent-bird crosswalk and independently
adjudicated specimen references, or audit a replacement empirical source with
those records. Then freeze a justified sampling/evaluation design before point
fitting. A decision to use conditional sampling assumptions would need a separate,
explicit protocol and would not certify independent biological truth.

Local verification passed 1,616 scientific tests and 1,068 foundation tests
(with 103 optional-dependency skips), plus both installed offline smokes. The
installed empirical assessment reproduced every output byte. The wheel/source
archive matched all 208 source modules; Python 3.11 grammar and the existing
`>=3.11,<3.13` metadata were checked. Existing estimators, defaults and schema-1
curation/protocol files are byte-identical to the preceding master.

Preservation and exact-head Python 3.11/3.12 CI are publication requirements;
completion is recorded separately after final master CI, rather than inferred
from local Python 3.12 checks. Full evidence stays local under
`~/.codex/outputs/super-deli/empirical-gates-20261025/`.
