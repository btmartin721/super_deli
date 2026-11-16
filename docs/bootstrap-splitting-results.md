# Bootstrap splitting and cluster-size eligibility: results

All production models, defaults, preprocessing, resampling and historical p-values remain byte-for-byte unchanged. The checkpoint adds benchmark-only diagnostics with ordinary 100 starts. Detailed evidence stays local.

Validation used 24 fresh four-modality parents (six per N/law cell) and 24 direct Gaussian-score parents (twelve per N). Each parent supplied 25 paired bootstrap/unique-support comparisons, 25 independent known-parameter reference cohorts and 1,024 independent evaluation units. The main study ran 3,856 searches, with a separate 660-search preflight. Eight main whole-workflow replays preserve exact numerical arrays. No new significance tests, empirical fits, neural training or author contact.

## Single-population controls

Counts below are conditional draws, not independent datasets. Each raw-data row has six independent parents; each head-only row has twelve. K>1 is a component split, not rejection of a hypothesis or evidence for a species.

| Space | N | Parents | Ordinary bootstrap K>1 | Unique support K>1 | Independent reference K>1 |
|---|---:|---:|---:|---:|---:|
| Four raw modalities | 64 | 6 | 80/150 | 31/150 | 13/150 |
| Four raw modalities | 256 | 6 | 93/150 | 0/150 | 0/150 |
| Direct Gaussian scores (head only) | 64 | 12 | 267/300 | 38/300 | 7/300 |
| Direct Gaussian scores (head only) | 256 | 12 | 177/300 | 0/300 | 0/300 |

Original point fits also split 1/6 raw-control and 1/12 direct-Gaussian datasets at N=64; neither N=256 set split. Component selection can therefore overpartition before resampling as well. These small point samples do not establish a general error rate.

Independent reference cohorts come from the known simulation law, not an empirical fitted null. Four-group alternatives have shape signal only and balanced fixed generating counts; reference cohorts preserve those counts, while bootstrap changes group representation. This comparison supplies no unknown-label resampling solution. Direct Gaussian-score controls bypass ingestion, scaling, GPA, view PCA and fusion. Unique support changes weights, training N, BIC, preprocessing and row ordering together; these comparisons do not isolate any one mechanism.

## Four shape groups

Correct recovery requires K=4 and adjusted Rand agreement at least 0.8 with generating labels, assessed after fitting. Point results have six independent observations per N; bootstrap totals remain repeated measurements from those six.

| N | Point correct | Ordinary correct | Unique correct | Independent reference correct | Ordinary K=4 eligible |
|---|---:|---:|---:|---:|---:|
| 64 | 6/6 | 31/150 | 53/150 | 148/150 | 32/150 |
| 256 | 6/6 | 150/150 | 150/150 | 150/150 | 150/150 |

Post-fit truth accounting separates two mechanisms. At N=64, 86/150 draws contained a generating group with fewer than nine distinct original individuals. Of the 118 draws with ineligible native K=4, 85 also had this generating-group support deficit. The other 33 had adequate support for every generating group but different fitted partitions: their minimum fitted component sizes were one to eight distinct individuals. None of these 33 K=4 candidates reached ARI 0.8 on the original individuals (median 0.436), and none improved independent prediction over the selected lower-K candidate. Simply removing the size gate would not repair these saved candidates. Distinguishing an unfound alternative optimum from a change in the empirical likelihood objective remains unresolved; this checkpoint adds no truth-initialized fits. At N=256, none of the 150 draws had either deficit. Generating labels are used only for this post-fit diagnosis.

## Interpretation and limits

Ordinary bootstrap splitting exceeded the independent-reference rate in 4/4 control cells. These are descriptive operating-range comparisons, with six or twelve independent parents per cell. The JSON retains parent-level intervals, paired differences, exact parent-event intervals, conditional prediction gains, K=4 eligibility and failure bounds.
For raw multimodal controls at N=64, 80/80 audited bootstrap splits had no positive independent evaluation log-density gain over K=1 in the same fitted space.
For raw multimodal controls at N=256, 93/93 audited bootstrap splits had no positive independent evaluation log-density gain over K=1 in the same fitted space.
For direct Gaussian scores at N=64, 267/267 audited bootstrap splits had no positive independent evaluation log-density gain over K=1 in the same fitted space.
For direct Gaussian scores at N=256, 177/177 audited bootstrap splits had no positive independent evaluation log-density gain over K=1 in the same fitted space.

A bootstrap sample repeats observed individuals and loses distinct support. The nine-distinct-individual minimum applies to each component of the single best fitted estimator per K. The study does not salvage another start, lower the minimum, or interpret ineligibility as proof that fewer biological groups exist. Fixed full-retention coordinates preserve the selected view PCs; they do not remove finite-sample uncertainty or bootstrap discreteness.

Coassignment, exact cluster recovery/merging, common-point ARI and out-of-bag coverage are retained per diagnostic draw. Independent reference OOB quantities are undefined. Numerical disagreement withholds a result, preserves native evidence and stays in denominators; no replacement or K=1 fallback. Absolute densities or BIC are not compared across fitted coordinate spaces.

Intervals condition on the frozen optimizer and resampling streams. Descriptive t intervals across independent parents can be unreliable or degenerate near boundaries. Exact intervals describe explicitly defined parent-level events, such as any split among the fixed 25 draws; they do not turn draws into independent false-positive trials. No new resampling policy or uncertainty claim is promoted. A correction or significance workflow requires a separately bounded checkpoint.

## Verification and provenance

The fit-free archive trace covers all 2,250 prior draws, including 750 ordinary-100 draws. Independent checks reconstruct winning-candidate densities, distinct component sizes and likelihood-versus-BIC costs without changing prior arrays. The earlier unused-diverse-batch ambiguity is retained under its existing scope.

Local fixed-source testing passed 2,731 scientific tests and 1,858 foundation tests. Package and Python 3.11/3.12 CI verification accompany the published commit. All 571 existing source modules outside version/CLI registration are unchanged; only lazy CLI registration and the package version change outside the added benchmark modules.

Engineering checks exposed missing installed-package metadata, a source-freeze test racing with unfinished edits, and an external controller confusing .npy file digests with float-array digests. Original failures are retained. Repairs affected environment setup, test scheduling and the external metadata audit only; no scientific array, fitting rule, setting, numerical tolerance or historical p-value changed.
