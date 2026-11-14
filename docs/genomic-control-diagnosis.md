# Fit-free genomic neural control diagnosis

This separately versioned CLI/API reads the completed schema-0.78 genomic study.
It does not read a new genomic dataset, fit a model, draw samples or generate
p-values. It preserves the existing neural/PCA commands and their serialization.

Use Python 3.11 or 3.12 and install `.[baseline]` from the repository. Torch and
SNPio are unnecessary for this archival command: original SNPio allele calls,
dosages and fitted parameters are read from the sealed study artifacts.

Create a configuration bound to your completed local archive:

```python
from pathlib import Path
from super_deli.core.provenance import fingerprint_file
from super_deli.benchmarking.genomic_diagnosis_config import GenomicDiagnosisConfig
from super_deli.io.config import write_json

archive = Path("/absolute/path/to/genomic-neural-20261114")
config = GenomicDiagnosisConfig(
    source_completion_sha256=fingerprint_file(archive / "completion.json").sha256,
    source_manifest_sha256=fingerprint_file(archive / "verification-manifest.json").sha256,
    source_commit="c12ea70a74f77bb405e0439b9b80ab17348b063d",
)
write_json(config.to_dict(), "diagnosis.json")
```

```bash
super-deli diagnose-genomic-neural /absolute/path/to/genomic-neural-20261114 \
  --diagnosis-config diagnosis.json --output-dir genomic-diagnosis-01
```

The directory must contain its completed source receipt, verification manifest
and lossless `verification-evidence` tree. The command accepts only the declared
source version and role, verifies each consumed regular file against its seal,
and rejects missing, mutated, escaping or inconsistent records without fallback.
`--case-id` may be repeated for an explicit audit subset; subsets never produce
a development go/no-go decision. All output directories must be new.

The API is `benchmarking.genomic_diagnosis.run_genomic_diagnosis`, with archive
path and explicit `config`, `output_dir` and optional `case_ids`. Outputs contain
per-map JSON/NPZ diagnostics, a checksum-bound source inventory, aggregate
`summary.json` and a final artifact-hashed `completion.json`. Failures retain
`failure.json` and withhold the scientific decision. No source is overwritten.

Invented algebraic fixtures use a distinct `invented_fixture` role/version and
always report `invented_fixture_no_scientific_decision`. The installed smoke
constructs those numeric artifacts without estimator or optimizer calls:

```bash
python examples/genomic-control-diagnosis/smoke.py --output-dir invented-diagnosis-01
```

See [the completed diagnosis and stopping decision](genomic-control-diagnosis-results.md).
Coefficients and moments are descriptive; they do not identify causal mechanisms,
calibrate significance or establish species. Source preflight observations,
main datasets, neural seeds and archived draws remain separately accounted for.
