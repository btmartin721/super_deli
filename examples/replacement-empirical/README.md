# Fit-free replacement empirical source audit

`plan.json` contains source fingerprints and the bounded review policy for the
juvenile-skate candidate. It contains no raw specimen rows, voucher crosswalks
or sequences. `results/` supplies small audited aggregates only.

See [method and command](../../docs/replacement-empirical-audit.md) and
[findings](../../docs/replacement-empirical-audit-results.md). Source-reported
COI assignments, literal source identities and candidate molecular namespace
links remain separate. No empirical benchmark or independence assumption is
approved by an ingestion success.

All files under `fixture/` are entirely invented engineering artifacts. The DOCX
is a minimal synthetic membership-table container, and its primary-article XML
is invented text. Neither represents real scientific or biological evidence.
Run the offline installed-CLI smoke with numerical imports and network
connections blocked:

```bash
python examples/replacement-empirical/smoke.py --output-dir /private/tmp/replacement-audit-smoke
```

Use a new output directory. No estimator, sliding, PCA, bootstrap or significance
workflow runs, and source zeroes/missing tokens remain unchanged.
