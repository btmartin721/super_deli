# Fit-free empirical evidence assessment

`evidence.json` binds reviewed sources/claims to the unchanged empirical curation
profile and protocol. It contains no source capture IDs, analytical records,
source-body snapshots or fit permissions. Published results are small aggregates.

See [method/CLI](../../docs/empirical-evidence-gates.md) and
[findings](../../docs/empirical-evidence-gates-results.md). Real-data reproduction
requires the preserved local source, curation output and reviewed evidence
snapshots; no network access occurs in the assessment CLI.

The fixture is entirely invented: its `primary_publication` role exercises a
schema path and denotes no actual publication or biological evidence. The smoke
reuses the invented 12-row foundation source, retains its 10 complete rows, then
runs the assessment with numerical imports and network connections blocked:

```bash
python examples/empirical-gates/smoke.py --output-dir /absolute/new/smoke
```

No model, bootstrap or significance test runs. Source/measurement identity is
separate from biological independence, and reference provenance is separate from
independent specimen-level truth. Existing defaults, p-values and schema-1
curation behavior are unchanged.
