# Amphipod evidence gates

`fixture/` contains only invented records and documentary statements. Its labels,
sequences and identifiers are engineering cases, not biological ground truth.
The preceding `../amphipod-audit/fixture` supplies the unchanged invented trait
audit. `plan.json` locks real public evidence retained separately, not bundled here.

```bash
python examples/amphipod-gates/smoke.py --output-dir /tmp/trait-evidence-smoke
```

The installed smoke blocks numerical-library imports and network connections.
It runs no clustering, bootstrap or significance analysis. Use a new output path.
See the [method](../../docs/amphipod-evidence.md) and
[findings](../../docs/amphipod-evidence-results.md).
