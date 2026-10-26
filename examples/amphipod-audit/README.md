# Groundwater-amphipod source audit

`fixture/` contains six entirely invented rows and four invented molecular
records. It tests failure and caution handling, not biological performance.
`plan.json` locks the real source bytes; those bytes must be obtained separately
and retained locally. No actual specimens, measurements, sequence strings or
personal metadata are distributed here. The source's people fields remain local.

```bash
super-deli audit-trait-source examples/amphipod-audit/fixture/plan.json \
  --source-dir examples/amphipod-audit/fixture --output-dir /tmp/invented-trait-audit
python examples/amphipod-audit/smoke.py --output-dir /tmp/installed-trait-smoke
```

The command requires an unused output directory. It performs no network requests,
formula evaluation, model fitting, bootstrap or significance analysis. See the
[method](../../docs/amphipod-audit.md) and [findings](../../docs/amphipod-audit-results.md).
