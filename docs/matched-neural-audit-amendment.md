# Approved numerical-audit amendment for matched neural integration

Status: explicitly approved by the user on 2026-11-02 after the original
scientific runs and serial-audit failure. No candidate settings changed.

The 472 preflight and 2,708 main scientific workflows have finished without
replacement. The main independent serial NumPy optimizer reconstruction stopped
on heavy_tail_n64_0001 / paired_202: final parameters differ by at most
1.43e-7, and standardized latent coordinates by 2.19e-7. The full rollout misses
the frozen componentwise weight guard. Preserve that failed audit and every fit;
do not increase that guard or claim that the original main audit passed.

Approved narrowly bounded repair:

1. Preserve the current source freezes, all failed serial audits, all original
   inputs/outputs/settings, and the completed scientific candidate selections.
   Keep models, defaults, all historical p-values and empirical gates unchanged.
2. Add a separate independent NumPy auditor that checks the actual optimizer
   trajectory one update at a time. A read-only observer captures the current
   numeric parameters, raw gradients and Adam moments before/after each update.
   Independently compute every parameter gradient, moment recurrence and update;
   compare it with that actual step, then discard transient matrices. Save
   per-step error maxima, hashes, initialization/final/loss parity and failures.
   This tests each step directly instead of demanding that independent floating-
   point rollouts remain on an identical trajectory after 500 nonlinear updates.
3. Freeze per-step tolerances before running it: gradients rtol=1e-10,
   atol=1e-12; moments rtol=1e-11, atol=1e-13; parameters rtol=2e-11,
   atol=2e-12. Require exact initial/final weights and all 501 training losses
   against the original deterministic PyTorch fits. Never change tolerances
   after preflight or main outcomes. Test deliberate numerical corruptions,
   read-only observer parity, and independent finite-difference gradients.
4. Run a separate 24-replay full-settings preflight: all three seeds at original
   N=64/256 and retained D=4/8/12 (18), plus seed-zero subsamples at N=51/204
   for all three widths (6). Use already retained observations, no fresh data.
   Require a separately audited runtime/storage gate. Preserve failed gates.
5. If that passes, perform one deterministic training-only verification replay
   for each of the 342 preflight and 1,818 main completed neural fits: 2,160
   verification replays, distinct from scientific candidates and the 24 probes.
   Check all 500 steps in each. Rerun no HDBSCAN/GMM search, make no candidate,
   optimizer-seed or model-selection changes, and draw no new subsamples.
6. Keep original serial-rollout failures as diagnostic evidence. Complete all
   other source/PCA/mixture/density/support/replay audits. Publish a report only
   if the new per-step audit and exact replay parity pass; otherwise stop with
   the failure retained. State this audit amendment and its post-run timing.
7. Keep the original 2 GiB retained-evidence cap, at least 8 GiB free, at most
   two single-thread workers, and require projected cumulative main plus all
   audit/replay work below two hours. No cap increase or reduced scientific grid.
   Run Python 3.11/3.12 tests, package/installed checks and exact-master CI;
   publish code/docs/figures/small aggregates only and stop. Contact nobody.

This is additional numerical verification, not additional statistical samples,
biological evidence, a model repair or a relaxed species/significance guarantee.
