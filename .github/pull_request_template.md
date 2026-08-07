# What & why

<!-- What does this change do, and why? Link the issue: Closes #NNN -->

## Type of change

- [ ] Decode kernel / device code
- [ ] Format support (LZ4 / Snappy / GDeflate / Zstd)
- [ ] API surface
- [ ] Performance
- [ ] Bug fix
- [ ] Refactor / code quality
- [ ] Build / CI / supply chain
- [ ] Docs

## Engineering checklist

- [ ] Fail-closed preserved: every new parse/decode path rejects invalid
      input with a defined error; no out-of-bounds access is reachable from
      hostile input, and every reject path has a negative test.
- [ ] Oracle coverage: decode output is diff-tested against the CPU
      reference (liblz4 / zlib / libzstd) for every touched format path.
- [ ] Determinism preserved (same input → same output, bit-exact), or the
      break is a documented, gated decision.
- [ ] Every CUDA API call's error is checked; no exceptions cross the C ABI.
- [ ] An adversarial review (`/security-review`) was run for changes to
      kernels, format parsing, the C-ABI boundary, build/tools, or workflows.
- [ ] Review record: per lens, its verdict and its findings, with **CONFIRMED
      __ / PLAUSIBLE __** counts as raised (a finding is CONFIRMED only if a
      reproduction was produced). Every finding of either kind carries a
      disposition - FIX, a reasoned DECLINE, or DEFER as its own issue.

## GPU sanitizer gate

Required from M1 onward for every change that touches device code. CI has no
GPU, so the pasted output below is the evidence and there is nothing that
reds without it. Produced by `scripts/sanitize-gpu.sh` inside the pinned
container; the invocation is in [CONTRIBUTING.md](../CONTRIBUTING.md).

- [ ] Not applicable: this change touches no device code.

```
commit:    <sha the sweep ran against>
gpu:       <name, driver version>
cuda:      <nvcc release>
sanitizer: <compute-sanitizer version>

  <test>           memcheck   PASS/FAIL
  <test>           racecheck  PASS/FAIL
  <test>           initcheck  PASS/FAIL
  <test>           synccheck  PASS/FAIL
```

racecheck reports shared-memory hazards only, so a clean sweep is not
clearance for global memory. If no route to a device the sanitizer can attach
to was available, say so here with the command that shows it, and leave the
block empty rather than removing it.

## Performance checklist

Fill in for any change on the hot path.

- [ ] The claim is measured, not reasoned: before/after numbers with GPU
      model, driver, CUDA version, corpus, and chunk-size distribution.
- [ ] No regression against the recorded baseline (docs/BENCHMARKS.md), or
      the regression is justified below.

## Quality checklist

- [ ] Minimal: the change adds no more code than the problem requires; no
      duplication, no dead code.
- [ ] Self-documenting: intention-revealing names and small, single-purpose
      functions; comments explain why, not what.
- [ ] Conformance tests pass, and any new structural property this change
      establishes is locked in as a new conformance test.

## Verification

Paste the local gate results (or confirm CI is green).

- [ ] `cmake --build build` - builds clean.
- [ ] `ctest --test-dir build` - green (oracle diffs, negative tests,
      conformance tests).
- [ ] Prettier (`**/*.{md,yml,yaml}`) - clean.
- [ ] Docs synced: MASTERPLAN / README / BENCHMARKS reflect any behavior,
      API, or performance change.

## Notes

<!-- Trade-offs, declined review findings and why, follow-ups, or anything
explicitly out of scope. -->
