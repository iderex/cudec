# Contributing

## Process

Issue-driven, gate-driven:

1. Every change — feature, fix, perf work, docs — starts as a **GitHub issue**
   with a type, `area:` and `priority:` label and a milestone.
2. Work happens on a short-lived branch off `main`:
   `feature/…`, `fix/…`, `perf/…`, `harden/…`, `chore/…`, `refactor/…`.
3. The PR fills the template honestly and references the issue (`Closes #N`).
   PRs merge with a merge commit once CI and review are green.

## Code standard

- **Fail-closed decoding.** A bitstream that does not validate produces a
  defined error code, never an out-of-bounds access, never a partial guess.
  Every reject path carries an explicit negative test.
- **The oracles decide.** Decode output is diff-tested on real and fuzzed
  corpora against the reference implementations — liblz4 today, with zlib and
  libzstd joining as the DEFLATE and Zstd formats land.
- **Format provenance.** Every format is implemented from its public
  specification only — the LZ4 block/frame spec, Snappy, DEFLATE (RFC 1951),
  the GDeflate spec, Zstd (RFC 8878). The reference decoders (liblz4, zlib,
  libzstd) are test oracles, never a source to copy from: they ship under their
  own BSD/zlib licenses and pasted code would muddy the provenance of an
  Apache-2.0 tree. "The oracles decide" means the reference settles
  _correctness by differential test_ — not that its code may be _derived from_.
  nvCOMP is proprietary: never copy its headers or source, and cudec claims no
  compatibility with it (nominative references and CPU-only benchmark
  comparisons are fine).
- **Termination**: every loop whose exit depends on a value read from the
  bitstream carries an explicit decrementing fuel cap, sized so no input the
  validation ladder admits can reach it. A decoder that hangs on hostile input
  has failed open in the availability direction; a configure-time check reds
  the build on a fuel-free loop in the decode path, and every ctest entry has a
  finite `TIMEOUT`.
- **Warp collectives.** Four points, checked on every kernel review:
  1. **Mask provenance** — a collective's mask is the full-warp constant or
     `__activemask()`, never a value computed from the bitstream.
  2. **All participants reach it** — every lane the mask names arrives at the
     collective; loop bounds and branches around one stay lane-uniform.
  3. **No source lane or predicate from unvalidated input** — a shuffle's
     source lane and a vote's predicate are program constants or validated
     quantities, never a parsed length, offset, or token.
  4. **No legacy intrinsics** — `__shfl`, `__ballot`, `__any`, `__all`,
     `__match_any` and friends are banned; only the `_sync` forms are used.
- **Determinism**: same input → bit-identical output, on every path and in
  every supported launch configuration — the `gpu_to_gpu` level in NVIDIA's
  CCCL vocabulary. A kernel that maps work onto lanes states which geometries
  it supports and refuses the rest, rather than assuming its caller. The scope,
  the qualifiers, and the tested axes are in
  [docs/DETERMINISM.md](docs/DETERMINISM.md); no floating-point type or atomic
  belongs anywhere in the sources.
- **Performance claims are measured**, never reasoned: numbers ship with GPU
  model, driver, CUDA version, corpus, and chunk-size distribution.
- **Readable kernels.** Small single-purpose device functions with
  intention-revealing names; comments explain _why_, never _what_. Every
  CUDA API call's error is checked; no exceptions cross the C ABI.
- **Minimal.** The least code that does the job; structural rules are locked
  in by conformance tests — do not weaken them.

## Building & testing

```sh
cmake -B build && cmake --build build   # host-only; needs just a C compiler
```

The CUDA engine and its on-device tests are opt-in (`-DCUDEC_ENABLE_CUDA=ON`)
and need a CUDA 12.x toolchain plus a GPU for the tests — see the README's
container command for the maintained path (build directory `build-cuda`,
`ctest --test-dir build-cuda`).

All repo artifacts — code, comments, commits, PRs, issues — are written in
English.
