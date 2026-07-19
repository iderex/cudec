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
- **Determinism**: same input → same output, bit-exact per code path.
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
