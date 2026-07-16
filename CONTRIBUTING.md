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
- **The oracles decide.** Decode output is diff-tested against the reference
  implementations (liblz4, zlib, libzstd) on real and fuzzed corpora.
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
cmake -B build && cmake --build build   # host stub today; CUDA arrives with M1
ctest --test-dir build                  # once tests/ exists
```

All repo artifacts — code, comments, commits, PRs, issues — are written in
English.
