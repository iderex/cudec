# Contributing

## Process

Issue-driven, gate-driven:

1. Every change - feature, fix, perf work, docs - starts as a **GitHub issue**
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
  corpora against the reference implementations - liblz4 and snappy today,
  with zlib and libzstd joining as the DEFLATE and Zstd formats land.
- **Format provenance.** Every format is implemented from its public
  specification only - the LZ4 block/frame spec, Snappy, DEFLATE (RFC 1951),
  the GDeflate spec, Zstd (RFC 8878). The reference decoders (liblz4, zlib,
  libzstd) are test oracles, never a source to copy from: they ship under their
  own BSD/zlib licenses and pasted code would muddy the provenance of an
  Apache-2.0 tree. "The oracles decide" means the reference settles
  _correctness by differential test_ - not that its code may be _derived from_.
  nvCOMP is proprietary: never copy its headers or source, and cudec claims no
  compatibility with it (nominative references and CPU-only benchmark
  comparisons are fine).
  **GDeflate is the one deliberate exception**, and it is narrow. That format
  has no single canonical specification, and the artifact that carries a
  patent grant is the Apache-2.0 code rather than the draft: the M4 work may
  read and derive from the DirectStorage `GDeflate/` reference and the NVIDIA
  fork of libdeflate, and every reliance is recorded at the site and in the
  pull request - which file, what was taken, why. Section 16 of
  [MASTERPLAN.md](docs/MASTERPLAN.md) is where that posture and its reason
  live; it widens to no other format and to no other decoder, and it is
  reasoning discipline rather than a statement that anything is patent-safe.
- **Termination**: every loop whose exit depends on a value read from the
  bitstream carries an explicit decrementing fuel cap, sized so no input the
  validation ladder admits can reach it. A decoder that hangs on hostile input
  has failed open in the availability direction; a configure-time check reds
  the build on a fuel-free loop in the decode path, and every ctest entry has a
  finite `TIMEOUT`.
- **Warp collectives.** Four points, checked on every kernel review:
  1. **Mask provenance** - a collective's mask is the full-warp constant or
     `__activemask()`, never a value computed from the bitstream.
  2. **All participants reach it** - every lane the mask names arrives at the
     collective; loop bounds and branches around one stay lane-uniform.
  3. **No source lane or predicate from unvalidated input** - a shuffle's
     source lane and a vote's predicate are program constants or validated
     quantities, never a parsed length, offset, or token.
  4. **No legacy intrinsics** - `__shfl`, `__ballot`, `__any`, `__all`,
     `__match_any` and friends are banned; only the `_sync` forms are used.
- **Determinism**: same input → bit-identical output, on every path and in
  every supported launch configuration - the `gpu_to_gpu` level in NVIDIA's
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
  in by conformance tests - do not weaken them.

## Building & testing

```sh
cmake -B build && cmake --build build   # host-only; needs just a C compiler
```

The CUDA engine and its on-device tests are opt-in (`-DCUDEC_ENABLE_CUDA=ON`)
and need a CUDA 12.x toolchain plus a GPU for the tests - see the README's
container command for the maintained path (build directory `build-cuda`,
`ctest --test-dir build-cuda`).

### The tests that allocate over 4 GiB

Some arms of the decoder are selected by a capacity test and nothing smaller
reaches them: `src/chunk_decode.cuh` picks a 64-bit overlap gather for a match
longer than 2^32, and no ordinary fixture is that large. `huge_match_gpu`
reaches that one with a single 4 GiB decode. It is built by every CUDA
configure, so it cannot rot, and it is registered as a ctest entry only under
`-DCUDEC_HUGE_TESTS=ON`, because 4 GiB walked by one warp is not a shape every
run should carry. Its label is `huge` rather than `gpu`, which keeps it out of
the sanitizer sweep:

```sh
cmake -B build-huge -DCUDEC_ENABLE_CUDA=ON -DCUDEC_HUGE_TESTS=ON
cmake --build build-huge -j
ctest --test-dir build-huge -L huge --output-on-failure
```

On a device that cannot hold the allocation it FAILS with the shortfall in the
message, and does not skip. A skip on the only test that reaches an arm reads
exactly like a pass, and the guard was unprovable for as long as no test
selected it. Write the next capacity-selected arm the same way: reach it, or
say in the pull request that nothing does.

### The GPU sanitizer gate

Any change that touches device code carries a Compute Sanitizer sweep, from M1
onward. All four tools run over every `gpu`-labelled ctest target, and the
output goes in the pull-request body: CI has no GPU, so nothing in the
workflow can red for a missing sweep and the paste is the whole evidence.

`scripts/sanitize-gpu.sh` is the runner. It discovers its targets with
`ctest -L gpu -N` rather than from a list, and it refuses a sweep that covered
nothing - an absent `compute-sanitizer`, an empty discovery, a missing binary
or a known GPU test gone from the discovered set are errors there and never
skips. The commit is passed in because the image ships no git:

```sh
docker run --rm --gpus all -v "$PWD:/w" -w /w \
  nvidia/cuda:12.6.2-devel-ubuntu24.04@sha256:738fba0fbdb225b7a2931c58a5c8f03a84d3cd2f6a84975826a157339ef750b8 \
  sh -c "apt-get update -q >/dev/null && \
         apt-get install -yq cmake >/dev/null 2>&1 && \
         cmake -B build-cuda -DCUDEC_ENABLE_CUDA=ON && \
         cmake --build build-cuda -j && \
         CUDEC_COMMIT=$(git rev-parse HEAD) scripts/sanitize-gpu.sh build-cuda"
```

racecheck checks shared-memory hazards only. A clean run says nothing about
global-memory races, and reading it as clearance for them is the mistake this
sentence exists to stop.

The gate is written down here in full, and it is **parked**. No route to a
device the four tools can attach to exists on this project's route, and the
one attempt that could have produced one was made and failed. Under WSL2 the
device is in WDDM mode, the debugger interface the sanitizer needs is absent,
and all four tools report the same two initialization errors against a program
with nothing wrong in it. Re-measured on 2026-08-24 against the newest pairing
this machine can hold (driver 610.88, CUDA 13.3, compute-sanitizer 2026.2.1.0),
run in the WSL distribution directly rather than in the container above:

```
for t in memcheck racecheck initcheck synccheck; do
  compute-sanitizer --tool "$t" --error-exitcode 1 ./clean; done
every tool: exit=1
========= Error: Failed to initialize WDDM debugger interface. Please run EnableDebuggerInterface.bat as an administrator
========= Error: Device not supported. Please refer to the "Supported Devices" section of the sanitizer documentation
```

A seeded 4 MiB out-of-bounds write does exit nonzero under memcheck, and that
nonzero says nothing: what surfaces is the runtime's own
`cudaErrorIllegalAddress` on `cudaDeviceSynchronize` with a host backtrace, no
`Invalid __global__ write` and no device backtrace. The same fault run without
the sanitizer produces the same information.

Parked means owed and unproducible, not waived and not moved. It was not moved
to a paid GPU runner: that is a standing cost, and it comes back as its own
issue if device-side coverage becomes load-bearing. What would unpark it is a
machine whose device offers a debugger interface, and the untried route is the
Windows-native CUDA toolchain, which ships the `EnableDebuggerInterface.bat`
the first error names. Until then, say in the pull request that the sweep could
not be produced, with the command that shows it, rather than leaving the block
looking answered. Issue #258 holds the parking.

### The fuzz gate

The differential libFuzzer targets under `fuzz/` run in CI from a committed,
hash-pinned seed corpus: `.github/workflows/fuzz.yml`, bounded on every pull
request and longer on a weekly schedule. It is not a required check - which
contexts a merge requires is branch-ruleset state and the workflow does not set
it - so read its result rather than assuming a merge waited for it.

The build is Clang-only (libFuzzer ships with Clang) and needs no GPU: `fuzz/`
has no CUDA sources and links no cudec archive, so it runs on a plain runner
while the rest of the gate runs in the CUDA container.

```sh
cmake -B build-fuzz -DCUDEC_FUZZ=ON -DCUDEC_SANITIZE=address,undefined \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-fuzz
build-fuzz/fuzz/fuzz_lz4_block work-dir fuzz/corpus/fuzz_lz4_block \
  -max_total_time=60 -max_len=4096
```

**Every finding is kept forever, in two places.** An input that crashed a
target, tripped a sanitizer, or diverged from a reference becomes a permanent
seed in `fuzz/corpus/<target>/` with its hash in `fuzz/corpus/SHA256SUMS`, AND
the same bytes become a negative in `tests/` pinning the exact status the
parser now returns. Neither alone is enough: a corpus entry is covered only
while the fuzz job runs, and a pinned negative alone loses the bytes that found
the defect. `scripts/check-fuzz-corpus.sh` refuses a corpus that no longer
matches its manifest and an emptied target directory - the shape libFuzzer
otherwise reads as a legal start-from-scratch run and exits 0 on.

`-DCUDEC_ENABLE_HIP=ON` exists and currently refuses on every machine: there
are no HIP device sources yet, so the option fails the configure rather than
handing back a host-only library that looks like a working port. The two
options are mutually exclusive in one build tree.

### Documents and the paths they name

A repository path written in a tense-present document must resolve.
`scripts/check-doc-paths.sh` refuses one that does not, in the `format` job, and
it runs its own self-tests first so a regex that stopped matching cannot pass
the tree silently.

Where a document must name a file the tree only **owes** - an M4 design
paragraph naming the GDeflate table headers, say - declare it and name the issue
that owes it:

```
The canonical table will live in `path/to/file.h` (planned: #NNN).
```

The placeholder in that example is deliberate and is worth one sentence, because
the trap costs a red gate on this file. A real path written here would be a live
declaration like any other, judged against the issue it names, so illustrating
the spelling with one of the M4 table headers and its issue number would make
this section red the day that issue closes - the check refusing its own
documentation. Worse, it reds immediately, because the illustration is a
reference to a file the tree does not hold. A placeholder outside the tree's
directories is invisible to the scan and is the safe way to show the shape.

That is not a grace period. The declaration is judged against the issue: the
moment it closes without the path existing, the reference reds exactly as a
broken present-tense path would, so a promise cannot outlive the work it pointed
at. A path that neither resolves nor carries a declaration is refused outright.

Two things the gate deliberately does not reach, so a green run is not read as
"no document contradicts the tree". A path under a directory this tree does not
have is another project's file and passes - which is how `docs/MASTERPLAN.md`
may keep naming nvCOMP's `src/lowlevel/gdeflateKernels.cu` - and with it any
path whose whole directory is still owed. And `CHANGELOG.md` and
`docs/BENCHMARKS.md` are out of scope entirely: they are append-only records of
what was true when each entry was written, so a since-renamed path is honest
text there rather than a defect.

All repo artifacts - code, comments, commits, PRs, issues - are written in
English.
