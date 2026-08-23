# cudec - masterplan

The design record: what is being built, why it is shaped this way, and in
which order. Settled decisions live here with their rationale; open questions
are listed at the end and get settled through design issues before the code
that depends on them is written.

## 1. Positioning

cudec is an open-source CUDA library that batch-decodes the standard
compression formats - LZ4, Snappy, GDeflate/DEFLATE, and eventually Zstd -
on the GPU.

The field today:

| Existing work                   | Why it does not fill the gap                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| nvCOMP / nvCOMPDx (NVIDIA)      | Proprietary since v2.3; NVIDIA-only; not auditable                                                                                   |
| dietgpu (Meta)                  | Open, but its own rANS format - not format-compatible                                                                                |
| GDeflate reference (MS/NV)      | Open spec + CPU codec; its GPU decoder is the HLSL shader in the row below                                                           |
| DirectStorage `GDeflate.hlsl`   | Open (Apache-2.0) GPU GDeflate decoder, but a shader inside a D3D12 runtime                                                          |
| DirectStorage `zstd/` (dev)     | Open HLSL zstd decode shaders; the sample says not for production, and the runtime compiles versions of them in as a driver fallback |
| vkd3d-proton `cs_gdeflate.comp` | Open (LGPL-2.1) GLSL GDeflate decoder, bound to the vkd3d-proton runtime                                                             |
| hipCOMP-core (AMD/ROCm)         | MIT fork of nvCOMP branch-2.2 on HIP; early-access preview, not for production                                                       |
| Academic prototypes             | Unmaintained research code                                                                                                           |

Where those rows come from, so the next pass does not re-derive them from
memory: `GDeflate/shaders/GDeflate.hlsl` and the `zstd/` tree on the
`development` branch of
[microsoft/DirectStorage](https://github.com/microsoft/DirectStorage);
`libs/vkd3d/shaders/cs_gdeflate.comp` in
[vkd3d-proton](https://github.com/HansKristian-Work/vkd3d-proton);
[ROCm/hipCOMP-core](https://github.com/ROCm/hipCOMP-core). Read at those
paths on 2026-08-05.

The zstd row is worth a paragraph of its own, because the open/production
line runs through the middle of it rather than around it. `zstd/README.md` on
that branch calls the shaders "reference implementations ... authored in HLSL"
and says in red that the code "is currently in development and should NOT be
used in production environments or for released products" - and, four lines
later, that "versions of these same shaders are also included/compiled inside
the DirectStorage runtime to be used as a fallback for GPUs that do not have
optimized zstd GPU decompression driver support", most game-ready drivers
carrying their own faster implementation. So an open GPU zstd decoder exists,
some of it ships, and the fast paths on both sides of it are closed. The
shaders live under `zstd/zstdgpu/Shaders`.

No milestone here claims a "first", and that is deliberate. A code search
for GDeflate in CUDA sources on the same day returned a partial decoder in
an application tree (`crush-gpu/src/shader/gdeflate_decompress.cu` in
john-agentic-ai-tools/crush: fixed-Huffman blocks only, no declared licence),
so a bare "the first open CUDA GDeflate decoder" is refutable and is not
made. hipCOMP-core's `src/lowlevel/gdeflateKernels.cu` is not a
counterexample in the other direction - it is a status-conversion shim
around an external `gdeflate::` component, not a decoder.

There is no maintained open-source CUDA library that decodes the standard
formats on the GPU. "Maintained", "CUDA" and "library" are all load-bearing:
the open GPU decoders above are shaders belonging to a graphics runtime, and
hipCOMP-core is a HIP preview downstream of a four-year-old fork point. The
value proposition is not price (nvCOMP is free to use) but
**auditability** (decompressors are classic attack surface; every bounds
check here is readable, tested, and fuzzed), **portability** (a HIP port is a
planned milestone; what that claim is and is not is section 3 item 5, and it
is not an empty AMD field), and **hackability**.

## 2. Scope

- **Decode-only.** Compression stays on the CPU: that is where the encoders,
  the tooling, and the flexibility live, and encode throughput is rarely the
  bottleneck. Decode is.
- **Batch-oriented.** The GPU wins when many independent chunks decode in
  parallel - asset streaming, analytics scans, ML data loading. A single
  small file over a cold PCIe bus loses to the CPU and always will; the
  documentation says so.
- **Formats over percentage points.** The library's value is format
  coverage. Once a format hits a healthy fraction of memory bandwidth, the
  next format outranks the next percent.

## 3. Format ladder (and why this order)

1. **LZ4** (M1/M2) - byte-oriented, no entropy coding: token/literal/match
   sequences only. The simplest correct GPU decode, the fastest path to an
   honest benchmark, and the kernel-family foundation everything else
   reuses. Block format first, then the frame format and the batch API.
2. **Snappy** (M3) - structurally close to LZ4 (varint-length literals and
   copies, no entropy stage); a cheap second format that proves the kernel
   family generalizes.
3. **GDeflate** (M4) - the strategic differentiator. The format is designed
   for GPU decode (the DEFLATE bitstream is split into 32 interleaved
   sub-streams so a full warp reads Huffman codes in parallel) and the spec
   is open. Open GPU decoders already exist, but each one is a shader owned
   by a graphics runtime - DirectStorage's HLSL decoder and vkd3d-proton's
   GLSL one (section 1). What does not exist is an open CUDA GDeflate
   decoder callable as a library, on Linux, in HPC, anywhere, and that is
   what cudec ships.
4. **Zstd decode** (M5) - the flagship and by far the hardest: FSE/Huffman
   entropy stages feeding an LZ77 sequence executor with long-range matches.
   Attempted only once the kernel family, the oracle net, and the benchmark
   discipline are proven on 1–3. An open GPU zstd decoder already exists, in
   HLSL, in a graphics runtime, and partly shipped (section 1); the drivers'
   own zstd paths are closed and so is nvCOMP. What does not exist is an open
   CUDA zstd decoder callable as a library, which is the claim space here and
   is not a first.
5. **HIP port** (M6) - portability as a moat. The kernels stay on warp-level
   primitives available on both vendors to keep this tractable. AMD is not
   an empty field: hipCOMP-core carries HIP LZ4 and Snappy decode (section
   1). The claim M6 can make is one audited fail-closed kernel family,
   single-source across vendors behind a stable C ABI - never "first on
   AMD".

## 4. Architecture pillars

- **C ABI, thin C++ inside.** The public surface is a small C header
  (`include/cudec.h`); no exceptions cross it, every function returns a
  defined status. Internally plain CUDA C++ with no dependencies beyond the
  CUDA toolkit.
- **Batch API.** The core entry point decodes N independent chunks described
  by device-side arrays (src pointers/sizes, dst pointers/capacities) on a
  caller-provided stream, reporting per-chunk status - modeled on the shape
  proven by nvCOMP's batch interface, implemented independently.
- **Kernel dispatch by chunk size.** The shipped M1 decoder is a single
  warp-per-chunk kernel with no dispatch heuristic, no shared memory, and no
  `cp.async` staging: the design panel settled that shape on the arithmetic
  (section 9), and the asynchronous batch ABI rules out host-side
  histogramming (section 8). A block-per-chunk path (a warp team sharing
  staged input via `cp.async`) and device-side chunk-size binning remain
  deferred options for large-chunk or future-format workloads - not a
  decision the current code took.
- **Two memory paths.** Device-resident (caller already has the data on the
  GPU) and a pinned-host streaming path that overlaps H2D copies with decode -
  the asset-streaming shape.
- **Fail-closed decode contract.** Every value decoded from the bitstream is
  validated before it is used as an address, length, or offset; size
  arithmetic is overflow-checked; a chunk that fails validation reports a
  defined error and produces no partial output presented as success. Hostile
  input is the expected case, not the exception.
- **Termination is part of fail-closed.** A decoder that hangs on hostile
  input has failed open in the availability direction - nvCOMP 5.3's release
  notes record fixing an indefinite Snappy-decompression hang on malformed
  input, so this is a demonstrated bug class in exactly this product
  category, not a theoretical one. Every loop whose exit depends on a value
  read from the bitstream carries an explicit decrementing fuel cap whose
  bound is unreachable for any input the validation ladder admits: the accept
  set is unchanged, and a future guard bug degrades to a defined reject
  instead of a hung device. The parser's per-call liveness contract is
  asserted over truncated, mutated, and crafted streams
  (`tests/termination.cpp`, `tests/termination_gpu.cu`), every ctest entry
  carries a finite `TIMEOUT`, and a configure-time check reds the build on a
  fuel-free loop in the decode path.
- **Warp collective integrity.** Kernels use only the `_sync` warp
  primitives; the legacy non-`_sync` intrinsics are banned outright, because
  since Volta's independent thread scheduling they cannot express which lanes
  participate at all. A collective's participation metadata - its mask, its
  source lane, its predicate - is never derived from a value read out of the
  bitstream: it is the full-warp constant or `__activemask()`, and the loop
  bounds that decide which lanes reach a collective stay lane-uniform.
  Corrupting that metadata is a published attack on CUDA kernels
  ("Gerrymandering the Warp", arXiv:2606.11878) that produces wrong results
  without touching control flow, and the shipped decoder's redundant
  all-lane parse is what keeps every lane's path identical. Configure-time
  checks in `tests/CMakeLists.txt` enforce both halves; `CONTRIBUTING.md`
  carries the four-point review checklist.
- **Determinism.** Same input → bit-identical output on every path: the level
  is `gpu_to_gpu` in NVIDIA's CCCL vocabulary, held by construction rather
  than by tuning - integer-only arithmetic, every output byte written exactly
  once by a statically determined lane as a pure function of lower addresses,
  and no inter-chunk coupling. "Exactly once" holds for a supported launch
  geometry, which the kernel enforces rather than assumes. The scope, the
  reasoning, and the tested axes (including launch geometry and stream count,
  which a same-batch-twice compare cannot see) are in
  [DETERMINISM.md](DETERMINISM.md).
- **The oracles decide correctness.** liblz4, zlib, and libzstd are vendored
  as test dependencies; decode output is diff-tested against them on real
  corpora (Silesia, enwik) and on fuzzed/mutated streams, including the
  negative case: whenever the reference rejects, cudec must reject.
- **Target hardware.** Baseline `sm_80`, tuned on `sm_86` (RTX 3080: 68 SMs,
  10 GB GDDR6X, ~760 GB/s). The honest performance ceiling for LZ4 decode is
  output-bandwidth-bound; benchmarks report achieved GB/s against that
  ceiling, per corpus and chunk-size distribution.

## 5. Testing & benchmark discipline

- Unit + oracle-diff + negative tests run under `ctest`; structural rules
  (fail-closed coverage, C-ABI purity, no-dependency rule) are encoded as
  conformance tests so every PR is checked mechanically. The configure-time
  conformance checks are **drift detectors, not tamper-proofing**: they
  snapshot the library target's link surfaces and flags when `tests/` is
  configured and defend against honest refactoring mistakes; adversarial
  evasion (and additions made after that snapshot) remain code review's
  job.
- **The harness is framework-less; ctest is the runner** (settled in the #4
  design review): one executable per test group over the small assertion
  header `tests/require.h`; discovery, labels, parallelism, and rerun come
  from ctest; buffer diffs report first-mismatch offset plus a hex window -
  the one assertion domain a framework would not improve. Rationale:
  near-zero supply-chain surface for the security net itself, and
  early-abort semantics fit contract-sequence tests. Recorded reassessment
  trigger: if early-abort measurably masks failure clusters at M5 mutant
  scale, or an outside contributor base emerges, migrate to Catch2 (pinned) -
  the REQUIRE-shaped macros survive that move verbatim, and no framework
  headers are compiled through nvcc-only paths that would complicate it.
- **A GPU test never self-skips** (banned pattern): no "no device, exit 0"
  branches - skipping is exclusively the runner's decision via ctest labels
  (`-LE gpu` on the GPU-less CI runner; the full run on the local gate). A
  mislabeled GPU test in CI therefore fails loudly instead of passing
  vacuously; `--no-tests=error` closes the zero-tests-selected route; CI
  prints the deferred `-L gpu -N` listing and pins the known GPU tests by
  name.
- **The GPU sanitizer gate**, required from M1 onward for every PR that
  touches device code: all four Compute Sanitizer tools - memcheck,
  racecheck, initcheck, synccheck - over every `gpu`-labelled ctest target,
  through `scripts/sanitize-gpu.sh`, which fails closed on a sweep that
  covered nothing. CI cannot run it: the free plan has no GPU runners, so no
  workflow reds for a missing sweep and the output pasted into the pull
  request is the whole evidence, carrying the commit, the GPU, the driver and
  the CUDA version it was produced against. **racecheck detects
  shared-memory hazards only** - global-memory races are outside its scope,
  and a clean sweep is never read as clearance for them. Today the gate is
  owed and unproducible on the maintainer route: under WSL2 the device is in
  WDDM mode, the debugger interface the tools need is absent, and all four
  report the same two initialization errors against a program with no fault in
  it, which #258 records with its commands. That is a gap in the evidence, not
  a dispensation from the gate.
- **Oracle pinning policy**: oracles are vendored via FetchContent from
  maintainer-uploaded release assets, pinned by a self-computed SHA-256
  cross-checked against a second packaging ecosystem; auto-generated
  `/archive/` tarballs are avoided (GitHub regenerated them in 2023 and
  their hashes moved). Where a project publishes no uploaded asset, the
  archive tarball is pinned and a future hash mismatch is treated as the
  invariant working, not as noise. FetchContent pins are invisible to
  Dependabot - oracle bumps are deliberate manual PRs owned by the
  supply-chain sweep. Third-party oracle code compiles with SYSTEM includes
  and without the project's strict flags; only the translation units the
  oracle's decode path needs, never its build system. For liblz4 that is one
  file; the Snappy oracle needs two, because the source abstraction its
  exact-consume decode reads through lives in a second one; the GDeflate
  oracle needs six, because it is the only one that also compresses.
- **The GDeflate oracle deviates from that policy in three named ways**, and
  they are deliberate rather than oversights. The NVIDIA fork of libdeflate
  publishes no release asset and no tag at all, so the pin is a commit
  archive - one step further from the preferred shape than snappy's tag
  archive. No second packaging ecosystem carries the fork, so the
  cross-check is the commit sha together with the archive hash rather than a
  second party's hash. And the fork sits far behind upstream libdeflate,
  including whatever correctness fixes those commits carry, which is
  acceptable in a thing whose job is to be where real data came from and is
  the reason it is never a source to derive from. It is under the manual-bump
  rule above like every other FetchContent pin: Dependabot cannot see it, and
  it moves only in a deliberate pull request under the supply-chain sweep.
  The pin, the deviations and the commands that re-derive them are in
  `cmake/CudecGDeflateOracle.cmake`, next to the pin they are about;
  section 11.7 is where they were decided.
- Fuzzing: mutation-based corpus fuzzing of the host-side parsers, plus
  GPU-vs-oracle diff loops on mutated streams for the kernels.
- Benchmarks live in `bench/` with recorded methodology; every published
  number carries GPU model, driver, CUDA version, corpus, chunk-size
  distribution, and run count. Baselines are recorded in
  `docs/BENCHMARKS.md`; regressions against them block merges unless
  explicitly justified.

## 6. Milestones

| Milestone        | Deliverable                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| M0 - Foundation  | Toolchain decision + GPU probe, CMake+CUDA skeleton, CI gate, test harness + LZ4 oracle, bench skeleton, masterplan complete |
| M1 - LZ4 block   | Warp-cooperative LZ4 block decode, fuzz-diffed against liblz4, first honest numbers                                          |
| M2 - LZ4 batch   | Frame format, batch API, pinned-host streaming path, published benchmark + methodology write-up                              |
| M3 - Snappy      | Snappy decode on the shared kernel family                                                                                    |
| M4 - GDeflate    | GDeflate decode as an auditable CUDA library (32-substream warp decode)                                                      |
| M5 - Zstd        | Zstd decode: FSE/Huffman stages + sequence execution                                                                         |
| M6 - Portability | HIP port of the kernel family                                                                                                |

## 7. Risks, named

- **Scope discipline between M2 and M4.** The value is format coverage; the
  temptation is endless LZ4 tuning. The milestone gates exist to force the
  ladder.
- **Zstd complexity.** M5 is months, not weeks; the README does not promise
  it until M4 has shipped.
- **Single-machine development.** CI has no GPU; kernel tests run on the
  maintainer's hardware. The CI gate covers build + host-side tests; GPU
  test results are recorded in the PR before merge.

## 8. Open design questions (settled via design issues before use)

The M2 list is empty. Both questions it held are resolved below, one retired
and one settled.

**Device-side chunk-size binning for mixed batches - RETIRED (issue #78,
2026-08-08), with the trigger that reopens it.** The async batch ABI rules out
host-side histogramming, so the block-per-chunk / binning option in section 4
resolved to device-side binning or nothing, and it is nothing.

What binning would buy is a second dispatch shape: a block per large chunk
beside the warp per chunk the kernel ships. The evidence against paying for one
is that nothing has ever measured the dispatch losing. Every recorded
distribution is effectively uniform 64 KiB:

    grep -n "chunk sizes:" docs/BENCHMARKS.md

Six of the seven recorded corpora report median 65536 and max 65536, the
minimum being one short trailing chunk (8066, 57600, or 65536 itself). The
seventh is the 12-chunk, 0.21 MB builtin corpus the harness self-checks on,
which is a correctness fixture and not a throughput measurement. So no recorded
run has put a mixed chunk-size distribution on the device at all, and the case
for binning rests on a shape none of the numbers describes.

Against that, perf passes 1-3 (issues #16, #21, #36, all in section 9) put the
kernel at a structural local optimum and named the bottleneck: the redundant
32-lane parse ceiling and latency-bound short-run copies. Dispatch is not on
that list. Binning would add a device-side counting pass, a second kernel
shape, and a second review surface on the sensitive kernel path, for a gain no
measurement has located - which is what "formats over percentage points" ranks
below the next format.

**Falsification trigger.** Reopened when a recorded benchmark shows the
dispatch losing on a mixed distribution: the same bytes measured at a mixed
chunk-size distribution and at a uniform one, in the same container on the same
GPU, with the mixed rung slower by more than the run-to-run spread both rungs
report. Both numbers go in [docs/BENCHMARKS.md](BENCHMARKS.md) with the full
methodology block before anything is designed. The asset-like benchmark corpus
settled below is the natural source of such a distribution, so that
measurement is the cheapest route to reopening this - and until it exists,
this stays retired.

**The benchmark corpus set beyond Silesia/enwik - SETTLED (issue #139,
2026-08-09).** The game-asset-like corpus is GENERATED, deterministically in
the bench harness, rather than fetched from a redistributable asset pack. No
network, no mirror that can rot, no licence review on a third-party package,
and the pattern already stands twice in the tree: `BuildWorst4bCorpus` and
`BuildLongmatchCorpus` construct their block directly, round-trip it through
`LZ4_decompress_safe` before any timing, and assert the intended shape. The
generator is `--assetlike`, locked against rot by the CPU-only
`bench_assetlike_selfcheck` ctest.

The price is in the documentation rather than the small print: a generated
corpus is a MODEL of the workload and not the workload. A synthetic mixture of
texture-, mesh- and audio-shaped regions reproduces the regime - longer
matches, a higher incompressible share, a different sequence density - and is
expressly not a substitute for a measurement on real game data. That sentence
sits beside the corpus entry in
[BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md), and the baseline block
in [BENCHMARKS.md](BENCHMARKS.md) carries it too, so anyone quoting a number
from the corpus sees what it was measured against.

A hash-pinned asset pack would have been more representative, and for
benchmarking that is what counts - the argument against the decision at full
strength. It would have cost a licence review, two mirrors and a manifest, and
put CI on the network; building both would have created two regimes to keep
apart at citation time. The option stays open: a pack with checked terms joins
the set BESIDE the generator rather than replacing it, and CI stays offline.

Settled: test framework and oracle vendoring (section 5, via the #4 design
review); dev-container image and CI toolchain pinning (issues #1/#3 -
digest-pinned `nvidia/cuda` 12.6.2 in CI and the local gate); the LZ4
kernel decomposition (section 9, via the #6 design panel - single-pass
warp-per-chunk, with a recorded measurement-based falsification trigger;
the exact batch upper bound is pinned by the zero-visible-devices
technique from the #4 harness once the geometry lands:
`CUDEC_ERR_CUDA` = passed validation and reached the launch,
`CUDEC_ERR_INVALID_ARGUMENT` = rejected, no constant published); the
Snappy raw-format decode and the seam it rides (section 10, via the #85
design panel - parser contract, validation ladder, batch entry, and the
framing format ruled out of scope with its evidence); the benchmark corpus
set beyond Silesia/enwik (issue #139 - a deterministic in-harness generator
for the game-asset-like regime rather than a redistributable pack, with the
representativeness it gives up stated wherever its numbers are, argued in full
above).

## 9. M1 kernel design (settled via the #6 design panel, 2026-07-17)

Three candidate designs (single-pass warp-cooperative, two-phase
scan-then-copy, and a simplest-that-saturates challenger) were developed
independently and scored by two judges with independent bandwidth and
occupancy arithmetic. The convergent result:

**One kernel. One warp per chunk over a grid-stride loop. All 32 lanes
parse the sequence stream redundantly in lockstep** (identical bytes,
identical arithmetic, identical registers in every lane - no leader lane,
no shuffles, no parse divergence), **and fan out by lane for every copy.
No shared memory, no sequence table, no dispatch heuristic.**

Why the arithmetic forces this shape:

- The kernel is parse-bound, not bandwidth-bound. The Silesia-shaped batch
  moves ~450–550 MB per run (src read + dst write + match-gather misses
  against 5 MB L2) - a ~300–380 GB/s bandwidth ceiling - while the serial
  per-chunk parse chain (3–5 dependent L1 loads per sequence, ~3,300
  sequences per 64 KiB chunk) bounds a naive kernel to ~100–200 GB/s.
  Parallelism therefore comes from chunks: 3,239 Silesia chunks against
  3,264 resident warps on the RTX 3080. Warp-per-chunk exposes 32
  concurrent parse streams per SM; a block-per-chunk two-phase design
  exposes 4, starving the binding resource to accelerate copies that were
  never the bottleneck.
- Table-in-smem two-phase is dead on arithmetic alone, recorded here so
  nobody re-proposes it: a 64 KiB chunk admits up to 16,385 sequences;
  at 16 B per table entry that is ~256 KiB - 2.6× an SM's usable shared
  memory.

**Overlap copy, closed form.** An overlapping match is not a copy that
chases itself; it is a modular gather from bytes already final:
`dst[d + i] = dst[d - off + (i mod off)]` reads only `[d - off, d)`, which
the `__syncwarp()` preceding every copy has frozen. Each output byte is
written exactly once, by a statically determined lane, as a pure function
of lower addresses - deterministic because no ordering exists to get
wrong. "Exactly once" holds for a supported launch geometry, which the
kernel enforces rather than assumes: the copy loops stride by the warp
size, so a block that is not a whole number of warps would leave a fixed
slice of every destination written by nobody, and that geometry returns
without decoding ([DETERMINISM.md](DETERMINISM.md)). The shipped inner loop computes that `i mod off` directly - one
64-bit modulo per output byte, so every lane pays a division per
iteration (`dst[seq.match_dst + i] = dst[seq.match_src + (i % offset)]`
in `src/lz4_decode.cuh`). Cutting that cost is deferred, measurement-gated
perf work, not a present property: eliding the modulo when the match
cannot wrap (`off >= len`, #36) and narrowing it to 32-bit (#58), neither
merged. The fully incremental `r += step; if (r >= off) r -= off` scheme
was itself tried and rejected by measurement - see perf pass 1 below.

**The validation ladder** (fail-closed; every stream-decoded value checked
before first use as address, length, or offset): token existence before
every token read; length accumulation in 64-bit with the capacity bound
re-checked inside the 255-continuation loop (each step adds ≤ 255 after
the check, so the accumulator cannot wrap for any caller-supplied sizes);
literal presence vs. remaining src; literal fit vs. remaining dst
capacity; offset-field presence; `offset == 0` rejected;
`offset > bytes written so far` rejected; match-length fit vs. remaining
capacity; the LZ4 parsing restriction that a match may not end within the
last `LASTLITERALS` (5) output bytes; success ONLY at exact stream
consumption after a literals-only tail. Edge semantics (end-of-block rules, empty-block tokens) are settled
empirically by oracle parity - whenever liblz4 rejects, cudec rejects; for
accepted mutants the comparison is against the oracle's own output and
size. One crafted negative fixture per ladder branch, its oracle verdict
asserted in-test, keeps every reject path CI-covered.

**Failure contract:** on any reject, `bytes_written = 0` and a non-OK
status; dst contents up to the failure point are unspecified but are never
presented as success. On success, writes touch exactly
`dst[0, bytes_written)`. Check-before-load is batch isolation, not just
parity: on a GPU an out-of-bounds read is not a per-chunk failure - it can
fault the launch and poison the whole batch.

**One deliberate divergence from liblz4 (settled empirically at ladder
step 2):** a match offset of 0 is invalid per the LZ4 block spec, but
liblz4 tolerates it as a defined self-referential copy (its decoder
silences an msan warning there rather than erroring). cudec rejects it -
fail-closed on spec-invalid input (prime directive 1) outranks bug-for-bug
parity with the reference. This makes cudec's accept set a strict subset
of liblz4's: the twin test asserts the two security-critical directions
(where cudec accepts, liblz4 accepts and the bytes match; where liblz4
rejects, cudec rejects) and reports the count of stricter-than-liblz4
cases so the divergence stays visible rather than silent. It is the only
such point.

**Anti-pattern rule (from the two-phase candidate's disproof):** no packed
or narrowed field anywhere in the decoder unless its bound derives from an
ABI-enforced invariant - the 64 KiB chunk cap is a project convention, not
an ABI guarantee, and the frozen `size_t` capacities admit larger values.
A crafted test (valid stream, `dst_capacity > 65536`) pins that the ladder
stays correct beyond the convention.

**Mechanical gates for every kernel PR** (determinism is enforced, not
argued): compute-sanitizer memcheck + racecheck clean; same-batch-twice
bit-compare; whole-Silesia GPU-vs-oracle byte diff; the armed mutant
reject-parity; the capacity-beyond-convention test. Numbers for an
unverified decoder are not numbers.

**Occupancy plan:** target ≤ 64 registers/thread for ≥ 32 warps/SM,
pinned with `__launch_bounds__`; achieved registers and occupancy are read
out at the kernel PR and recorded - a 24-warp fallback is a measured
choice, never an accident.

**Measurement decision rule (recorded before the kernel lands):** the
bench harness gains `PARSE_ONLY` and `COPY_ONLY` compile variants. The
parse-only number simultaneously ceilings this design and any two-phase
phase-1 (identical serial ladder, no more concurrent chains), so one
measurement settles the decomposition question: proceed if parse-only
projects ≥ ~10× the CPU p50 baseline (≥ ~35 GB/s); reopen two-phase only
if the shipped kernel measures below ~15× CPU after the first perf pass or
profiling attributes the majority of stalls to copy starvation.

**Measured outcome (issue #15, 2026-07-17): SETTLED for single-pass.** On
Silesia the minimal-correct kernel decodes at ~18 GB/s (~5× CPU) and the
parse-only ceiling is ~35 GB/s (~10× CPU) - so parse and copy each cost
roughly half the wall time. Since parse-only ceilings any two-phase phase-1
(shared serial parse), two-phase cannot exceed ~35 GB/s; its only lever is
a faster copy, which single-pass optimizes equally without a table or
barrier. Recorded in [docs/BENCHMARKS.md](BENCHMARKS.md).

**Perf pass 1 outcome (issue #16): the designed micro-optimizations were
rejected by measurement** - the incremental-mod gather regressed 16% (its
loop-carried dependency pipelines worse than the independent modulo the
compiler overlaps) and the vectorized literal copy regressed 6% (real
literal runs are too short for the wide path); a `__syncwarp` elision was
neutral. The kernel is at a local optimum for its structure; the bottleneck
is structural (the redundant 32-lane parse ceiling and latency-bound
short-run copies), so raising throughput needs higher occupancy or
warp-specialization - larger than a micro-op pass, tracked as issue #21. The
falsification trigger does NOT reopen two-phase: it stays ruled out by the
shared parse ceiling regardless of the numeric condition (see
BENCHMARKS.md).

**Perf pass 2 outcome (issue #21): the occupancy lever is rejected by
measurement too.** Profiling confirmed 46 registers/thread → 40 warps/SM
(~83%) on sm_86; register granularity puts 100% occupancy behind ≤ 40
registers/thread, reachable only by narrowing the 64-bit parser state (barred
by the anti-pattern rule above) or by forcing a spill. `__launch_bounds__`
with a min-blocks target of 12 forces the spill and regresses the full decode
~5% (17.4 → 16.6 GB/s), parse-only flat as the control; no code shipped.
Raising the parse ceiling needs the warp-specialization rewrite that abandons
the redundant-lockstep-parse invariant (its own design panel), which under
"formats over percentage points" ranks below the next format. Recorded in
[docs/BENCHMARKS.md](BENCHMARKS.md).

**Perf pass 3 outcome (issue #36): the non-overlap match fast path is
rejected by the worst case.** Splitting the match copy on the warp-uniform
`offset >= match_len` predicate (straight copy instead of the modular gather
when the match cannot wrap - bit-identical by construction, all oracle/
determinism gates green) measured +9–10% on Silesia and +38–42% on the new
copy-dominated `--longmatch` corpus (throughput speedup), but a consistent
−5–9% on `--worst4b`:
the per-match predicate lands in the hottest path of exactly the
maximum-sequence-density adversarial input. Rejected under the
pre-registered zero-regression rule - the worst-case number is the
DoS-resistance margin, and average-case gains do not buy it back. No kernel
code shipped; the `--longmatch` corpus and its selfcheck ctest shipped so
the regime stays measurable. Recorded in
[docs/BENCHMARKS.md](BENCHMARKS.md).

**Known limits, published:** the redundant-parse family ceiling is roughly
250–400 GB/s after perf passes - deliberately accepted under "formats over
percentage points"; batches under ~2,000 chunks underfill the machine and
land near CPU speed (documented, not hidden); warp-synchronous discipline
is load-bearing - every `__syncwarp` is reviewed as such.

**M3/M4 seam:** the chunk decoder is to become
`template<class Parser, bool ParseOnly>` - Snappy (M3) swaps the parser and
keeps everything else; GDeflate (M4) keeps the copy engine, validation
posture, and result contract while bringing its format-native 32-substream
parse model. Written as a commitment rather than as a present property,
because it is one: `src/lz4_decode.cuh` is `template <bool ParseOnly>` with
`Lz4Parser` named directly in the kernel body, and no `Parser` template
parameter exists under `src/` today. Section 10 fixes the contract the seam
must carry; the M3 kernel rung materializes it.

**The M1 PR ladder** (each independently gated): (1) this design section -
closes the design issue; (2) the sequence parser + validation ladder as a
single-source `__host__ __device__` header with a CPU-compiled twin test
running the full mutant corpus and the crafted negatives on the GPU-less
CI runner - the security heart of M1 lands under CI before any kernel;
(3) the kernel, minimal-correct, behind the frozen batch ABI - the stub
deleted, the gpu_fixture expectations flip, all mechanical gates recorded
(security-review gated: decoder validation is this project's login path);
(4) the bench GPU path + the split variants + the first recorded GPU
baselines - numbers and kernel never move in the same PR; (5) a measured
perf pass (register-window parse staging, vectorized multi-byte lane
copies), accepted only on recorded improvement with all gates green.

## 10. M3 Snappy decode design (settled via the #85 design panel, 2026-07-27)

Section 9 already fixed the kernel family's shape - single-pass,
warp-per-chunk, redundant 32-lane lockstep parse, closed-form modular
gather, the whole validation ladder living inside the parser - and Snappy
inherits every one of those decisions unchanged. Nothing below reopens
them. What M3 had to settle is narrower: what the format seam actually
promises, what the Snappy parser and its ladder are, what the batch entry
owes, whether the framing format is in scope, and which locks keep all of
that from drifting.

The format ground truth is the google/snappy 1.2.2 sources, not
`format_description.txt`, which says of itself that it is not a formal
specification. The reference implementation is the de-facto authority, and
the readings this design leans on are executed rather than argued, by two
routes: `tests/snappy_probes.cpp` drives hand-built streams straight at the
oracle for the accept-set behaviours a compressor never emits, and every
crafted negative in the twin asserts the reference's verdict on the same
bytes as well as cudec's, so a reject branch is demonstrably in parity
rather than merely present.

**Where this section is a commitment and where it is a description.** The
parser core has landed; the seam and the batch entry have not. Each
subsection below says which it is, so the document cannot be read as a
report of shipped code. The tree is the authority for the second kind and
the commands that read it are given inline.

### The seam: what a parser owes the chunk decoder

**A commitment.** The chunk decoder becomes
`template <class Parser, bool ParseOnly>` in `src/chunk_decode.cuh`: the
loop body, the fuel cap, the two copy loops, the `__syncwarp()` placement,
the geometry guards and the argument validation move across unchanged,
because that code already IS the template modulo one hard-wired type.
`src/batch.cu` then instantiates it once per format and the bench keeps its
parse-only ceiling instantiation. LZ4 pays a rename and nothing else.

What a `Parser` owes, and what the kernel is entitled to assume:

- construction from the chunk's source pointer, source size and
  destination capacity, and nothing else;
- one entry point returning exactly three outcomes: an element to execute
  and call again, the single success exit, or a reject status;
- **liveness** - every call consumes at least one source byte. This is what
  keeps the kernel's existing source-size-derived fuel bound valid for
  every instantiation, so it is a seam requirement rather than an LZ4
  accident, and a parser that can return without advancing breaks
  termination for the whole family;
- **lane-uniformity** - the parser reads source bytes and its own state and
  nothing else, which is what makes the redundant lockstep parse sound;
- **all input validation inside the parser.** The copy engine performs no
  input-derived bound check of its own. A parser that hands back an element
  it has not fully bounded has already failed closed-ness, and no later
  stage will catch it.

Rejected at the panel and recorded so it is not re-proposed: a per-parser
associated element type. It invites exactly the divergence the copy
engine's single contract exists to prevent, for no gain.

The seam serves the byte-oriented family (LZ4, Snappy). M4 brings a
format-native 32-substream parse model and revisits the lock deliberately;
that is the lock working rather than an obstacle.

### The Snappy parser and its validation ladder

**Landed** as `src/snappy_block.h`, single-sourced `__host__ __device__`,
twin-tested host-side on the GPU-less runner before any kernel.

The preamble is a little-endian base-128 varint of the declared
uncompressed length, at most five groups, with the fifth group refused at
or above 16 rather than truncated - a decoder that masked the excess away
would accept a declaration the reference rejects outright. Then, before a
single element is parsed, **the declared length is checked against the
caller's destination capacity**. The varint is attacker-controlled and the
capacity is the caller's truth; nothing anywhere is ever sized or allocated
from the declaration.

An element is a literal OR a copy, not a literal run followed by a match,
so one range and a kind describe it. Lengths never continue across an
unbounded chain of extension bytes: an element's header is the tag plus at
most four trailing bytes, which is the reference's `kMaximumTagLength`, so
the per-element termination argument needs no fuel counter at all - a real
simplification over LZ4's 255-continuation accumulator, and the reason the
Snappy parse chain is structurally shorter. The header-window check is made
against the **actual** width the tag implies, not against a blanket
five-byte lookahead: a stream ending in a one-byte literal is valid, and a
blanket lookahead would reject it and hand the twin an oracle-diff failure.

Success is reported only at exact source consumption AND exact production
of the declared length, which is what the reference enforces on both ends.
Under-production, over-production and trailing bytes all reject.

**The status mapping, decided.** `CUDEC_ERR_OUTPUT_TOO_SMALL` fires in
exactly one place: the declared length against the caller's capacity, at
the preamble, before anything is written. Every later bound is checked
against the _declared remaining output_, and busting it is
`CUDEC_ERR_CORRUPT_INPUT` - once a stream has declared its length, an
element that overruns it is a stream inconsistency a larger buffer would
not make valid. LZ4 has no declared length, which is why its ladder maps
length-versus-capacity to `OUTPUT_TOO_SMALL` instead; the asymmetry between
the two ladders is a decision rather than an inconsistency.

**The accept set is as load-bearing as the reject set.** Over-strictness is
an oracle-diff failure, so the parser accepts what the compressor never
emits but the reference decoder takes: the four-byte-offset copy form, copy
lengths below four, offsets at or above 65536 up to the bytes produced,
non-minimal varints, and non-minimal literal-length encodings.

**The panel expected no stricter-than-reference divergence for Snappy. One
was found in the building, and it is recorded here rather than absorbed.**
The panel's reasoning was that LZ4's tolerated zero offset has no Snappy
analogue, which holds: the reference refuses a zero offset in all three
copy forms, so cudec refusing it is parity and not strictness. The
divergence is elsewhere. A literal's length is encoded as length minus one,
and the reference performs that addition in 32 bits, so the four-byte
length class spelling `0xFFFFFFFF` wraps to a zero-length literal and the
reference accepts the stream. cudec refuses it, on the same ground that
settled LZ4's offset zero: a length that exists only because an accumulator
wrapped is not a length, and fail-closed on spec-nonsense outranks
bug-for-bug parity. That makes cudec's Snappy accept set a strict subset of
the reference's at exactly one point, and the twin pins the divergence
explicitly - the oracle's acceptance and cudec's refusal both asserted on
the same bytes, on an otherwise empty stream and on one carrying a real
literal behind it - so it stays visible instead of hiding inside a parity
count. The aggregate count over the mutation corpus is separately pinned at
zero, which is what makes the one named exception readable as the whole of
it.

The enumeration in `src/snappy_block.h` is the authority for which reject
branches exist; this document does not restate the list, because a restated
list drifts against the header. The panel sketched twelve branches and the
shipped ladder carries thirteen, and the two sets do not correspond one to
one: the implementation splits some of the panel's rungs, folds others, and
adds ones the sketch had no place for, including the varint loop's
compiler-required exit, which is argued unreachable and declared so in the
twin rather than carried as a negative. The header, not the sketch and not
this paragraph, is where the current set is read. Count it rather than
trusting this sentence:

    sed -n '/^enum SnappyReject {/,/^};/p' src/snappy_block.h |
        grep '    kSnappyReject' | grep -vc 'None\|Count'
    13

**One departure from the panel record, with its reason.** The panel put the
preamble inside the element call and rejected a separate initialization
method, on the grounds that it widens the contract for one format's need
and gives LZ4 a dead call site. The landed parser exposes an idempotent
`Begin()` and has the element call go through it. The seam requirement it
was rejected to protect is unaffected - `Begin()` is not part of the seam,
the element call still consumes at least one byte on every invocation, and
the three outcomes are unchanged - and it buys the batch entry the declared
length without parsing the varint twice. The panel's objection to a _seam_
method stands; this is a parser-local one.

**Not yet executed:** the panel also called for promoting the LZ4 sequence
type to one shared element vocabulary in `src/decode_sequence.h`. The
parser core kept a format-local element type instead, so there are two
today. Materializing the seam is where that is either executed or overturned
with a stated reason; it cannot be left implicit, because one shared element
vocabulary is what the "copy engine checks nothing" rule rests on.

### The batch entry

**A commitment.** `cudec_snappy_decompress_batch` carries the
`cudec_lz4_decompress_batch` contract with no deltas: device-side argument
arrays, per-chunk results, synchronous argument-validation refusal through
the shared validator with the identical geometry refusal, asynchronous
launch on the caller's stream, the same pending-CUDA-error semantics,
`bytes_written` zero on any reject, and writes touching exactly the bytes
reported.

The open question was whether the declared-length preamble changes the
per-chunk result semantics. **It does not.** Capacity stays the caller's
bound, the declaration is validated against it, and no new status value is
needed, so the ABI stays frozen. The public header will document the
supported surface as raw Snappy streams; M3 ships the batch entry only, and
generalizing the streaming path to Snappy was a separate scope decision, filed
rather than smuggled in here and now settled below.

Today `include/cudec.h` declares no Snappy entry (`grep -c snappy
include/cudec.h` reports 0). The batch-entry rung carries it.

### The framing format: out of scope, with the evidence

The official Snappy framing format is deliberately not implemented, and the
reason is a consumer survey rather than a preference: no major data-plane
user runs it. LevelDB frames raw per block, Hadoop has its own container,
Kafka uses a third format from xerial, and Parquet stores raw pages. If it
ever comes back it is a host-side chunk walker plus a masked CRC-32C in
front of the same raw batch entry, never kernel work. Recorded here so the
question stays settled instead of being re-argued each time the format is
named.

### The streaming entry: deferred, with the trigger

The scope decision this design filed rather than smuggled in (issue #120) is
settled the same way the framing format above is: **cudec does not offer a
Snappy streaming entry, and no milestone carries one.** Not "not in M3" - not
scheduled at all, until something below fires.

The reason is that the demand is not there and the cost is not zero. The
streaming path exists for LZ4 because the frame format is a stream of blocks a
consumer meets incrementally. Raw Snappy is what the data-plane consumers
actually store - LevelDB frames raw per block, Parquet stores raw pages - and a
consumer holding a raw page holds all of it. The framing format, which is the
shape that would make an incremental entry mean something, is out of scope
above on its own evidence. Building a resumable Snappy context now would mean
carrying a second streaming state machine, its own geometry refusals and its
own device gate set, for a caller nobody has named.

Nothing in the parser blocks it. The seam contract is already incremental -
`SnappyParser::Next` with the lazy preamble - so this is a decision about what
to build, not about what is possible.

**Trigger.** Reopened by a named consumer with a stream it cannot hold whole:
an issue that says which system produces the stream, why the raw batch entry
does not serve it, and what the chunking looks like. A request for symmetry
with the LZ4 streaming context is not that trigger, because symmetry is the
argument this paragraph is refusing.

### The locks that keep this from drifting

- **The decode-seam lock** (a commitment, landing with the seam): each
  parser type defined in exactly one header, the shared element type in
  exactly one header, and the kernel entry point existing in exactly one
  place. Same honest scope as the existing RAII lock - a drift detector
  under the known spellings, not tamper-proofing. Its single-entry-point
  clause is revisited as a reviewed edit when M4's parse model arrives.
- **The ladder-coverage lock** (landed with the parser core): every reject
  branch is an enumerator, every enumerator is reached by a crafted
  negative, and `tests/snappy_parser_twin.cpp` refuses both directions - a
  branch no negative reaches reds the suite, and a negative naming a branch
  that no longer exists reds it too. Adding a reject path without a test is
  therefore not a thing that can be done quietly.

### The M3 rung ladder, confirmed

Design (this section, decisions only, no code); the oracle and its
empirical probes; the parser core and its twin; the kernel, which
materializes the seam first and instantiates Snappy second, plus the batch
entry and the device gate set; the bench rung; the measured perf pass. The
two follow-ups this design calls for were filed at the time: the bench rung
with its two density-locked worst-case corpora, and the Snappy
streaming-entry scope decision. Numbers and kernel never move in the same
PR, and the perf pass inherits section 9's pre-registered zero-regression
rule on the worst case from its first baseline.

**No open maintainer decision.** Every fork above was settled at the panel;
none of them is a call that needs the release gate.

## 11. M4 GDeflate format dossier

What GDeflate is, sourced, so that the kernel design panel and the
table/parse core work from one record instead of each re-reading a draft.
Facts and provenance only. No design decision is taken below, and the legal
posture is settled in its own maintainer-gated issue.

### 11.1 Provenance: three artifacts, no single specification

There is no canonical GDeflate specification. Three artifacts together
stand in for one, and they do not carry equal weight.

**The IETF draft.** `draft-uralsky-gdeflate-00`, "GDEFLATE bitstream
specification", Y. Uralsky, NVIDIA Corporation, 7 July 2024, intended
status Informational.
<https://www.ietf.org/archive/id/draft-uralsky-gdeflate-00.txt>. The
datatracker record
(<https://datatracker.ietf.org/doc/draft-uralsky-gdeflate/>) reports it as
an expired individual Internet-Draft with no stream and no working group,
one version only, last updated 2025-01-08. It is the most precise prose
anywhere and it is not a standard, was never adopted, and will not be
revised unless somebody refiles it. Archived and citable; BCP 78 and the
IETF Trust Legal Provisions apply, so it is implementable freely with
standard attribution.

**The reference implementation**, and the de-facto normative artifact:
`GDeflate/` in `microsoft/DirectStorage`
(<https://github.com/microsoft/DirectStorage>). Two licence facts that are
easy to get backwards, both read from the tree rather than from the
repository's landing page: the repository root is MIT, and the GDeflate
subtree carries its own Apache-2.0 `LICENSE` with SPDX headers naming both
NVIDIA CORPORATION & AFFILIATES and Microsoft Corporation. Apache-2.0 is
what the M4 oracle work has to satisfy, and it is the subtree's licence,
not the repository's. Beside the codec sits `GDeflateTest`, which validates
against the shipping runtime, so bit-exact interop with DirectStorage is
the contract that is actually tested. **Where the draft and the reference
disagree, cudec follows the reference**, because the reference is what real
data was produced by.

**The Vulkan extension**, which exposes the format without defining it.
`VK_EXT_memory_decompression` is extension 551 in the registry, marked
`ratified="vulkan"`, promoted from `VK_NV_memory_decompression`
(extension 428). Its method bitmask carries exactly one method bit,
`VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT`. Its valid usage is
where the 64 KiB decode unit shows up independently of the draft: when the
method is that bit, `decompressedSize` must be at most 65536 bytes
(`VUID-VkDecompressMemoryInfoEXT-decompressionMethod-11762`, and the same
bound again on the region structure). Khronos defines none of the
bitstream; it standardizes a way to ask a driver for it. Read from
`xml/vk.xml` and `chapters/memory_decompression.adoc` in
`KhronosGroup/Vulkan-Docs`.

### 11.2 The decode unit and the substream machine

Every fact in this subsection is the draft's, with its section number.

- **Pages.** Data is split into 64 KB pages that are "completely
  independent, enabling compression and decompression to operate on
  multiple pages in parallel" (section 4). The page is the unit of random
  access and the unit of parallelism.
- **Substreams.** The sequence of variable-length codes is split into 32
  independent substreams, "each sub-stream is assigned to a fixed SIMD
  lane" (section 5.1). Thirty-two is not a tuning parameter; it is in the
  bitstream.
- **The bit buffer.** The draft expects per-lane state in a pair of 32-bit
  registers plus a counter of valid bits, and sets the read threshold at 32
  bits so at least one symbol per lane is decodable every round (section
  5.2). The threshold is sized by the longest possible code, which the
  draft states as 31 bits: 15 bits of worst-case Huffman code plus at most
  16 extra bits.
- **The refill schedule is the interleaving.** The first round always loads
  32 consecutive words, which initializes all 32 lane states, "as a result,
  any valid GDeflate bit stream cannot be smaller than 4 \* 32 = 128 bytes"
  (section 5.3). Later rounds load a variable number of words depending on
  how fast each lane consumed. The stream is then "words from the 32
  sub-stream interleaved in the order in which they are read during
  decoding", and that reordering "applies end-to-end across all blocks in
  the bit stream, disregarding block boundaries" (section 5.3).

  **This is the load-bearing consequence for a decoder, and it is worth
  stating as a threat rather than as a mechanism: there are no substream
  markers, and substream boundaries are not self-delimiting.** Where the
  next word belongs is derived entirely from the decode that has happened
  so far. A decoder that mis-decodes one symbol does not resynchronize; it
  reads a different word next, for every lane, for the rest of the page. So
  there is no cheap structural check that a walk stayed on the rails, and
  every bound has to be enforced at the point of use.

- **Block headers.** "All block headers are always assigned to sub-stream 0
  and require one dedicated SIMD round of processing per header, where only
  1 SIMD lane is active"; non-compressed and dynamic-Huffman blocks take
  two rounds, one generic and one block-specific (section 6).
- **Copies.** A copy is a length symbol followed by a distance symbol, so
  it is "always split across two SIMD rounds", and the two symbols "are
  always assigned to the same sub-stream so that they get processed by the
  same SIMD lane during two back-to-back SIMD rounds" (section 11).
- **End of block.** Code 256 is monitored per round, and when a lane
  decodes it, "all SIMD lanes following the lane that has decoded the
  end-of-block symbol should not attempt to perform symbol decode and
  consume any bits from their state variables" (section 10). A decoder that
  lets a later lane consume bits there has silently desynchronized every
  subsequent round.
- **Table description.** The code-length alphabet is up to 19 three-bit
  values, one per substream, decoded in a single round with `HCLEN + 4`
  lanes active (section 8.1). The literal/length and distance code lengths
  follow as up to 318 symbols (up to 286 plus up to 32), "evenly
  distributed across 32 sub-streams" and needing up to 10 rounds, each
  round taking a group of 32 consecutive symbols (section 8.2). Static
  blocks have no block-specific header and use RFC 1951's predefined tables
  with the same distribution (section 9).

### 11.3 Divergences from RFC 1951, enumerated

Each item is testable on its own and is numbered so the table and parse
core can pin them one at a time.

**D1. Length code 285 is not a fixed 258.** It is "re-purposed to represent
an LZ copy of length in the range between 3 and 65538 bytes", with 16 extra
bits following the code (section 2.1).

**D2. Distance codes 30 and 31 exist.** Previously unused, they now
"represent look back distances of up to 65536 bytes", each followed by 14
extra bits, covering 32769 to 49152 and 49153 to 65536 respectively
(section 2.1). The reachable window is therefore 65536 bytes, exactly the
page size, which is what makes pages self-contained.

**D3. Non-compressed blocks lose the length complement and the byte
alignment.** "In GDeflate, the one's complement of length is dropped from
the header and the header is no longer required to be byte-aligned. As a
result, data across all blocks, compressed or non-compressed, forms one
contiguous bit stream" (section 2.2). Dropping the complement removes
RFC 1951's one redundancy check on a stored block's length, so the declared
length reaches the decoder unchecked by the format.

**D4. A non-compressed block's body is entropy-shaped anyway.** Its bytes
are distributed across the 32 substreams as "fixed-size 8 bit atoms", each
lane writing one byte and consuming 8 bits per round, so a round moves 32
bytes and consumes eight 32-bit words; the block takes
`floor(block_size_in_bytes / 32)` full rounds and a final round with
`block_size_in_bytes % 32` lanes active (section 7). There is no memcpy
path.

**D5. Block headers are lane-serial, not stream-serial.** Per D3 the
bitstream is contiguous, and per section 6 the header still rides substream
0 alone. The header is therefore not findable by scanning; it is reached
only by decoding to it.

**D6. One table set per block, read by all 32 lanes.** Sections 8.1 and 8.2
describe one code-length alphabet and one literal/length-plus-distance
description per dynamic block, and section 9 the shared predefined tables
for static blocks. Every lane in the group decodes against the same tables.

### 11.4 What the format does not carry

- **No uncompressed size.** The page bitstream does not state how much
  output it produces. Enveloping is external, whether an array of sizes or
  the DirectStorage TileStream container, and the 65536 ceiling comes from
  the consumer rather than from the bits.
- **No checksum, anywhere.** The draft's security considerations say only
  that "any corruption of the data is likely to have severe effects and be
  difficult to correct" and recommend that systems "provide some means of
  validating the integrity of the compressed data" (section 13). **Every M4
  guarantee is therefore structural.** There is no CRC backstop of the kind
  the LZ4 frame's xxHash32 provides, so a bitstream that decodes without
  tripping a bound is accepted, and the bounds are the entire defence.
- **No self-delimiting substreams**, per 11.2. Repeated here because it is
  the same class of absence: nothing in the encoding lets a decoder confirm
  it is where it thinks it is.

### 11.5 What the draft does not say

Three things this project should not read into the source, recorded because
each was believed at some point in the M4 planning and only one of them
survived reading the text.

- **The draft recommends no trailing padding.** Section 5.3 says only that
  words at the very end "may not be fully packed with variable-length
  codes, which means leaving some available bits unused". The 128-byte
  figure that circulates alongside this is section 5.3's **minimum valid
  stream size** (`4 * 32`), not a recommended pad. The practical conclusion
  is unchanged and is now better founded: **cudec cannot rely on any
  padding after the stream, and the validation ladder must bound-check the
  tail round rather than assume readable bytes past the end.**
- **The draft does not say a lane that decoded a copy sits out the next
  round.** Section 11 says the length and distance symbols occupy two
  back-to-back rounds on the same lane. What the lane is doing on the
  second round is a decoder-side question; the only quiesce rule the draft
  states is section 10's, after end-of-block.
- **The draft states no bit width for a non-compressed block's length
  field.** It says what was removed, in D3, and no more.

### 11.6 Open questions for the kernel design panel

Written here so the panel starts from an agenda rather than rediscovering
it. None of these is answered above, and answering them is not this
section's job. Section 13.2 is where the panel answered them, one by one
and in this order.

1. **Table residency.** One table set per block is read by all 32 lanes
   every round (D6). Shared memory, registers, or a hybrid, and what that
   costs in occupancy against section 9's measured register budget for the
   LZ4 family.
2. **Enforcing the end-of-block quiesce without divergence.** Section 10
   forbids lanes after the end-of-block lane from consuming bits. Doing
   that with a predicate that stays warp-uniform, rather than with a
   branch, is the difference between a correct decoder and a fast one.
3. **Bounding the tail round.** With no padding guaranteed (11.5) and a
   refill that loads whole words, the last round of a page can want bytes
   the page does not have. Where that check lives, and what it costs on
   every round rather than only the last, is open.
4. **The desynchronization threat has no cheap detector** (11.2). Whether
   anything short of full per-use bounds checking is worth having, and
   whether a page-level structural invariant exists at all, is worth one
   deliberate look before the ladder is designed around its absence.
5. **The stored-block path is not a fast path** (D4). Whether it is worth
   any specialization at all is a measurement question and belongs to the
   perf pass, not to the first kernel.

### 11.7 The M4 oracles: what is pinned, and what is refused

**Primary oracle: the NVIDIA fork of libdeflate, branch `gdeflate`.** It
both compresses and decompresses, so one dependency generates the corpora
and serves as the diff target.

A branch name is not a pin. The pin is the commit and the archive hash of
the URL the build actually fetches, in the same shape the LZ4, Snappy and
Zstd oracles already use:

    https://github.com/NVIDIA/libdeflate/archive/
        8ba9502fb30d2bf728592d121f0d402e40c8cb05.tar.gz
    sha256 d1b4c38dce43e68a5f4c28d0fbb3f81a01953039a3dea63f4bd1a84d7ff80592

Recorded deviations from how the other three oracles are pinned, so the
vendoring rung does not have to rediscover them:

- The fork publishes no releases, so this is a commit archive rather than a
  release asset. Snappy is already fetched from a tag archive rather than a
  release asset, so the shape is not new; the novelty is pinning a commit
  with no tag behind it, which is the only thing the fork offers.
- **The fork is not dormant and it does lag.** Both halves are measurable
  and both matter. `gh api repos/NVIDIA/libdeflate/commits/gdeflate`
  reports a commit dated 2026-07-29, so it is alive; and
  `gh api repos/ebiggers/libdeflate/compare/master...NVIDIA:libdeflate:gdeflate`
  reports `ahead_by=4 behind_by=425 status=diverged`, so it sits 425
  upstream commits behind, including whatever correctness fixes those
  carry. That is acceptable in an oracle, whose job is to be the thing real
  data came from, and it is the reason the fork is never a source to derive
  from.
- Licensing on the fork is mixed per file rather than uniform.
  `lib/gdeflate_compress.c` and `lib/gdeflate_decompress.c` each carry Eric
  Biggers' MIT notice from upstream and an
  `SPDX-License-Identifier: Apache-2.0` line with an NVIDIA copyright, and
  the repository's `COPYING` is upstream's MIT text.

Compression levels, read from the fork's own `libdeflate.h` rather than
assumed: the valid range is `[0, 12]`, with 1 fastest, 6 medium and
default, 9 slow, 12 slowest, and the sliding window fixed at compile time
to 65536, "the largest size permissible in the GDEFLATE format". **Level 0
is the interesting one for coverage**: the header defines it as "create a
valid stream, but only emit uncompressed blocks", so it reaches the stored
block type by construction rather than by luck.

**Interop evidence: there is none, and this paragraph claimed there was.**
It recorded the DirectStorage `GDeflateTest` vectors as the one piece of
evidence coming from a different implementation lineage than the libdeflate
fork, and as load-bearing rather than decorative. Both halves are wrong, and
both were found by reading the upstream tree instead of its description.

There are no vectors:

    gh api 'repos/microsoft/DirectStorage/git/trees/main?recursive=1' \
      --jq '[.tree[]|select(.path|startswith("GDeflate/GDeflateTest/"))|.path]'
    ["GDeflate/GDeflateTest/CMakeLists.txt","GDeflate/GDeflateTest/GDeflateTest.cpp"]

`GDeflateTest.cpp` opens no file. It builds its inputs at run time from
`std::default_random_engine`, compresses them and validates the result, so
nothing is written out that anyone could check in.

And the lineage is not a second one. The `GDeflate/` library those two files
call takes its codec from a submodule, at the commit this project already
pins in `cmake/CudecGDeflateOracle.cmake`:

    gh api repos/microsoft/DirectStorage/contents/GDeflate/3rdparty/libdeflate \
      --jq '{submodule_git_url, sha}'
    {"sha":"8ba9502fb30d2bf728592d121f0d402e40c8cb05","submodule_git_url":"https://github.com/NVIDIA/libdeflate"}

`GDeflate::Decompress` allocates that fork's decompressor and calls
`libdeflate_gdeflate_decompress`, the same entry `tests/oracle_gdeflate.cpp`
already drives. Bytes taken from there and diffed here would be one copy of
libdeflate agreeing with another, and they would read as cross-implementation
evidence because of where the wrapper lives, which is worse than holding
none.

What section 11.1 says about `GDeflateTest` stays true and is a different
statement: upstream does cross-check the shipping runtime against that
reference codec, in both directions, so bit-exact interop is the contract
tested **there**. The runtime is the one genuinely external decoder in the
picture, it is reached through `dstorage.h` as a Windows binary component,
and getting its bytes into this tree is a host question rather than a
harness one. Issue #169 carries what is left of the intake.

**Independent second decoder: rejected.** The candidate was `sk-zk/GisDeflate`
(MIT, C#). The decision is no, and the reason is not the harness cost.

Its own README states: "this code is a port of the reference implementation
by Microsoft and NVIDIA", naming both trees, one of them at a pinned
libdeflate commit. A port of the same implementation is not a
non-shared-ancestry second opinion; it shares exactly the ancestry that a
second decoder exists to avoid sharing, so it would agree with the primary
oracle precisely where a shared bug would hide. The cost is real too - it
is a managed .NET component in a tree that carries no .NET anywhere, for
CI and for local runs - but that cost is what would have been weighed if
the coverage had been genuine. It is not.

**What that loses, stated rather than waved away, and it is more than this
passage admitted.** M4 has no independent-lineage oracle for the _reject_
direction on arbitrary bytes: where the libdeflate fork refuses a stream, no
second implementation confirms it should. It has none for the accept
direction either. This list named three compensations and put the
`GDeflateTest` vectors first, as "lineage-independent but cover the accept
direction only, on a fixed set"; the paragraph above records why that is
wrong in both of its terms. Two are left, and neither is a full substitute:

- cudec's own CPU twin, which is written from the draft rather than from
  the fork, so a fork-specific parsing bug does not propagate into it by
  construction - the twin is the second implementation, and this section
  records that it has to carry that weight;
- fail-closed posture as the tiebreaker: where the two disagree and no
  third opinion exists, cudec rejects. Being stricter than an oracle is
  reportable, not fatal, and section 10 already runs that discipline for
  Snappy.

If an independent-lineage decoder appears later, it is worth reopening.
Nothing here forecloses it.

### 11.8 The corpus, and why the level list is not enough on its own

The corpus is Silesia plus the asset-like corpus, compressed through the
fork across a level sweep, in the 64 KiB page geometry section 11.2 fixes.
The asset-like half is still an open M2 decision at the time of writing, so
M4 inherits whatever that settles rather than defining a second one.

The levels to sweep, and what each is there to reach:

- **0** - stored blocks, by construction, per the header text quoted above.
  This is the only level whose block type the source guarantees.
- **1** - the fast end of the search, the level most likely to emit static
  Huffman blocks on small or low-entropy inputs.
- **6** - the default, and therefore the shape most real data arrives in.
- **12** - the slowest search, the level whose table descriptions are
  densest and whose block boundaries are chosen most aggressively.

**And that list is not the coverage argument, only the plan for it.** A
compressor chooses block types for its own reasons and is free to change
them between versions; a corpus that assumes level 1 produced a static
block has assumed its own coverage. So the rule is the one the Snappy
corpus already follows: **each family asserts the block-type composition it
actually got, by inspecting the emitted stream, and a family that did not
reach its target block type fails the test rather than quietly costing
coverage.** Block type is also not a pure function of level - incompressible
input forces stored blocks at any level - so the sweep is level crossed
with input character, and the assertion is what closes the gap between the
two.

The generated corpus carries a pinned digest, as the LZ4 and Snappy corpora
do, so that moving the oracle pin moves the corpus consciously rather than
letting it drift.

### 11.9 Attribution, and what the licences actually require here

Two facts first, both read from the upstream trees rather than assumed,
because the obligation depends on them:

- **Neither upstream ships a NOTICE file.** `GDeflate/` in
  `microsoft/DirectStorage` contains none, and the libdeflate fork ships
  `COPYING` rather than a NOTICE. Apache-2.0 section 4(d) propagates the
  contents of a NOTICE file that exists; where there is none, it has
  nothing to propagate. The obligation this project was expecting to carry
  is therefore not the one it actually has.
- **cudec distributes no upstream bytes.** The oracles are fetched at
  configure time by the test build, exactly as liblz4, snappy and libzstd
  already are, and the no-vendored-binaries rule keeps them out of the
  tree. What cudec distributes is its own source and, where corpora are
  published, data a compressor emitted.

What still gets recorded, and why it is a decision rather than a
compulsion: `NOTICE.md` gains an attribution section naming the three
upstreams that the M4 verification apparatus depends on - the libdeflate
fork with its pinned commit and its mixed MIT/Apache-2.0 file headers, the
DirectStorage GDeflate subtree with its own Apache-2.0 `LICENSE` and its
NVIDIA and Microsoft copyrights, and the IETF draft with its BCP 78 terms.
Recording where the verification apparatus came from is worth doing on its
own terms, and it means the vendoring rung has nothing left to invent.

**This subsection is about attribution only.** The patent position and the
implementation-provenance posture are a separate, maintainer-gated
decision, and nothing here pre-empts or softens it.

## 12. M5 Zstd v1 decode subset (RFC 8878)

Zstd is the one format on the ladder where "what cudec decodes" is a design
decision rather than a reading. The spec's full envelope reaches windows the
spec itself measures in terabytes, dictionaries, and frames that declare no
content size, and none of those three is batch-GPU-shaped. The spec sanctions
refusing part of that envelope in as many words, so the subset below is a
decoder choice made inside the standard rather than a departure from it.

Every fact quoted here was read out of RFC 8878 on 2026-08-06 at
<https://www.rfc-editor.org/rfc/rfc8878.txt>, with the section number beside
it, so a later reader can check the subset against the source instead of
against this paragraph.

### 12.1 The two facts that fix the batch unit

Blocks inside one frame are **not** independent. Section 3.1.1.3 puts
back-references at "previous decoded data, up to a distance of Window_Size,
or the beginning of the Frame, whichever is smaller", and section 3.1.1.4
adds that "all offsets leading to previously decoded data must be smaller
than Window_Size". Entropy state carries across blocks in the same direction:
Treeless literals reuse the previous block's Huffman table, and a sequence
section in Repeat_Mode reuses the previous FSE tables. So decode inside a
frame is inherently sequential, and no choice of kernel shape changes that.

Frames **are** independent. Section 3.1: "Each frame is independent and can
be decompressed independently of other frames. The decompressed content of
multiple concatenated frames is the concatenation of each frame's
decompressed content."

Therefore **the batch unit is the frame**, which is the chunk model the batch
ABI already has. The parallelism M5 sells is across frames; a single large
frame is a sequential decode with intra-block parallelism only. That is the
honest shape, and it is stated here so that no later section has to walk it
back.

### 12.2 The accepted envelope

Inside the subset, everything the format offers is decoded: all three usable
block types (Raw, RLE, Compressed), all four literals section types including
Treeless, all four sequence table modes including Repeat, full repcode
semantics, and the XXH64 content checksum verified whenever the frame carries
one. Section 3.1.1 fixes what that checksum is: "the result of the XXH64()
hash function digesting the original (decoded) data as input, and a seed of
zero. The low 4 bytes of the checksum are stored in little-endian format."

The gate sits at the frame header, and nowhere narrower:

| Frame property              | Accepted                        | Refused with            |
| --------------------------- | ------------------------------- | ----------------------- |
| Magic number                | The Zstandard magic             | `CORRUPT_INPUT`         |
| Skippable frame magic       | Never; 0x184D2A50 to 0x184D2A5F | `UNSUPPORTED`, see 12.4 |
| `Dictionary_ID_flag`        | 0 only                          | `UNSUPPORTED`           |
| `Frame_Content_Size`        | Present                         | `UNSUPPORTED`           |
| `Single_Segment_flag` set   | Yes; window is the content size | -                       |
| `Single_Segment_flag` clear | Window_Size up to 8 MB          | `UNSUPPORTED` above it  |
| Reserved bits in the header | Zero                            | `CORRUPT_INPUT`         |
| Block type                  | 0, 1 or 2                       | `CORRUPT_INPUT`         |
| Block size                  | Up to min(Window_Size, 128 KB)  | `CORRUPT_INPUT`         |

The two window rows come from section 3.1.1.1.2, which recommends "decoders
to support values of Window_Size up to 8 MB" and, in the same section, allows
"a decoder to reject a compressed frame that requests a memory size beyond
the decoder's authorized range". Refusing above 8 MB is therefore inside the
standard, and the number is the spec's own rather than one this project
picked.

`Single_Segment_flag` is the shape the GPU wants, and section 3.1.1.1.1.2
says why it costs nothing to prefer: "If this flag is set, data must be
regenerated within a single continuous memory segment. In this case,
Window_Descriptor byte is skipped, but Frame_Content_Size is necessarily
present." A declared content size is what lets output placement happen before
the first byte is decoded, which is the property the LZ4 batch path already
relies on.

Block_Maximum_Size is section 3.1.1.2.4, "the smallest of: Window_Size [and]
128 KB". Block type 3 is section 3.1.1.2.2: "Reserved: This is not a block.
This value cannot be used with the current specification. If such a value is
present, it is considered to be corrupt data." The spec assigns that rejection
to the corrupt class itself, so cudec does not have to argue for it.

### 12.3 Which rejection class, and why the line sits there

`CUDEC_ERR_CORRUPT_INPUT` means no conforming encoder produced these bytes.
`CUDEC_ERR_UNSUPPORTED` means a conforming encoder did, and cudec does not
decode that part of the format. The distinction is a contract with the
caller: a corrupt verdict is about the data and an unsupported verdict is
about this library, and a caller that would fall back to a CPU decoder needs
to tell them apart.

So a dictionary id, an absent content size and a window above 8 MB are all
`UNSUPPORTED`; each is a valid frame. Reserved bits, block type 3, an
oversized block and a failed checksum are `CORRUPT_INPUT`. Nothing in the
subset is refused silently, and no refusal may leave partial output that a
caller could read as a short decode.

### 12.4 Skippable frames are refused, not stepped over

A skippable frame is refused with `UNSUPPORTED`.

The spec makes stepping over one trivial. Section 3.1.2 gives the magic range
0x184D2A50 to 0x184D2A5F and a `Frame_Size` field "represented using 4 bytes,
little-endian format, unsigned 32 bits", so this is not a difficulty; it is a
scope line. Stepping over one means a chunk holds a frame sequence rather
than a frame, and the moment the unit is a sequence the per-chunk result has
to describe several decodes with one status and one byte count. That is a
batch-ABI question before it is a decode question, and it is what the
extension ladder's first rung is for. Until that rung is taken, one chunk is
one frame.

### 12.5 The extension ladder, with its triggers

Each rung carries the thing that would justify taking it, so that nobody
proposes one without new evidence:

1. **Multi-frame chunks, and skippable frames with them.** Taken when a real
   corpus arrives whose chunks are frame sequences. Needs a per-chunk result
   shape that can describe more than one decode.
2. **Frames with no `Frame_Content_Size`.** Taken when a producer that
   matters emits them. Needs either a sizing pre-pass or a growable output
   contract, and both break the single-pass placement the batch model rests
   on.
3. **Windows above 8 MB.** Taken when a corpus measurably needs one. The cost
   is device memory per in-flight frame, so the trigger is a measurement and
   not a request.
4. **Performance tiers.** Only once the subset decodes correctly. The
   project's ordering, correctness before measured performance, applies here
   with no exception.

Permanently out until a design says otherwise: **dictionaries**, because they
break the frame independence the batch unit rests on, and **legacy v0.x
frames**, which are a different format wearing the same name.

### 12.6 The recorded expectation

Zstd is the slowest of the standard formats to decode on a GPU, and M5 wins
at batch scale or it does not win. The entropy stages are serial by
construction, each sequence's copy depends on the sequences before it, and
12.1's block chain means one frame cannot be spread across the device. This
is written down before the first number is measured, so that the number,
whatever it turns out to be, is read against the expectation rather than
against a hope.

### 12.7 The validation ladder, and the failure contract

Fail-closed, on the same rule the M1 ladder in section 9 states: every value
that came out of the stream is checked before it is first used as an address,
a length or an offset. Zstd needs that rule more than the other three
formats, because a frame carries an optional XXH64 over the content and
nothing at all over the entropy stages, so a table description that decodes
into nonsense is caught by a bound or it is not caught.

The branch list itself is not written out here. It is nine enumerations in
the source, one per unit, and a copy of them in this document would drift
against the code that decides them. What this section fixes is the shape of
the ladder, the class each rung carries, and the property that keeps it
complete. The population is derived:

    grep -ohE 'kZstd[A-Za-z]+Reject[A-Za-z]+' src/zstd_*.h \
      | grep -vE 'RejectNone$|RejectCount$' | sort -u | wc -l
    89

reached from more sites than there are rungs, because several rungs are
refused from more than one place in a parse:

    cat src/zstd_*.h | tr '\n' ' ' \
      | grep -oE 'Refuse\( *kZstd[A-Za-z]+' | wc -l
    128

Where the 89 sit is the useful half, because it says which parts of the
format are dangerous. Counted per unit, `Bit` 6, `Blocks` 6, `Exec` 8,
`Frame` 15, `Fse` 10, `Huf` 11, `Literals` 14, `Repcode` 2, `Seq` 17, and
each of those is the same command with the unit name substituted.
Fifty-eight of the eighty-nine sit in the entropy layer and its two
consumers, which is the opposite of LZ4, where the whole ladder is one token
loop.

Walking the pipeline, and naming what each stage refuses rather than
enumerating the rungs it does it with:

- The frame header refuses a truncated or wrong magic, a truncated
  descriptor, a set reserved bit, a truncated header, and separately the
  three shapes that are legal and outside the subset (12.2).
- The block header refuses the reserved block type 3, a block above
  `min(Window_Size, 128 KB)`, a truncated header and a truncated body, and
  the frame refuses a checksum that is truncated or does not match.
- The literals section refuses a truncated header in each of its size
  formats, a regenerated size above the block bound, a payload that is short
  for what the header declared, Treeless reuse with no previously
  established table, a four-stream jump table that overruns its payload, a
  four-stream section with fewer literals than four streams can carry, and a
  stream a decode did not consume exactly.
- Huffman weight decode and table construction refuse an empty or truncated
  description, more weights than the alphabet holds, a weight above the
  maximum, a weight sum of zero, a table log above the format's ceiling, a
  last weight that does not complete the distribution, and a deepest rank
  left empty.
- The FSE table description refuses an empty or truncated description, an
  accuracy log above the ceiling, a symbol past the field's maximum, a
  probability set that does not normalise, a truncated state initialisation
  and a state outside the built table.
- The sequences section refuses a truncated `Number_of_Sequences`, a count
  larger than the block's regenerated size could hold, a count of zero
  followed by bytes nobody consumes, truncated or reserved compression
  modes, Repeat mode with no previous table, a truncated RLE table byte, a
  decoded symbol past the alphabet, a truncated sequence, and a backward
  bitstream that is missing, unstarted or not consumed to its start marker.
- Repcode resolution refuses a resolution to zero, which is the one state
  the rotation can reach that no valid stream reaches.
- Sequence execution refuses an offset of zero, an offset past the window,
  an offset before the bytes produced so far in the frame, literals
  exhausted before the plan consumed them, and a plan whose totals do not
  agree with the block it came from.

The class rule is 12.3's and is not restated. What is worth recording is
that the rule is executed rather than described: the whole Zstd path returns
`CUDEC_ERR_UNSUPPORTED` at exactly four rungs, and they are exactly the
shapes 12.2 and 12.4 put outside the subset.

    grep -B3 -hE '^\s+CUDEC_ERR_UNSUPPORTED' src/zstd_*.h \
      | grep -oE 'kZstd[A-Za-z]+Reject[A-Za-z]+' | sort -u
    kZstdFrameRejectContentSizeAbsent
    kZstdFrameRejectDictionaryId
    kZstdFrameRejectSkippableFrame
    kZstdFrameRejectWindowTooLarge

Everything else is a corrupt verdict about the data, a caller-argument
refusal, or a capacity refusal, and a rung that drifts across that line shows
up in the command above rather than in a review.

The failure contract is section 9's, unchanged and repeated here because it
is what the rungs above are for: on any reject, `bytes_written = 0` and a
non-OK status, destination contents up to the failure point unspecified and
never presented as a short decode. Check-before-load is batch isolation
rather than parity alone. On a device an out-of-bounds read is not a
per-chunk failure, it faults the launch and poisons the whole batch, so the
check has to happen before the load and not after it.

What keeps the ladder complete is a per-unit sweep in the twins rather than a
promise in this document. Each twin holds a coverage array sized by its
unit's `RejectCount`, marks the rung every declared negative reached, and
requires at the end that every rung between `None` and `Count` was reached by
a negative written for it. There is no exemption list, so a rung added to a
parser with no negative behind it lands as a named hole rather than as
silence. All nine units carry one:

    grep -l "RejectCount\]" tests/zstd_*.cpp | wc -l
    9

That is the conformance expectation this section fixes, and it is one crafted
negative per ladder rung, checked by walking the enumeration rather than by
counting to a number written down anywhere.

### 12.8 What bounds every loop

A decoder that rejects a hostile stream and a decoder that never returns on
it are the same outcome to a caller, and on a device they are not: a warp
that does not retire is a hang rather than a failure, and the fail-closed
contract admits neither. So every loop in the Zstd path is either counted
before it is entered or carries a fuel cap.

Counted before entry is the ordinary case and it is the whole reason the
rungs above sit where they do. The sequence loop runs `Number_of_Sequences`
times, and that count is refused above `block_size_max / 3` before the loop
exists. The literal decode runs to the regenerated size, refused above the
block bound first. Every table build runs to a table size fixed by an
accuracy log that was refused above its ceiling first. In each case the bound
is a value the ladder has already refused if it was wrong, which is what
makes the loop's trip count a fact rather than a hope.

Three loops in the path are not counted, and each one carries fuel:

    grep -cE '^\s*(while|do)\b' src/zstd_*.h src/frame.cpp | grep -v ':0'
    src/zstd_blocks.h:1
    src/zstd_fse.h:1
    src/frame.cpp:1

The two block loops walk a chain whose length the frame does not declare, so
each is capped by the smallest number of bytes a step can consume: a block
header is 3 bytes in `src/zstd_blocks.h` and the frame walk in
`src/frame.cpp` uses 4. Both caps sit beyond anything a frame the guards
admit can reach, so exhausting one is a bounds bug surfacing as a rejected
frame rather than as a spinning thread. The third is the FSE spread's
low-probability placement in `src/zstd_fse.h`, capped at the table size,
which a normalised vector cannot exhaust and an unnormalised one is refused
for.

There is no `goto` anywhere in the path, so the three above and the counted
loops are the whole of it.

### 12.9 The worst-case corpus families, and what would falsify the ladder

`docs/ZSTD-CORPUS.md` records the coverage corpus, whose job is to reach
every mode of the format at least once and to prove it did. That is a
different corpus from this one. What is named here is the adversarial set,
whose job is to make the ladder and the loop bounds above expensive rather
than merely reached, and each family is written so the corpus and fuzz rungs
can be built from it without re-deriving the shape.

- **Maximum sequence density.** A block whose regenerated size is at the
  ceiling and whose sequences are all at the three-byte minimum match, so the
  sequence count sits against the `block_size_max / 3` bound the loop is
  capped by. This is the family that makes the sequence loop's trip count
  worst and the per-sequence rung checks hottest.
- **Maximum table switching.** A frame whose blocks alternate the three
  fields' compression modes on every block, so `Set_Compressed` tables are
  rebuilt as often as the format allows and `Repeat` never gets to amortise
  anything. The table build is the serial part of the decode, and this is the
  shape that pays for it most often.
- **Adversarial repcode patterns.** Sequences that walk the rotation through
  every state it has, including the `literals_length == 0` shift and the
  rep1-1 case, arranged so a wrong rotation produces a plausible offset
  rather than an obviously wrong one. The rung that catches a wrong rotation
  is a resolution to zero, and a corpus that never approaches zero never
  exercises it.
- **Deep Huffman trees and four-stream literals.** Literals whose tree
  reaches the format's maximum code length with four streams and a jump
  table, so the stream bounds and the last stream's implied size are read at
  their tightest.
- **The high-search interop family.** libzstd at level 18 and above emits
  frames whose shape a decoder written against the common cases can get
  wrong, and `docs/ZSTD-CORPUS.md` already carries level 18, 19 and a
  level 22 long-window cell for it. Named here as an adversarial family and
  not only a coverage one, because what it is for is the divergence rather
  than the mode.
- **The tail shapes.** A frame whose last block is empty, a block whose
  sequence section is a count of zero, and a frame whose declared content
  size is reached exactly at the last literal, so the leftover-literals rule
  and the exact-consumption checks are read at their boundary rather than
  near it.

What would falsify the ladder, recorded so a run that finds nothing is not
read as a run that proved something:

- A mutant the pinned reference accepts and cudec rejects, on any family
  above. Over-strictness is a divergence in the same way over-permissiveness
  is, and the corpus that catches it has to keep the accepted mutants rather
  than filter them out.
- A mutant the reference rejects and cudec accepts. That is a fail-open and
  it is the finding this whole section exists to make findable.
- A fuzz run that reaches a rung no declared negative reaches. The sweep in
  12.7 says every rung has one, so this would mean the fuzzer found a route
  to the rung that the negative does not take, which is a narrower negative
  than the rung deserves.
- A stream on which any of the three fuel caps in 12.8 fires. Each cap is
  argued to be unreachable for every input the guards admit, so a firing cap
  is a bounds bug that the cap converted into a rejection, and the bug is the
  finding rather than the rejection.

## 13. M4 GDeflate kernel design (settled via the #94 design panel, 2026-08-23)

Three candidate shapes were developed against the section 11 dossier and
scored on one piece of arithmetic, the shared-memory footprint of a decode
table set against the resident-warp target section 9 fixed for this kernel
family. Two of the three are dead on that arithmetic alone and are recorded
here so nobody re-proposes them. The panel was run inside this work rather
than by a second person, which is stated here rather than implied; the
arithmetic below is reproducible from the script it carries, and that is
what stands in for a second judge.

**One kernel. One warp per 64 KiB page over a grid-stride loop. The 32
lanes are the format's 32 substreams, so the parse is genuinely
cooperative rather than section 9's redundant lockstep. One table set per
block in shared memory, with a fixed footprint that no input can grow, and
the copy engine, failure contract and determinism discipline carried over
from section 9 unchanged.**

### 13.1 The arithmetic that forces the shape

The binding resource is shared memory per resident warp. Section 9's
occupancy plan targets at most 64 registers per thread for at least 32
warps per SM, and section 9's own two-phase disproof reads an SM's usable
shared memory as roughly 100 KiB, the sm_86 figure from the vendor's
compute-capability table. sm_80, the baseline, has more of it; sm_86, the
tuning target, is therefore the binding case. At 32 warps per SM the budget
is 100 KiB / 32 = 3200 bytes of shared memory per warp, and every candidate
below is scored against that one number.

**Candidate A, two-level decode tables in shared memory, is dead.** A
GDeflate block's code lengths are attacker-chosen inside what the format
admits, so the table has to be sized for the worst legal case and not for a
corpus. That worst case is computable in closed form. A root of R bits
resolves every code no longer than R in one lookup; each root slot whose
codes run longer gets a subtable of `2^d` entries, where d is the deepest
code under that slot minus R. A complete subcode of depth d costs at least
d+1 symbols, and filling k root slots with codes of length at most R costs
at least `popcount(k)` symbols, so the worst case is a small knapsack over
the symbol budget:

```python
def worst(n, R, L):
    root = 1 << R
    best = 0
    for s in range(0, min(root, n) + 1):
        budget = n - bin(root - s).count("1")
        if budget < 0:
            continue
        dp = [[-1] * (budget + 1) for _ in range(s + 1)]
        dp[0][0] = 0
        for d in range(1, L - R + 1):
            cost, val = d + 1, 1 << d
            for c in range(1, s + 1):
                for b in range(cost, budget + 1):
                    if dp[c - 1][b - cost] >= 0:
                        dp[c][b] = max(dp[c][b], dp[c - 1][b - cost] + val)
        best = max(best, root + max(dp[s]))
    return best


for n in (286, 32):
    print(n, min((worst(n, R, 15), R) for R in range(4, 13)))
```

    286 (2504, 10)
    32 (642, 8)

So the cheapest two-level literal/length table any root can produce is 2504
entries, at a 10-bit root, and the cheapest distance table is 642 entries
at an 8-bit root. At two bytes per entry, the narrowest entry that still
carries a symbol index up to 285 and a code length up to 15, that is
(2504 + 642) x 2 = 6292 bytes per warp. It is 1.97 times the 3200-byte
budget, and a 4-warp block of it takes an SM to 16 resident warps, half the
family target. The 2048-entry figure that circulated in the M4 planning
before this pass is not a bound on anything, and the script above is where
that was found.

**Candidate B, a phase split that materializes the symbol stream, is dead
for the reason section 9 killed table-in-smem two-phase.** A 64 KiB page
decodes to at most 65536 bytes, so its symbol stream can be 65536 literals;
at 4 bytes per record that is 256 KiB for one page, 2.6 times an SM's
shared memory, for a single tile rather than for a warp. Moving that
scratch to global memory does not rescue it: it would put a per-tile
workspace into a batch ABI that has none today, and it would spend two
passes over that scratch at full residency to accelerate a stage that is
entropy-bound rather than bandwidth-bound.

**Candidate C, block per tile, buys nothing the format can use.** Thirty
two substreams are 32 lanes, so a 128-thread block leaves 96 lanes idle
through every parse round, and the only work they could take is the copy
fan-out, which is ordered by the in-tile LZ77 dependency and therefore
cannot run ahead of the parse without the phase split candidate B already
rules out. It also turns the section 10 end-of-block quiesce rule from a
warp-level ballot into a block-wide barrier on every round.

**The shape that survives is the fixed-footprint table.** The decode state
is sized so that no input can change how much shared memory a warp holds,
which is the anti-pattern rule section 9 states for packed fields, applied
to an allocation instead of to a field. Per warp:

| what                                              | bytes |
| ------------------------------------------------- | ----- |
| canonical symbol order, 286 lit/len + 32 distance | 636   |
| per-length counts, first codes and first indices  | 192   |
| literal/length root accelerator, 10 bits          | 2048  |
| distance root accelerator, 7 bits                 | 256   |
| total                                             | 3132  |

The symbol-order array and the per-length arrays are a complete canonical
decoder on their own, and their size is fixed by the alphabet rather than
by the code lengths, so the worst case above cannot arise. The two root
tables are accelerators over that decoder: a code no longer than its root
resolves in one lookup, and a longer one falls back to the length-by-length
walk, which is the case that is rare by construction, because a long code
is what a rare symbol gets. The footprint is 3132 bytes against the
3200-byte budget, so a 4-warp block takes 12528 bytes, an SM fits eight of
those blocks inside 100 KiB, and that is exactly the 32 resident warps
section 9 asks for. The 68 bytes of margin per warp are slack for the
allocation granularity rather than a rounding accident.

Registers stay where the draft puts them. Section 11.2's per-lane state is
a bit buffer and a count of valid bits, which `src/gdeflate_schedule.h`
already carries as one 64-bit buffer per lane
(`kGDeflateLaneBufferBits`). Per-lane state has no reason to sit in shared
memory, and moving it there would put the one thing that is genuinely
private into the one resource that is shared.

### 13.2 The five open questions from 11.6, answered

**Table residency** is settled by 13.1: shared memory, at a footprint no
input can grow, with the two-level table rejected on its worst case rather
than on its average.

**The end-of-block quiesce without divergence.** Section 10 forbids every
lane after the one that decoded code 256 from consuming bits. The decode
round is therefore speculative and its commit is predicated: every lane
decodes, `__ballot_sync` collects the lanes that produced 256, `__ffs` of
that mask names the first of them, and each lane commits its new bit
position only if its own index is at or below that lane. A lane that must
not have consumed keeps the bit position it entered the round with, which
is a select rather than a branch, and the warp stays convergent. The rule
is enforced by not committing, never by rolling back.

**Bounding the tail round.** Section 11.5 forbids assuming any bytes past
the stream, so the refill is bounded by the page's word count, which comes
from the caller's compressed size rather than from the bits. The refill
schedule is warp-uniform, so that is one comparison per refill and not one
per lane, and `src/gdeflate_schedule.h` already fails it closed with a
frozen cursor and a sticky failure. A round that does not refill pays
nothing for it.

**The desynchronization threat.** The deliberate look 11.6 asked for, with
its answer: there is no cheap detector during a walk, there are two free
ones at the end of it, and both are mandatory rather than optional. The
first is exact word consumption, since a correct decode leaves the cursor
inside the page's last word and a desynchronized one lands anywhere. The
second is exact output size, which GDeflate itself does not carry (11.4)
but the envelope does, so the declared uncompressed size is compared
against what was produced rather than trusted to size a buffer. Neither
catches a desynchronization early; together they turn one into a rejection
instead of a wrong answer, which is the whole of what a checksum-less
format offers. Every bound stays enforced at its point of use, as 11.2
requires.

**The stored-block path** is not specialized in the first kernel. D4 makes
it an entropy-shaped path like any other, and whether it deserves a fast
path is a measurement question that belongs to the perf pass, filed as #206
against a measured block-type mix.

### 13.3 The validation ladder

Every value read from the stream is checked before its first use as an
address, a length or an offset, and the ladder is the M4 instance of
section 9's rather than a second discipline:

- the page is at least the 128 bytes 11.2 makes the minimum, and its word
  count is derived from the caller's compressed size, never from the bits;
- the reserved block type never appears, and a block header round that
  claims one is a reject;
- the code-length alphabet is checked for oversubscription and for
  incompleteness in both directions, since an incomplete literal/length
  code admits a decode that walks off the end of the symbol order;
- a repeat code 16 with nothing to repeat, and a 17 or 18 run that carries
  the length vector past the 318 the format allows, are rejects;
- a distance greater than the bytes already produced in this tile is a
  reject, because a tile is independent and a match may never reach before
  its start;
- a stored block whose declared length overruns the page or the remaining
  destination capacity is a reject, and D3 removes the one's complement
  that would otherwise have caught it, so this bound is the only one there
  is;
- a block that ends without its end-of-block symbol is a reject, and a lane
  that consumes bits after the end-of-block lane is prevented rather than
  detected, per 13.2;
- the refill is bounded on every round, per 13.2;
- success only on exact word consumption and exact declared output size,
  per 13.2.

The round loop carries fuel in the shape section 12.8 records for Zstd: a
cap on rounds per page derived from the page's word count, argued to be
unreachable for every input the bounds above admit, so a firing cap is a
bug that the cap converted into a rejection and the bug is the finding.

**Failure contract**, unchanged from section 9: on any reject
`bytes_written` is zero and the status is non-OK, the destination up to the
failure point is unspecified and is never presented as success, and every
bound is checked before the load rather than after it, because on a GPU an
out-of-bounds read can fault the launch and poison the whole batch rather
than one tile.

### 13.4 The batch entry

`cudec_gdeflate_decompress_batch` takes raw pages with a caller-provided
size array, which is the draft's external-enveloping model (11.4) and the
shape `cudec_lz4_decompress_batch` already froze. One page is one chunk, so
per-page status maps onto `cudec_chunk_result` with no new vocabulary and
the 16-byte layout its static assertions pin does not move. The kernel is
envelope-ignorant: the DirectStorage TileStream container is parsed on the
host, above this entry, and hands it pages. That split is #216's to
implement and #177's to drive, and neither is redesigned here.

A page whose declared uncompressed size exceeds the destination capacity
the caller gave is `CUDEC_ERR_OUTPUT_TOO_SMALL` with `bytes_written` zero,
and the declared size never sizes an allocation, per the anti-pattern rule.

### 13.5 The worst case, and what would falsify this design

The M4 adversarial corpus is not LZ4's `worst-4Bmatch` in another format.
What costs this kernel is rounds and refills per decoded byte, so the shape
to construct is a page that maximizes them: symbols spread so that every
lane consumes near its watermark on every round, copies split across two
rounds on the same lane (section 11), block headers frequent enough that
the one-active-lane header rounds are a real share of the total, and code
lengths long enough that the root accelerator misses and the canonical walk
runs. The quantity that corpus locks is rounds per decoded byte, and its
generator asserts a floor on it, so a generator that stops being
adversarial reds CI rather than quietly reporting a better number. That
corpus is #226 and the density floor is its proof.

What would falsify this design, recorded before it is built:

- A measured table-build cost per block that is a large share of a page's
  decode time. The fixed-footprint table trades a bounded build for a
  slower tail on long codes and is chosen on memory rather than on time. If
  the build dominates, the answer is a smaller root and a cheaper build,
  not the two-level table 13.1 rejected, whose worst case is what made it
  dead.
- A measured root-accelerator miss rate high enough that the canonical
  walk, rather than the parse, sets the round time. That is a measurement
  against the real corpus and against #226's adversarial one, the two are
  allowed to disagree, and the adversarial one is the number that binds.
- Achieved registers above 64 per thread, which would put resident warps
  below the 32 this whole budget is derived from and invalidate the
  3200-byte figure rather than merely miss a target. Registers are read out
  at the kernel pull request and recorded, as section 9 requires.
- Any input that makes the shared-memory footprint depend on the stream.
  That is the anti-pattern rule broken, and it is a defect in this design
  rather than a tuning result.

## 14. M5 Zstd kernel shape (settled via the #208 design panel, 2026-08-23)

**One kernel, fused, with phase barriers inside the block, at block per
frame with 128 threads.** The frame is the chunk, as 12.1 fixes; the block
of four warps is what decodes it; and the phases inside a Zstd block are
separated by `__syncthreads()` rather than by a launch. Shared memory holds
one entropy table set per frame, the destination buffer holds everything
else, and no workspace is added to the batch ABI.

As in 13, the candidates were developed and scored inside this work rather
than by a second person; the arithmetic below is the evidence in place of a
second judge, and every constant it uses is read out of the tree or out of
the section it cites rather than recalled.

### 14.1 The table set is the whole budget

What a frame must hold across a block boundary is fixed by 12.1 and by the
subset in 12.2, and the ceilings are already constants in the tree:

| table                                 | entries    | bytes |
| ------------------------------------- | ---------- | ----- |
| Literal_Lengths FSE, accuracy log 9   | 512 x 4 B  | 2048  |
| Match_Lengths FSE, accuracy log 9     | 512 x 4 B  | 2048  |
| Offsets FSE, accuracy log 8           | 256 x 4 B  | 1024  |
| Literals Huffman, table log 11        | 2048 x 2 B | 4096  |
| Huffman weight description FSE, log 6 | 64 x 4 B   | 256   |
| total per frame                       |            | 9472  |

The three sequence accuracy logs are `kZstdLitLenAccuracyLogMax`,
`kZstdMatchLenAccuracyLogMax` and `kZstdOffsetAccuracyLogMax` in
`src/zstd_fse.h`; the literals tree depth is `kZstdLiteralsMaxTableLog` in
`src/zstd_literals.h`, which is the format's own eleven-bit limit and not a
project choice; the weight description's ceiling is
`kZstdHufWeightAccuracyLogMax` in `src/zstd_huf.h`. A four-byte FSE entry
is a next state, a symbol and a bit count, and a two-byte Huffman entry is
a symbol and a bit count.

That 9472 bytes is per FRAME and not per warp, because 12.1 makes the
entropy state cross block boundaries inside a frame and not between frames.
It is the number that decides the granularity.

### 14.2 Warp per frame is dead, and 128 threads is the smallest block that is not

An SM's usable shared memory is the roughly 100 KiB section 9's two-phase
disproof already reads for sm_86, the tuning target and the tighter of the
two architectures. So an SM holds `floor(102400 / 9472) = 10` frames,
whatever else it does.

Section 9's occupancy plan is at most 64 registers per thread for at least
32 resident warps per SM, and 32 warps is 1024 threads, which is exactly
the 65536 registers an SM has at 64 per thread. Against those two limits:

- **Warp per frame** makes ten frames ten warps. Shared memory becomes the
  binding resource and costs 22 of the 32 warps the register budget would
  have paid for. The table set is per frame, so a warp-per-frame kernel
  pays the whole 9472 bytes for 32 lanes.
- **Block per frame, four warps (128 threads)** needs eight blocks for 32
  warps, and eight blocks take 75776 bytes, inside the 100 KiB with room
  left. Registers bind, shared memory does not, and the family's occupancy
  target is met rather than argued down.
- **Block per frame, two warps (64 threads)** needs sixteen blocks for 32
  warps, and sixteen blocks want 151552 bytes, which is over. Shared memory
  binds again at ten blocks and twenty warps.
- **Block per frame, eight warps (256 threads)** meets 32 warps at four
  blocks and fits easily, but it puts four frames per SM in flight where
  128 threads puts eight, and inter-frame parallelism is the only
  parallelism M5 sells (12.1). Fewer, wider blocks is the wrong direction
  for a batch decoder.

So 128 threads is the smallest block size at which the table set stops
being the binding resource, and being the smallest is the point: it
maximizes the number of frames an SM decodes at once.

### 14.3 The phase pipeline is dead on its intermediates

The one production-grade decomposition to weigh is a phase pipeline of
separate passes, parse then table init then literals then sequence decode
then prefix sums then execution, with the intermediates in global memory.
The arithmetic that kills it here is the size of those intermediates.
`Number_of_Sequences` is a three-byte varint biased by
`kZstdSeqCountThreeByteBias` (0x7F00) in `src/zstd_seq.h`, so a block may
declare `0xFFFF + 0x7F00 = 98047` sequences. At twelve bytes for a
literals length, a match length and an offset, that is about 1.1 MiB of
tuples for one block, and at the eight resident frames per SM of 14.2
across an RTX 3080's 68 SMs it is on the order of 600 MiB of scratch for
the batch, live for the whole launch.

That scratch has nowhere to come from. `cudec_lz4_decompress_batch` takes
source pointers, destination pointers, capacities and a results array, and
nothing else; a phase pipeline would have to add a workspace argument to
the batch ABI, or allocate behind the caller's back inside an entry that is
asynchronous on the caller's stream and today allocates nothing. Neither is
a cost this shape has earned, and the fused shape does not incur it.

**The fused shape needs no scratch at all, and 12.2 is why.** The subset
requires `Frame_Content_Size` to be present, so the destination region for
a frame is known before the first byte is decoded. A block's literals are
decoded into the TAIL of the frame's remaining destination region: with `R`
bytes of declared output left at the start of the block and `L` literal
bytes regenerated, they land at `[P + R - L, P + R)`. Sequence `i` then
writes its literal run at `P + prefix_i`, and `prefix_i` is the literals
before it plus the match bytes before it, so `prefix_i <= R - L +
litprefix_i` for every `i`. Every literal run therefore moves toward lower
addresses, and run `i` ends at or before run `i+1`'s source begins, so
executing them in increasing order is safe in place. The prefix sum itself
is a warp scan with a running carry rather than a materialized array, which
is what `src/zstd_exec.h` already says the device form of that unit is.

That leaves one thing in shared memory, the table set, and one thing in
global memory, the caller's destination. The phase barriers that a fused
kernel needs are `__syncthreads()` between the literals phase and the
sequence phase of each block, which is a barrier inside a block of four
warps and not a launch.

### 14.4 What this keeps from the family and what it breaks

Kept: the frame-to-chunk mapping and the batch ABI shape; the failure
contract, with `bytes_written` zero on every reject; determinism, since
every copy is a pure function of lower addresses exactly as section 9
records; the overlap semantics `src/zstd_exec.h` already carries; and the
rule that every stream-decoded value is checked before its first use.

Broken, knowingly: section 9's "no shared memory" property, which section 4
already records as an LZ4 decision rather than a family invariant, and
section 9's redundant 32-lane lockstep parse, which cannot survive a format
whose entropy decode is serial per stream. The `template<class Parser>`
chunk-decoder seam section 9 describes is not extended to M5: a Zstd frame
is not a chunk with a different parser, it is a block loop with its own
phases, and pretending otherwise would put a parser interface between the
phases that has to carry all of them.

### 14.5 The M5 pull-request ladder

The entropy and format rungs landed first and are closed: the frame and
block header walk, the FSE and Huffman table machinery, the literals
section in all four types, the sequences section and its three-state loop,
repcode resolution, sequence execution against a prefix sum, and the block
loop that carries state across blocks, each as a host-twinnable header
under `src/` with its twin under CI. The validation ladder, the fuel story
and the worst-case corpus families are recorded in 12.7 to 12.9.

What is left, in order, each its own gated pull request:

1. The intra-kernel work assignment, which is #213's to settle and is
   deliberately not settled here: which lanes run the serial FSE cores,
   how the four literal streams are assigned, and how the sequence
   destinations are scanned.
2. The kernel and `cudec_zstd_decompress_batch` (#203), minimal-correct,
   behind the frozen batch contract, with achieved registers and occupancy
   read out and recorded at that pull request.
3. The device gate set: determinism, two-directional mutant reject parity
   against libzstd, and the capacity adversarials.
4. The first recorded GPU baselines and the per-phase split (#231, #233).
   Numbers and kernel never move in the same pull request.
5. The measured perf levers (#234 to #239), each accepted only on a
   recorded improvement with every gate green.

Nothing above re-cuts a closed issue. The rungs that landed are format
units, and a kernel shape does not move where a format's bounds are; the
two rungs this section binds are the kernel and its ABI entry, and it binds
them by naming where the literals live.

### 14.6 What would falsify the shape

- **A measured literals phase that dominates.** The four-stream literals
  decode is at most four-way parallel, and a `Size_Format` of zero makes it
  one-way, so 124 of 128 threads have nothing to do in that phase. If it is
  the majority of a frame's time, block per frame is buying occupancy that
  is idle, and warp per frame's ten resident frames stop being the worse
  trade. That is the one measurement that reopens 14.2, and #236 and #237
  are the levers that would be tried first.
- **Achieved registers above 64 per thread.** The whole of 14.2 is derived
  from 32 resident warps at 64 registers. Above it the block count falls
  and the shared-memory headroom the four-warp block relies on stops being
  headroom.
- **An extension that widens a table.** 12.5's extension ladder is where a
  wider window or a larger table log would arrive, and the 9472 bytes moves
  with it. A literals tree of depth twelve alone would take the table set
  to 13568 bytes and eight blocks to 108544, which is over the 100 KiB, and
  the block size would have to be re-derived rather than assumed to still
  hold.
- **A frame where the literal tail placement does not fit.** The invariant
  in 14.3 says it always does, so a case where it does not is a defect in
  that invariant and is the finding, not a reason to add a workspace.
