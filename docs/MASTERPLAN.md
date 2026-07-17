# cudec — masterplan

The design record: what is being built, why it is shaped this way, and in
which order. Settled decisions live here with their rationale; open questions
are listed at the end and get settled through design issues before the code
that depends on them is written.

## 1. Positioning

cudec is an open-source CUDA library that batch-decodes the standard
compression formats — LZ4, Snappy, GDeflate/DEFLATE, and eventually Zstd —
on the GPU.

The field today:

| Existing work              | Why it does not fill the gap                              |
| -------------------------- | --------------------------------------------------------- |
| nvCOMP / nvCOMPDx (NVIDIA) | Proprietary since v2.3; NVIDIA-only; not auditable        |
| dietgpu (Meta)             | Open, but its own rANS format — not format-compatible     |
| GDeflate reference (MS/NV) | Open spec + CPU codec; the GPU decoder is Windows-runtime |
| Academic prototypes        | Unmaintained research code                                |

There is no maintained open-source library that decodes the standard formats
on the GPU. The value proposition is not price (nvCOMP is free to use) but
**auditability** (decompressors are classic attack surface; every bounds
check here is readable, tested, and fuzzed), **portability** (a HIP port is
a planned milestone — a vendor binary can never follow), and **hackability**.

## 2. Scope

- **Decode-only.** Compression stays on the CPU: that is where the encoders,
  the tooling, and the flexibility live, and encode throughput is rarely the
  bottleneck. Decode is.
- **Batch-oriented.** The GPU wins when many independent chunks decode in
  parallel — asset streaming, analytics scans, ML data loading. A single
  small file over a cold PCIe bus loses to the CPU and always will; the
  documentation says so.
- **Formats over percentage points.** The library's value is format
  coverage. Once a format hits a healthy fraction of memory bandwidth, the
  next format outranks the next percent.

## 3. Format ladder (and why this order)

1. **LZ4** (M1/M2) — byte-oriented, no entropy coding: token/literal/match
   sequences only. The simplest correct GPU decode, the fastest path to an
   honest benchmark, and the kernel-family foundation everything else
   reuses. Block format first, then the frame format and the batch API.
2. **Snappy** (M3) — structurally close to LZ4 (varint-length literals and
   copies, no entropy stage); a cheap second format that proves the kernel
   family generalizes.
3. **GDeflate** (M4) — the strategic differentiator. The format is designed
   for GPU decode (the DEFLATE bitstream is split into 32 interleaved
   sub-streams so a full warp reads Huffman codes in parallel), the spec is
   open, and the reference implementation is CPU-only: the only shipping GPU
   decoder lives inside the Windows DirectStorage runtime. An open CUDA
   GDeflate decoder — usable on Linux, in HPC, anywhere — does not exist
   until cudec ships it.
4. **Zstd decode** (M5) — the flagship and by far the hardest: FSE/Huffman
   entropy stages feeding an LZ77 sequence executor with long-range matches.
   Attempted only once the kernel family, the oracle net, and the benchmark
   discipline are proven on 1–3.
5. **HIP port** (M6) — portability as a moat. The kernels stay on warp-level
   primitives available on both vendors to keep this tractable.

## 4. Architecture pillars

- **C ABI, thin C++ inside.** The public surface is a small C header
  (`include/cudec.h`); no exceptions cross it, every function returns a
  defined status. Internally plain CUDA C++ with no dependencies beyond the
  CUDA toolkit.
- **Batch API.** The core entry point decodes N independent chunks described
  by device-side arrays (src pointers/sizes, dst pointers/capacities) on a
  caller-provided stream, reporting per-chunk status — modeled on the shape
  proven by nvCOMP's batch interface, implemented independently.
- **Kernel dispatch by chunk size.** Warp-per-chunk for small chunks,
  block-per-chunk (a warp team sharing staged input) for large ones; the
  dispatcher picks per batch histogram. Shared-memory staging via
  `cp.async`.
- **Two memory paths.** Device-resident (caller already has the data on the
  GPU) and a pinned-host streaming path that overlaps H2D copies with decode
  — the asset-streaming shape.
- **Fail-closed decode contract.** Every value decoded from the bitstream is
  validated before it is used as an address, length, or offset; size
  arithmetic is overflow-checked; a chunk that fails validation reports a
  defined error and produces no partial output presented as success. Hostile
  input is the expected case, not the exception.
- **Determinism.** Same input → bit-identical output on every path. There is
  no floating point in a lossless decoder; this invariant is cheap here and
  is locked in by tests.
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
  from ctest; buffer diffs report first-mismatch offset plus a hex window —
  the one assertion domain a framework would not improve. Rationale:
  near-zero supply-chain surface for the security net itself, and
  early-abort semantics fit contract-sequence tests. Recorded reassessment
  trigger: if early-abort measurably masks failure clusters at M5 mutant
  scale, or an outside contributor base emerges, migrate to Catch2 (pinned)
  — the REQUIRE-shaped macros survive that move verbatim, and no framework
  headers are compiled through nvcc-only paths that would complicate it.
- **A GPU test never self-skips** (banned pattern): no "no device, exit 0"
  branches — skipping is exclusively the runner's decision via ctest labels
  (`-LE gpu` on the GPU-less CI runner; the full run on the local gate). A
  mislabeled GPU test in CI therefore fails loudly instead of passing
  vacuously; `--no-tests=error` closes the zero-tests-selected route; CI
  prints the deferred `-L gpu -N` listing and pins the known GPU tests by
  name.
- **Oracle pinning policy**: oracles are vendored via FetchContent from
  maintainer-uploaded release assets, pinned by a self-computed SHA-256
  cross-checked against a second packaging ecosystem; auto-generated
  `/archive/` tarballs are avoided (GitHub regenerated them in 2023 and
  their hashes moved). Where a project publishes no uploaded asset, the
  archive tarball is pinned and a future hash mismatch is treated as the
  invariant working, not as noise. FetchContent pins are invisible to
  Dependabot — oracle bumps are deliberate manual PRs owned by the
  supply-chain sweep. Third-party oracle code compiles with SYSTEM includes
  and without the project's strict flags; one translation unit per oracle,
  never its build system.
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
| M0 — Foundation  | Toolchain decision + GPU probe, CMake+CUDA skeleton, CI gate, test harness + LZ4 oracle, bench skeleton, masterplan complete |
| M1 — LZ4 block   | Warp-cooperative LZ4 block decode, fuzz-diffed against liblz4, first honest numbers                                          |
| M2 — LZ4 batch   | Frame format, batch API, pinned-host streaming path, published benchmark + methodology write-up                              |
| M3 — Snappy      | Snappy decode on the shared kernel family                                                                                    |
| M4 — GDeflate    | The first open GPU GDeflate decoder (32-substream warp decode)                                                               |
| M5 — Zstd        | Zstd decode: FSE/Huffman stages + sequence execution                                                                         |
| M6 — Portability | HIP port of the kernel family                                                                                                |

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

- Benchmark corpus set beyond Silesia/enwik (game-asset-like data) — M2.
- Device-side chunk-size binning for mixed batches — M2 (see section 9:
  the async ABI rules out host-side histogramming, so section 4's
  "dispatcher picks per batch histogram" pillar resolves to device-side
  binning or nothing; M1 ships without a dispatch heuristic).

Settled: test framework and oracle vendoring (section 5, via the #4 design
review); dev-container image and CI toolchain pinning (issues #1/#3 —
digest-pinned `nvidia/cuda` 12.6.2 in CI and the local gate); the LZ4
kernel decomposition (section 9, via the #6 design panel — single-pass
warp-per-chunk, with a recorded measurement-based falsification trigger;
the exact batch upper bound is pinned by the zero-visible-devices
technique from the #4 harness once the geometry lands:
`CUDEC_ERR_CUDA` = passed validation and reached the launch,
`CUDEC_ERR_INVALID_ARGUMENT` = rejected, no constant published).

## 9. M1 kernel design (settled via the #6 design panel, 2026-07-17)

Three candidate designs (single-pass warp-cooperative, two-phase
scan-then-copy, and a simplest-that-saturates challenger) were developed
independently and scored by two judges with independent bandwidth and
occupancy arithmetic. The convergent result:

**One kernel. One warp per chunk over a grid-stride loop. All 32 lanes
parse the sequence stream redundantly in lockstep** (identical bytes,
identical arithmetic, identical registers in every lane — no leader lane,
no shuffles, no parse divergence), **and fan out by lane for every copy.
No shared memory, no sequence table, no dispatch heuristic.**

Why the arithmetic forces this shape:

- The kernel is parse-bound, not bandwidth-bound. The Silesia-shaped batch
  moves ~450–550 MB per run (src read + dst write + match-gather misses
  against 5 MB L2) — a ~300–380 GB/s bandwidth ceiling — while the serial
  per-chunk parse chain (3–5 dependent L1 loads per sequence, ~3,300
  sequences per 64 KiB chunk) bounds a naive kernel to ~100–200 GB/s.
  Parallelism therefore comes from chunks: 3,239 Silesia chunks against
  3,264 resident warps on the RTX 3080. Warp-per-chunk exposes 32
  concurrent parse streams per SM; a block-per-chunk two-phase design
  exposes 4, starving the binding resource to accelerate copies that were
  never the bottleneck.
- Table-in-smem two-phase is dead on arithmetic alone, recorded here so
  nobody re-proposes it: a 64 KiB chunk admits up to 16,385 sequences;
  at 16 B per table entry that is ~256 KiB — 2.6× an SM's usable shared
  memory.

**Overlap copy, closed form.** An overlapping match is not a copy that
chases itself; it is a modular gather from bytes already final:
`dst[d + i] = dst[d - off + (i mod off)]` reads only `[d - off, d)`, which
the `__syncwarp()` preceding every copy has frozen. Each output byte is
written exactly once, by a statically determined lane, as a pure function
of lower addresses — deterministic because no ordering exists to get
wrong. The inner loop tracks the modulus incrementally
(`r += step; if (r >= off) r -= off`) so no lane pays a division per
iteration.

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
empirically by oracle parity — whenever liblz4 rejects, cudec rejects; for
accepted mutants the comparison is against the oracle's own output and
size. One crafted negative fixture per ladder branch, its oracle verdict
asserted in-test, keeps every reject path CI-covered.

**Failure contract:** on any reject, `bytes_written = 0` and a non-OK
status; dst contents up to the failure point are unspecified but are never
presented as success. On success, writes touch exactly
`dst[0, bytes_written)`. Check-before-load is batch isolation, not just
parity: on a GPU an out-of-bounds read is not a per-chunk failure — it can
fault the launch and poison the whole batch.

**One deliberate divergence from liblz4 (settled empirically at ladder
step 2):** a match offset of 0 is invalid per the LZ4 block spec, but
liblz4 tolerates it as a defined self-referential copy (its decoder
silences an msan warning there rather than erroring). cudec rejects it —
fail-closed on spec-invalid input (prime directive 1) outranks bug-for-bug
parity with the reference. This makes cudec's accept set a strict subset
of liblz4's: the twin test asserts the two security-critical directions
(where cudec accepts, liblz4 accepts and the bytes match; where liblz4
rejects, cudec rejects) and reports the count of stricter-than-liblz4
cases so the divergence stays visible rather than silent. It is the only
such point.

**Anti-pattern rule (from the two-phase candidate's disproof):** no packed
or narrowed field anywhere in the decoder unless its bound derives from an
ABI-enforced invariant — the 64 KiB chunk cap is a project convention, not
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
out at the kernel PR and recorded — a 24-warp fallback is a measured
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
parse-only ceiling is ~35 GB/s (~10× CPU) — so parse and copy each cost
roughly half the wall time. Since parse-only ceilings any two-phase phase-1
(shared serial parse), two-phase cannot exceed ~35 GB/s; its only lever is
a faster copy, which single-pass optimizes equally without a table or
barrier. Recorded in [docs/BENCHMARKS.md](BENCHMARKS.md).

**Perf pass 1 outcome (issue #16): the designed micro-optimizations were
rejected by measurement** — the incremental-mod gather regressed 16% (its
loop-carried dependency pipelines worse than the independent modulo the
compiler overlaps) and the vectorized literal copy regressed 6% (real
literal runs are too short for the wide path); a `__syncwarp` elision was
neutral. The kernel is at a local optimum for its structure; the bottleneck
is structural (the redundant 32-lane parse ceiling and latency-bound
short-run copies), so raising throughput needs higher occupancy or
warp-specialization — larger than a micro-op pass, tracked as issue #21. The
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

**Known limits, published:** the redundant-parse family ceiling is roughly
250–400 GB/s after perf passes — deliberately accepted under "formats over
percentage points"; batches under ~2,000 chunks underfill the machine and
land near CPU speed (documented, not hidden); warp-synchronous discipline
is load-bearing — every `__syncwarp` is reviewed as such.

**M3/M4 seam:** the chunk decoder is `template<class FormatParser>` —
Snappy (M3) swaps the parser and keeps everything else; GDeflate (M4)
keeps the copy engine, validation posture, and result contract while
bringing its format-native 32-substream parse model.

**The M1 PR ladder** (each independently gated): (1) this design section —
closes the design issue; (2) the sequence parser + validation ladder as a
single-source `__host__ __device__` header with a CPU-compiled twin test
running the full mutant corpus and the crafted negatives on the GPU-less
CI runner — the security heart of M1 lands under CI before any kernel;
(3) the kernel, minimal-correct, behind the frozen batch ABI — the stub
deleted, the gpu_fixture expectations flip, all mechanical gates recorded
(security-review gated: decoder validation is this project's login path);
(4) the bench GPU path + the split variants + the first recorded GPU
baselines — numbers and kernel never move in the same PR; (5) a measured
perf pass (register-window parse staging, vectorized multi-byte lane
copies), accepted only on recorded improvement with all gates green.
