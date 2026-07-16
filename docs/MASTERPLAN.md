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
  conformance tests so every PR is checked mechanically.
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

- Exact LZ4 kernel decomposition: single-pass warp-cooperative decode vs.
  two-phase (sequence scan, then parallel copy) — decided with measurements
  in the M1 design issue.
- Test framework choice (GoogleTest vs. Catch2) and oracle vendoring
  mechanism (FetchContent pinned to release hashes) — M0.
- CUDA dev container image and CI toolchain pinning — M0.
- Benchmark corpus set beyond Silesia/enwik (game-asset-like data) — M2.
