# Benchmark methodology

The performance write-up for cudec's LZ4 decode path: what is measured, on
what hardware, how to reproduce it, and how to read the numbers honestly.
The raw baseline record — every number with the full methodology block the
harness emits — lives in [BENCHMARKS.md](BENCHMARKS.md); this page is the
narrative around those recorded numbers and adds nothing that was not
measured. A performance claim without its measurement is not a claim
([MASTERPLAN](MASTERPLAN.md) section 5).

## What cudec is, and what it is measured against

cudec is an open-source, decode-only GPU decompression library for the
standard formats. NVIDIA's nvCOMP is the proprietary incumbent in this space;
cudec exists to be the auditable, fail-closed, fuzz-tested open alternative,
not a cheaper one (nvCOMP is free to use). This document therefore measures
cudec against two references that can be published in full:

- **The CPU oracle** — `LZ4_decompress_safe` from liblz4 1.10.0, single
  thread. This is the same reference implementation cudec is diff-tested
  against, so it is both the correctness oracle and the throughput baseline.
- **cudec's own internal ceilings** — the parse-only ceiling and the
  ~760 GB/s output-bandwidth ceiling of the GPU — which bound what any decode
  on this hardware can achieve.

No cudec-vs-nvCOMP performance numbers appear here or anywhere in the repo;
see [The nvCOMP comparison](#the-nvcomp-comparison-not-published) below for
why, and how to run one yourself.

## Test platform

Every number in [BENCHMARKS.md](BENCHMARKS.md) was recorded 2026-07-17 in the
digest-pinned dev container on one machine:

- **GPU:** NVIDIA GeForce RTX 3080 (GA102, `sm_86`), 10 GB GDDR6X,
  ~760 GB/s output-bandwidth ceiling, driver 560.94 (CUDA driver API 12.6).
- **Host CPU (for the liblz4 baseline):** AMD Ryzen 9 5950X, 16-core.
- **Toolchain:** the pinned container
  `nvidia/cuda:12.6.2-devel-ubuntu24.04`
  (digest `sha256:738fba0fbdb225b7a2931c58a5c8f03a84d3cd2f6a84975826a157339ef750b8`),
  nvcc 12.6.77, CUDA runtime 12.6, built `-arch=sm_86`.
- **Host OS path:** Windows 11 host, GPU reached through Docker Desktop's
  WSL2 backend with `--gpus all`. The streaming end-to-end numbers are
  measured on this WSL2/WDDM submission path, which matters for how they are
  read (below).

The device-resident GPU rows time only the decode kernel with the data
already resident on the GPU (host-to-device and device-to-host transfers
excluded), CUDA-event timed. GPU timing jitters ~1–2% run to run. CPU rows are
wall-clock timed around `LZ4_decompress_safe` alone (no allocation, no
clears). Every reported figure is the median (p50) of 30 measured runs after
3 warmup runs; output is byte-verified against the oracle before any timing.

## Corpora

Fetched hash-pinned by [`bench/get-corpora.sh`](../bench/get-corpora.sh)
(SHA-256, fail-closed on any mismatch) and never committed — the supply-chain
rule applied to test data. The harness splits each input file into 64 KB
(65536-byte) chunks and compresses each chunk in-harness with
`LZ4_compress_default`, so the batch is many independent chunks, which is the
workload the GPU path is built for.

- **Silesia** — the standard 12-file mixed corpus (dickens, mozilla, mr, nci,
  ooffice, osdb, reymont, samba, sao, webster, x-ray, xml). 3239 chunks,
  211.94 MB original, 102.44 MB compressed (ratio 0.483). Chunk sizes:
  min 8066 / median 65536 / max 65536 bytes. This is the representative
  average-case number.
- **worst-4Bmatch** — an adversarial-but-valid corpus constructed in-harness
  (`--worst4b`): back-to-back minimum matches (length 4, offset 1), the
  maximum sequence density a valid LZ4 block can carry. 3200 identical 64 KB
  chunks (~210 MB). The standard compressor never emits this shape (it would
  extend the offset-1 run into one long match), so the harness builds the
  block directly and the oracle validates it (round-trips through
  `LZ4_decompress_safe`) before any timing. This is the throughput worst case.
- **enwik8** — fetched and hash-pinned by the harness for completeness, but
  **not yet recorded with a GPU row**; when an enwik8 GPU number is published
  it will carry its own full methodology block. It is listed here so the
  corpus set is stated honestly, not to imply a measurement that does not
  exist.

## Measured results

All figures are p50 of 30 runs, recorded 2026-07-17 on the platform above.
The full per-report methodology blocks are in
[BENCHMARKS.md](BENCHMARKS.md); this is the summary.

### Silesia (average case)

| Path                                   | Throughput | Relative      |
| -------------------------------------- | ---------- | ------------- |
| CPU oracle, liblz4 single thread       | 3.46 GB/s  | 1× (baseline) |
| GPU decode, device-resident (cudec)    | 18.1 GB/s  | ~5.2× CPU     |
| GPU parse-only ceiling (copies elided) | 34.6 GB/s  | ~10× CPU      |

The device-resident kernel decodes Silesia at **~18 GB/s**, roughly 5× the
single-thread CPU baseline. The **parse-only ceiling is ~35 GB/s** (~10× CPU):
the serial per-lane parse and the match/literal copies each cost roughly half
the wall time. Both numbers sit well under the ~760 GB/s output-bandwidth
ceiling — LZ4 decode on this design is bound by the redundant lockstep parse,
not by memory bandwidth. Why that is a deliberate design choice, not a defect,
is covered in [The design rationale](#the-design-rationale-single-pass).

### worst-4Bmatch (adversarial worst case)

| Path                                   | Throughput | vs. its Silesia row |
| -------------------------------------- | ---------- | ------------------- |
| CPU oracle, liblz4 single thread       | 1.49 GB/s  | ~2.3× slower        |
| GPU decode, device-resident (cudec)    | 8.1 GB/s   | ~2.2× slower        |
| GPU parse-only ceiling (copies elided) | 15.3 GB/s  | ~2.3× slower        |

The worst case matters because it is the security-posture number: it is what
an attacker who fully controls a valid input can force. The honest findings:

- **The degradation is uniform and bounded, ~2.2–2.3× across every path** —
  CPU, GPU decode, and GPU parse-only all slow by the same factor. That factor
  is the sequence density (one sequence per 4 decoded bytes here, versus
  Silesia's longer average matches): the cost is linear in the number of
  sequences, exactly the redundant parse the design accepts. There is no
  super-linear blow-up, and this is below the ~4× the issue pessimistically
  estimated.
- **No size amplification.** The block barely compresses (ratio 0.750,
  ~1.33× expansion) and each chunk decodes to exactly 65536 bytes, capped by
  the caller's destination capacity. The throughput worst case and a
  decompression bomb are mutually exclusive for LZ4 — a bomb is one long
  match (a single fast sequence), the opposite shape — and cudec's fixed
  per-chunk output cap fail-closes the size axis regardless.
- **The GPU advantage holds under the worst input:** 8.1 GB/s worst-case GPU
  is still ~5.4× the CPU worst case and ~2.3× the CPU's Silesia _average_.

### Streaming, end-to-end (M2 reusable context)

The device-resident rows above exclude transfers; the streaming path
(`cudec_stream_ctx_create` / `cudec_lz4_decompress_stream_ctx` /
`cudec_stream_ctx_destroy`) measures the whole synchronous decode call —
pinned staging, host-to-device copy, decode, and, for host output, the
device-to-host readback — wall-clocked. On Silesia, reusing a warmed context:

| Output target            | Steady-state (reused ctx) | Cold (fresh ctx) |
| ------------------------ | ------------------------- | ---------------- |
| Device out               | ~0.92 GB/s (229.5 ms)     | ~0.89 GB/s       |
| Host out (sync readback) | ~0.58 GB/s (365.2 ms)     | ~0.57 GB/s       |

This is a different, honest metric — not a regression of the ~18 GB/s
device-resident kernel throughput. Read it plainly:

- **The streaming wall is submission-bound, not compute-bound.** Against a
  ~12 ms device-resident decode and a ~4 ms compressed host-to-device copy,
  the ~230 ms steady-state wall is dominated by the **per-wave serial
  submission** of the batch (Silesia is ~51 waves of 64 chunks) on this
  WSL2/WDDM path, where each submission flush costs milliseconds. Raising the
  wave granularity so the path submits once is the real lever, tracked as
  [issue #33](https://github.com/iderex/cudec/issues/33).
- **The per-call allocation is not the dominant cost** (~8 ms cold − steady),
  correcting the earlier assumption; the reusable context earns its place as a
  simplification and as the primitive #33 builds on, not as a speedup.
- **Copy/decode overlap is weak for LZ4 and was deliberately dropped.** LZ4's
  ~2:1 ratio keeps the compressed input small (~4 ms) relative to decode
  (~12 ms), so overlapping input transfer against the kernel saves ~4 ms at
  best (~25% ceiling) and was never worth its complexity. The genuine overlap
  lever for the higher-ratio formats to come (Zstd, GDeflate) is overlapping
  decode against the decoded-**output** readback, not the input — a change
  for those milestones, with its own test when it lands.

## The design rationale (single-pass)

cudec's LZ4 kernel is a single-pass, warp-cooperative design: each chunk is
decoded by a warp running a redundant 32-lane lockstep parse of the sequence
stream, with a closed-form modular gather for match bytes. This was chosen
over a two-phase (parse-then-copy) design in the #6 design panel and settled
by the #15 measurement:

- The parse-only number (~35 GB/s on Silesia) ceilings **both** single-pass
  and any two-phase design, because a two-phase phase-1 runs the identical
  serial parse and so cannot exceed it either. Two-phase's only lever is a
  faster phase-2 copy — but single-pass's copy is equally optimizable, with no
  table, no barrier, and no extra memory traffic. The decomposition question
  is therefore settled for single-pass.
- Two subsequent measured perf passes (issues #16 and #21) confirmed the
  kernel is at a local optimum for this workload: the designed
  micro-optimizations (incremental-mod gather, vectorized literal copy,
  `__syncwarp` elision) each regressed or were neutral, and forcing 100%
  occupancy regressed the decode ~5% because the only way to reach it under
  the fail-closed 64-bit-arithmetic invariant is a register spill. The
  remaining structural lever is a warp-specialization rewrite that abandons
  the redundant-parse invariant — its own design change, deferred behind the
  format ladder ("formats over percentage points").

The full derivations, the profile readout, and the falsification-trigger
verdicts are recorded in [BENCHMARKS.md](BENCHMARKS.md) and
[MASTERPLAN](MASTERPLAN.md) section 9.

## When the GPU wins, and when it does not

The honest framing, consistent with the README: the GPU wins when thousands
of independent chunks decode in parallel — batch asset streaming, analytics
scans, ML data loading, checkpoint restore. The device-resident numbers above
are that case. A single small file on a cold PCIe bus is **not** the GPU's
case: the transfer and per-submission latency dominate (visible directly in
the streaming rows), and the single-thread CPU wins it. This library will not
claim otherwise.

## Reproducing the numbers

Everything needed to reproduce every figure is in the tree. A third party
runs the same measurements — including any nvCOMP comparison — as follows.

1. **Fetch the corpora** (hash-pinned, fail-closed; needs `curl` and
   `unzip`):

   ```sh
   sh bench/get-corpora.sh
   ```

   This writes the verified corpora under `bench/corpora/` (git-ignored) and
   refuses any file whose SHA-256 does not match the pin.

2. **Build and run inside the pinned container** (from the repo root; on a
   Windows host run this via PowerShell, not Git Bash, so the `-v` volume
   spec is not path-mangled):

   ```sh
   docker run --rm --gpus all \
     -v "$PWD:/w" -w /w \
     nvidia/cuda:12.6.2-devel-ubuntu24.04 \
     sh -c "apt-get update -q >/dev/null && \
            apt-get install -yq cmake >/dev/null 2>&1 && \
            cmake -B build-cuda -DCUDEC_ENABLE_CUDA=ON && \
            cmake --build build-cuda -j && \
            ./build-cuda/bench/bench_lz4 --gpu bench/corpora/silesia/*"
   ```

3. **Select the regime** with `bench_lz4` flags:

   | Invocation                                 | What it measures                            |
   | ------------------------------------------ | ------------------------------------------- |
   | `bench_lz4 <files...>`                     | CPU oracle baseline over the given corpus   |
   | `bench_lz4 --gpu <files...>`               | + device-resident GPU decode and parse-only |
   | `bench_lz4 --gpu --gpu-stream-ctx <files>` | + streaming end-to-end (reusable context)   |
   | `bench_lz4 --worst4b --gpu`                | the adversarial worst case (self-generated) |
   | `bench_lz4 --selfcheck`                    | the CI rot check (a few chunks, CPU-only)   |

   `--runs N` / `--warmup N` set the run counts (defaults 30 / 3). Every run
   prints its own methodology block, so a pasted result is self-describing.

The construction of the worst-4Bmatch corpus is oracle-validated in-harness
and locked against rot by the `bench_worst4b_selfcheck` ctest on the GPU-less
runner, so the adversarial number cannot silently drift.

## The nvCOMP comparison (not published)

The one comparison this document deliberately does **not** publish is
cudec-vs-nvCOMP. **§8.9 of the NVIDIA Software License Agreement** — the EULA
governing the nvCOMP binary — restricts the customer from distributing or
disclosing to third parties the results of benchmarking, competitive analysis,
or regression/performance testing of the software without NVIDIA's prior
written permission (a "DeWitt clause"). cudec therefore publishes **no**
cudec-vs-nvCOMP throughput numbers, quotes no nvCOMP figures, and builds no
head-to-head table — by policy, not oversight. nvCOMP is referenced only
nominatively, as the proprietary incumbent this project offers an auditable
open alternative to.

This is why the reproducible harness above matters: a third party can build
cudec and nvCOMP on their own hardware, run both over the same hash-pinned
corpora, and draw their own head-to-head conclusion under their own
acceptance of the nvCOMP EULA. cudec ships the honest, published half — its
own measured numbers and the CPU baseline — and the tooling to complete the
picture privately.
