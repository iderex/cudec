# Benchmark methodology

The performance write-up for cudec's decode paths - LZ4 throughout, and the
GDeflate rows added under issue #228: what is measured, on
what hardware, how to reproduce it, and how to read the numbers honestly.
The raw baseline record - every number with the full methodology block the
harness emits - lives in [BENCHMARKS.md](BENCHMARKS.md); this page is the
narrative around those recorded numbers and adds nothing that was not
measured. A performance claim without its measurement is not a claim
([MASTERPLAN](MASTERPLAN.md) section 5).

## What cudec is, and what it is measured against

cudec is an open-source, decode-only GPU decompression library for the
standard formats. NVIDIA's nvCOMP is the proprietary incumbent in this space;
cudec exists to be the auditable, fail-closed, fuzz-tested open alternative,
not a cheaper one (nvCOMP is free to use). This document therefore measures
cudec against two references that can be published in full:

- **The CPU oracle** - `LZ4_decompress_safe` from liblz4 1.10.0, single
  thread. This is the same reference implementation cudec is diff-tested
  against, so it is both the correctness oracle and the throughput baseline.
- **cudec's own internal ceilings** - the parse-only ceiling and the
  ~760 GB/s output-bandwidth ceiling of the GPU - which bound what any decode
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
(SHA-256, fail-closed on any mismatch) and never committed - the supply-chain
rule applied to test data. The harness splits each input file into 64 KB
(65536-byte) chunks and compresses each chunk in-harness with
`LZ4_compress_default`, so the batch is many independent chunks, which is the
workload the GPU path is built for.

- **Silesia** - the standard 12-file mixed corpus (dickens, mozilla, mr, nci,
  ooffice, osdb, reymont, samba, sao, webster, x-ray, xml). 3239 chunks,
  211.94 MB original, 102.44 MB compressed (ratio 0.483). Chunk sizes:
  min 8066 / median 65536 / max 65536 bytes. This is the representative
  average-case number.
- **worst-4Bmatch** - an adversarial-but-valid corpus constructed in-harness
  (`--worst4b`): back-to-back minimum matches (length 4, offset 1), the
  maximum sequence density a valid LZ4 block can carry. 3200 identical 64 KB
  chunks (~210 MB). The standard compressor never emits this shape (it would
  extend the offset-1 run into one long match), so the harness builds the
  block directly and the oracle validates it (round-trips through
  `LZ4_decompress_safe`) before any timing. This is the throughput worst case.
- **asset-like** - a game-asset-like corpus generated in-harness
  (`--assetlike`): one 64 KB block tiled by four regions in the proportions of
  a shipped asset package - BC1-shaped block-compressed texture, interleaved
  32-byte vertex records, a triangle-list index buffer, and 16-bit stereo PCM
  audio - replicated to 3200 chunks (~210 MB), the same scale as the two rows
  above so the throughput numbers are directly comparable. Unlike them the
  wire is ordinary `LZ4_compress_default` output, because the shape being
  modelled is the source data and letting the standard compressor decide the
  sequence structure is the point. The oracle round-trips the block before any
  timing and the harness locks four quantities that name the regime -
  compression ratio, literal share, sequence count and mean match length - so
  a generator that drifts out of it reds CI. **It is a MODEL of the workload,
  not the workload**: a synthetic texture/mesh/audio mixture reproduces the
  regime - longer matches, a higher incompressible share, a different sequence
  density than any other corpus here - and is not a substitute for a
  measurement on real game data. No number taken on it may be quoted as one.
- **enwik8** - the first 100 MB of an English Wikipedia dump, so pure
  marked-up text rather than a mixture. 1526 chunks, 100.00 MB original,
  57.00 MB compressed (ratio 0.570). Chunk sizes: min 57600 / median 65536 /
  max 65536 bytes. Recorded 2026-08-05, later than the rows above and on the
  same platform. This is the text-heavy case, and it is kept because it does
  not reproduce the Silesia average.

## Measured results

All figures are p50 of 30 runs, recorded 2026-07-17 on the platform above,
except the enwik8 table, which was recorded 2026-08-05 on the same platform
and says so again where it sits. The full per-report methodology blocks are in
[BENCHMARKS.md](BENCHMARKS.md); this is the summary.

### Silesia (average case)

| Path                                   | Throughput | Relative      |
| -------------------------------------- | ---------- | ------------- |
| CPU oracle, liblz4 single thread       | 3.41 GB/s  | 1× (baseline) |
| GPU decode, device-resident (cudec)    | 18.1 GB/s  | ~5.3× CPU     |
| GPU parse-only ceiling (copies elided) | 34.6 GB/s  | ~10× CPU      |

The device-resident kernel decodes Silesia at **~18 GB/s**, roughly 5× the
single-thread CPU baseline. The **parse-only ceiling is ~35 GB/s** (~10× CPU):
the serial per-lane parse and the match/literal copies each cost roughly half
the wall time. Both numbers sit well under the ~760 GB/s output-bandwidth
ceiling - LZ4 decode on this design is bound by the redundant lockstep parse,
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

- The degradation is uniform and bounded at roughly 2.2–2.3× across every
  path. CPU, GPU decode, and GPU parse-only all slow by the same factor. That
  factor is the sequence density, one sequence per 4 decoded bytes here
  against Silesia's longer average matches: the cost is linear in the number
  of sequences, which is the redundant parse the design accepts. There is no
  super-linear blow-up, and the measured factor is below the ~4× the issue
  pessimistically estimated.
- **No size amplification.** The block barely compresses (ratio 0.750,
  ~1.33× expansion) and each chunk decodes to exactly 65536 bytes, capped by
  the caller's destination capacity. For LZ4 the throughput worst case and a
  decompression bomb are mutually exclusive, because a bomb is one long match
  and therefore a single fast sequence, which is the opposite shape. cudec's
  fixed per-chunk output cap fail-closes the size axis regardless.
- **The GPU advantage holds under the worst input:** 8.1 GB/s worst-case GPU
  is still ~5.4× the CPU worst case and ~2.3× the CPU's Silesia _average_.

### asset-like (generated game-asset model)

Recorded 2026-08-09, same platform and same container digest as the tables
above. `bench_lz4 --assetlike --gpu --warmup 3 --runs 30`. The corpus is a
MODEL of the workload and not the workload; nothing here is a measurement on
real game data.

| Path                                   | Throughput | Share of the decode |
| -------------------------------------- | ---------- | ------------------- |
| CPU oracle, liblz4 single thread       | 10.03 GB/s | -                   |
| GPU decode, device-resident (cudec)    | 44.2 GB/s  | 4.748 ms            |
| GPU parse-only ceiling (copies elided) | 81.5 GB/s  | 2.574 ms, 54%       |

The right-hand column is the split rather than a comparison against Silesia,
and that is deliberate: the Silesia rows above predate the gather narrowing
(issue #58), so a ratio against them would mix two kernels. Within this
invocation, copies are 2.174 ms of the 4.748 ms decode, and the GPU runs
~4.4× the CPU oracle - a smaller factor than Silesia's, on an input where the
oracle also has little to do. Where the regime sits against the other corpora
on today's kernel is not established by this row and needs them re-measured in
one session.

### enwik8 (text-heavy)

Recorded 2026-08-05, same platform and same container digest as the tables
above. `bench_lz4 bench/corpora/enwik8 --gpu --warmup 3 --runs 30`.

| Path                                   | Throughput | vs. its Silesia row |
| -------------------------------------- | ---------- | ------------------- |
| CPU oracle, liblz4 single thread       | 2.96 GB/s  | ~1.15× slower       |
| GPU decode, device-resident (cudec)    | 11.1 GB/s  | ~1.6× slower        |
| GPU parse-only ceiling (copies elided) | 21.8 GB/s  | ~1.6× slower        |

The corpus is kept pinned because of the right-hand column: the two GPU paths
slow by roughly 1.6× while the CPU path barely moves, so the GPU-over-CPU
factor falls from ~5.3× on Silesia to ~3.8× here. A corpus that reproduced
Silesia would have earned its removal instead.

Why it goes that way is not established by these numbers. enwik8 compresses
worse than Silesia (ratio 0.570 against 0.483), and the cost model recorded
for this kernel is linear in the number of sequences rather than in output
bytes, which predicts this direction; the harness reports no sequence count,
so nothing here measured it.

### Streaming, end-to-end (M2 reusable context)

The device-resident rows above exclude transfers; the streaming path
(`cudec_stream_ctx_create` / `cudec_lz4_decompress_stream_ctx` /
`cudec_stream_ctx_destroy`) measures the whole synchronous decode call -
pinned staging, host-to-device copy, decode, and, for host output, the
device-to-host readback - wall-clocked. On Silesia, reusing a warmed context:

| Output target            | Steady-state (reused ctx) | Cold (fresh ctx) |
| ------------------------ | ------------------------- | ---------------- |
| Device out               | ~7.65 GB/s (27.7 ms)      | ~1.39 GB/s       |
| Host out (sync readback) | ~1.36 GB/s (155.4 ms)     | ~0.76 GB/s       |

Recorded 2026-08-09, after the wave sizing landed (issue #33); the rows it
replaced read 229.5 ms and 365.2 ms at a fixed 64-chunk wave. This is a
different, honest metric - not a regression of the ~18 GB/s device-resident
kernel throughput. Read it plainly:

- **The streaming wall was submission-bound, not compute-bound, and that is
  what was fixed.** Against a ~12 ms device-resident decode and a ~4 ms
  compressed host-to-device copy, the old ~230 ms steady-state wall was
  dominated by the **per-wave serial submission** of the batch (Silesia was
  ~51 waves of 64 chunks) on this WSL2/WDDM path, where each submission flush
  costs milliseconds. Sizing the wave by its staging cost instead
  ([issue #33](https://github.com/iderex/cudec/issues/33)) makes Silesia one
  submission on the device path and brings the wall to 27.7 ms, an 8.5×
  improvement won in 5 of 5 alternated passes. Peak staging is bounded at
  384 MiB by construction, which is the price and is stated in
  [BENCHMARKS.md](BENCHMARKS.md) beside the table.
- **Host output is still ~5× the device path**, and not because of the wave:
  its readback targets pageable caller memory one chunk at a time, so it is
  3239 synchronous copies at any wave size. That is issues #133 and #135.
- **The per-call allocation is not the dominant cost** (~8 ms cold − steady),
  correcting the earlier assumption; the reusable context earns its place as a
  simplification and as the primitive #33 builds on, not as a speedup.
- **Copy/decode overlap is weak for LZ4 and was deliberately dropped.** LZ4's
  ~2:1 ratio keeps the compressed input small (~4 ms) relative to decode
  (~12 ms), so overlapping input transfer against the kernel saves ~4 ms at
  best (~25% ceiling) and was never worth its complexity. The genuine overlap
  lever for the higher-ratio formats to come (Zstd, GDeflate) is overlapping
  decode against the decoded-**output** readback, not the input - a change
  for those milestones, with its own test when it lands.

### GDeflate (M4, one warp per 64 KiB page)

The first recorded device numbers for the GDeflate path, on the same platform
and the same protocol as the rows above. The full blocks are in
[BENCHMARKS.md](BENCHMARKS.md) under "M4: the first recorded GDeflate GPU
baselines"; this is the summary and it adds nothing that was not measured.

| Corpus, level             | Ratio  | CPU oracle | GPU decode   | Relative |
| ------------------------- | ------ | ---------- | ------------ | -------- |
| Silesia, level 6          | 0.3343 | 0.327 GB/s | 8.056 GB/s   | ~24.7×   |
| Silesia, level 0 (stored) | 1.0021 | 0.660 GB/s | 106.412 GB/s | ~161×    |
| asset-like, level 6       | 0.7029 | 0.275 GB/s | 5.930 GB/s   | ~21.6×   |
| worst-headers             | 2.8103 | 0.055 GB/s | 0.497 GB/s   | ~9.0×    |

**Read the level-0 row as the copy path and not as GDeflate.** The reference
emits uncompressed blocks by construction at level 0, so a page is a memcpy
with a header; 106 GB/s at a ratio of 1.0021 is the ceiling the copy engine
imposes on this kernel and says nothing about entropy decoding.

**The working number is the level-6 row, 8.06 GB/s at a ratio of 0.334.**
Against the LZ4 rows above that is roughly half the throughput at roughly half
the size, which is the trade the format exists for; the two are not comparable
as a single figure and no ratio between them is quoted here.

**`worst-headers` is the margin, at 0.497 GB/s.** That corpus emits one
dynamic block per group of matches, so a page pays a whole header decode for a
handful of decoded bytes and the kernel spends its time in table construction
rather than in the round loop. It is two orders of magnitude below the
headline row and it is the number a denial-of-service argument is made from.

**There is no parse-only ceiling for this path**, and its absence is a result.
The LZ4 and Snappy rows get one from a template flag that elides the copies
while running the identical parse; the GDeflate round loop's next round reads
bytes the previous round's copies produced, so a variant with the copies
elided would decode different symbols and ceiling nothing. Where the time
actually goes inside this kernel is therefore not settled by these numbers,
and the layout question that would move it is issue #204.

**No perf conclusion is drawn about the kernel's shape from this run.** The
achieved occupancy of the kernel these numbers were taken on is 24 of 48 warps
per SM on sm_86, recorded on #214 against the design panel's prediction of 32;
whether that costs throughput is a measurement nobody has taken.

## The design rationale (single-pass)

cudec's LZ4 kernel is a single-pass, warp-cooperative design: each chunk is
decoded by a warp running a redundant 32-lane lockstep parse of the sequence
stream, with a closed-form modular gather for match bytes. This was chosen
over a two-phase (parse-then-copy) design in the #6 design panel and settled
by the #15 measurement:

- The parse-only number (~35 GB/s on Silesia) ceilings **both** single-pass
  and any two-phase design, because a two-phase phase-1 runs the identical
  serial parse and so cannot exceed it either. Two-phase's only lever is a
  faster phase-2 copy - but single-pass's copy is equally optimizable, with no
  table, no barrier, and no extra memory traffic. The decomposition question
  is therefore settled for single-pass.
- Two subsequent measured perf passes (issues #16 and #21) confirmed the
  kernel is at a local optimum for this workload: the designed
  micro-optimizations (incremental-mod gather, vectorized literal copy,
  `__syncwarp` elision) each regressed or were neutral, and forcing 100%
  occupancy regressed the decode ~5% because the only way to reach it under
  the fail-closed 64-bit-arithmetic invariant is a register spill. The
  remaining structural lever is a warp-specialization rewrite that abandons
  the redundant-parse invariant - its own design change, deferred behind the
  format ladder ("formats over percentage points").

The full derivations, the profile readout, and the falsification-trigger
verdicts are recorded in [BENCHMARKS.md](BENCHMARKS.md) and
[MASTERPLAN](MASTERPLAN.md) section 9.

## When the GPU wins, and when it does not

The honest framing, consistent with the README: the GPU wins when thousands
of independent chunks decode in parallel - batch asset streaming, analytics
scans, ML data loading, checkpoint restore. The device-resident numbers above
are that case. A single small file on a cold PCIe bus is **not** the GPU's
case: the transfer and per-submission latency dominate (visible directly in
the streaming rows), and the single-thread CPU wins it. This library will not
claim otherwise.

## Reproducing the numbers

Everything needed to reproduce every figure is in the tree. A third party
runs the same measurements - including any nvCOMP comparison - as follows.

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
   | `bench_lz4 --assetlike --gpu`              | the game-asset-like regime (self-generated) |
   | `bench_lz4 --selfcheck`                    | the CI rot check (a few chunks, CPU-only)   |

   `--runs N` / `--warmup N` set the run counts (defaults 30 / 3). Every run
   prints its own methodology block, so a pasted result is self-describing.

The construction of the worst-4Bmatch corpus is oracle-validated in-harness
and locked against rot by the `bench_worst4b_selfcheck` ctest on the GPU-less
runner, so the adversarial number cannot silently drift. The asset-like corpus
is held the same way by `bench_assetlike_selfcheck`, over the four quantities
that define its regime rather than over validity alone.

4. **The GDeflate rows** come out of a second binary with the same flag
   vocabulary, and every one of its corpora is compressed by the pinned
   reference or emitted by the harness itself - never by a second compressor:

   | Invocation                            | What it measures                            |
   | ------------------------------------- | ------------------------------------------- |
   | `bench_gdeflate <files...>`           | CPU oracle denominator over the given files |
   | `bench_gdeflate --gpu <files...>`     | + device-resident GPU decode per level      |
   | `bench_gdeflate --gpu --assetlike`    | the game-asset model, per level             |
   | `bench_gdeflate --gpu --blocktypes`   | the forced-block-type coverage rows         |
   | `bench_gdeflate --gpu --worstrounds`  | the refill-bound adversarial corpus         |
   | `bench_gdeflate --gpu --worstheaders` | the header-bound adversarial corpus         |
   | `bench_gdeflate --selfcheck`          | the CI rot check (CPU-only, digest-locked)  |

   `--gpu` is refused together with `--selfcheck` and with `--blockmix`: the
   first runs on the GPU-less runner and the second times no decode, so in
   both cases the flag would be accepted and do nothing, which reads
   afterwards as a measurement that was taken.

   The device rows print under the same methodology block as the CPU rows of
   the same run, so the two cannot be quoted apart, and the harness refuses to
   time a batch whose pages did not all decode to their original size.

## The nvCOMP comparison (not published)

The one comparison this document deliberately does **not** publish is
cudec-vs-nvCOMP. **§8.9 of the NVIDIA Software License Agreement** - the EULA
governing the nvCOMP binary - restricts the customer from distributing or
disclosing to third parties the results of benchmarking, competitive analysis,
or regression/performance testing of the software without NVIDIA's prior
written permission (a "DeWitt clause"). cudec therefore publishes **no**
cudec-vs-nvCOMP throughput numbers, quotes no nvCOMP figures, and builds no
head-to-head table - by policy, not oversight. nvCOMP is referenced only
nominatively, as the proprietary incumbent this project offers an auditable
open alternative to.

This is why the reproducible harness above matters: a third party can build
cudec and nvCOMP on their own hardware, run both over the same hash-pinned
corpora, and draw their own head-to-head conclusion under their own
acceptance of the nvCOMP EULA. cudec ships the honest, published half - its
own measured numbers and the CPU baseline - and the tooling to complete the
picture privately.

## The hipCOMP comparison (protocol defined, no result)

The comparison the section above cannot publish has a sibling that it can.
hipCOMP is AMD's port of nvCOMP, it is open source, and its LZ4 decompressor
is warp-per-chunk like cudec's - so a run of both on one AMD device isolates
the kernel family instead of the vendor. This section defines that run. It
publishes no number, and there is none to publish: cudec has no HIP backend
yet, and nobody here has an AMD GPU.

The warp-per-chunk claim is read out of the source rather than carried over.
`src/lowlevel/LZ4CompressionKernels.cu` launches its decode with a
two-dimensional block whose first extent is one wavefront and whose second
selects the chunk, over a grid of chunk pairs, and the kernel takes the
wavefront width as a template parameter:

```sh
gh api repos/ROCm/hipCOMP-core/contents/src/lowlevel/LZ4CompressionKernels.cu \
  --jq '.content' | base64 -d | sed -n '215,228p'
```

That is the same shape cudec's LZ4 decoder has and the same shape the HIP
port is being cut to, which is why this comparison is worth the trouble: two
implementations of one strategy on one device. Nothing here is read into
cudec - the file is cited, never borrowed, per the legal guardrails in
[MASTERPLAN](MASTERPLAN.md).

### The licence, checked rather than assumed

Read on 2026-08-21 from `ROCm/hipCOMP-core` at the head of its default branch
`release/rocmds-26.03`:

```sh
gh api repos/ROCm/hipCOMP-core \
  --jq '{license: .license.spdx_id, default_branch, archived}'
{"archived":false,"default_branch":"release/rocmds-26.03","license":"MIT"}

gh api repos/ROCm/hipCOMP-core/commits/release/rocmds-26.03 \
  --jq '{sha, date: .commit.committer.date}'
{"date":"2026-07-07T23:41:56Z","sha":"22cc762f54fba7cdfca74a4c50c00f2aac4ace7a"}
```

Two licences apply, not one, and the second is the one worth reading. The
repository's own `LICENSE` is the MIT licence, copyright Advanced Micro
Devices. `NOTICES.md` then names a long list of files - the public headers and
much of `src/`, including the LZ4 and Snappy paths - as derived from NVIDIA
nvCOMP `branch-2.2` and governed by that project's 3-Clause BSD licence, which
the repository carries verbatim as `NVCOMP_2_2_LICENSE`.

**Neither licence restricts publishing benchmark results.** MIT grants use
without restriction subject only to the notice condition. BSD-3-Clause adds
two redistribution conditions and one more term, and that third term is about
endorsement rather than measurement: it forbids using NVIDIA's name or its
contributors' names to endorse or promote products _derived from that
software_. cudec is not derived from it - no line of nvCOMP or hipCOMP is read
or copied here, which is the standing rule in
[MASTERPLAN](MASTERPLAN.md) - so the term does not reach cudec, and it would
not restrict a measurement even if it did. There is no clause anywhere in
either file resembling §8.9 of the NVIDIA Software License Agreement.

So the finding is: **publishing a cudec-vs-hipCOMP comparison is
unencumbered**, and the reason is the licences, not the absence of a search
for one. Anyone re-checking should re-run the two commands above rather than
trust this paragraph: a licence is a fact about a repository at a commit, and
the commit moves.

### What must be said wherever a hipCOMP number appears

hipCOMP says two things about itself that a fair comparison repeats. Its
README leads with a caution that the release is an early-access software
technology preview and that production workloads are not recommended, and its
own algorithm table marks LZ4 and Snappy as supported and **not optimized**.
A throughput table that omits either turns a preview into a straw man.

That framing is a condition of publishing, not a caveat added afterwards: a
faster number against an unoptimised preview is evidence about that preview
and not about the two kernel families, and a comparison presented without it
is arguing something it did not measure.

### The protocol

Written so a reporter with an AMD device can execute it from this section
alone and produce two report blocks that are comparable. Nothing below has
been run.

1. **One device, one container, one session.** Both libraries are built and
   run inside the same ROCm container image, pinned by `sha256` digest, on the
   same GPU, in one sitting. Two runs on the same model of card on different
   days are not a same-hardware comparison. The image is the one the HIP CI
   job pins, and this section takes it from there rather than inventing a
   second one - read the digest out of `.github/workflows/ci.yml` at the
   commit you are running, because a digest restated in prose here is a second
   pin the moment the job moves. `docs/AMD-VALIDATION.md` does write it out,
   because a runbook has to be copy-pasteable; it is the same digest, taken
   from the same line.

2. **The same corpora, hash-pinned.** `sh bench/get-corpora.sh`, which refuses
   any file whose SHA-256 does not match its pin. Both sides read the same
   files from the same directory. A corpus either side assembled for itself is
   not the same corpus.

3. **The same chunk-size distribution.** cudec decodes independent chunks and
   so does hipCOMP; throughput depends on how the input was cut before either
   library saw it. The reporter states the chunk size used and applies one
   value to both sides in a run.

4. **Device-resident decode, host transfers excluded, event timing.** The
   region measured is the decode itself, with the compressed input already on
   the device and the output staying there, timed with device events - the
   same region and the same instrument as every number in this document.
   Including a host copy on one side and not the other is the easiest way to
   publish a wrong result honestly.

5. **Three warmup runs and thirty measured runs**, per side, which is what
   `--warmup 3 --runs 30` already does for `bench_lz4` and what the reporter
   configures on the hipCOMP side to match.

6. **Both report blocks pasted whole.** `bench_lz4` prints its own methodology
   block with every run, so a pasted cudec block is self-describing; the
   hipCOMP side is pasted with its build command, its ROCm version and its
   `gfx` target beside it. An edited or summarised block is not a result. Any
   number that reaches [BENCHMARKS.md](BENCHMARKS.md) arrives through the
   recording format that document defines, with the early-access framing above
   attached to it.

### What this section deliberately does not do

It states no result and implies none. It does not say which library is
expected to be faster, on which architecture, or by how much - there is no
cudec HIP backend to run and no AMD device here to run it on, so any such
sentence would be a guess wearing the clothes of a measurement. The first
comparison to fill this in will be somebody else's, run on their hardware,
under the protocol above.
