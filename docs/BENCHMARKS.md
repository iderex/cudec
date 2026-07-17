# Benchmarks

The baseline record. Every entry carries the full methodology block as
emitted by the harness (`bench/bench_lz4`, `--gpu` for the device path) — a
number without its methodology cannot be produced, by construction.
Regressions against the recorded baselines block merges unless explicitly
justified ([MASTERPLAN](MASTERPLAN.md) section 5). Corpora are fetched
hash-pinned via `bench/get-corpora.sh` and never committed.

## M1: first GPU decode (Silesia)

The first GPU decode numbers, recorded 2026-07-17 inside the digest-pinned
dev container (`nvidia/cuda:12.6.2-devel-ubuntu24.04`) on the RTX 3080
(sm_86, ~760 GB/s output-bandwidth ceiling). The GPU rows are
device-resident (data already on the GPU; H2D/D2H excluded), CUDA-event
timed. GPU timing jitters ~1–2% run to run.

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3239 chunks, 211.94 MB original, 102.44 MB compressed (ratio 0.483), compressed in-harness via LZ4_compress_default
- chunk sizes: min 8066 / median 65536 / max 65536 bytes
- method: 3 warmup + 30 measured runs; CPU wall clock per whole-batch decode; GPU device-resident, CUDA-event timed; output byte-verified before timing
- CPU decode throughput (liblz4 baseline): p50 3.46 GB/s
- GPU decode (cudec, device-resident): p50 11.7 ms, 18.1 GB/s (~5.2x the CPU baseline)
- GPU parse-only ceiling (copies elided): p50 6.1 ms, 34.6 GB/s
```

### Decision rule (masterplan section 9): the decomposition question, settled

The minimal-correct kernel decodes Silesia at **~18 GB/s** (~5x the
single-thread CPU baseline); the **parse-only ceiling is ~35 GB/s** (~10x
CPU), so the redundant lockstep parse costs roughly half the wall time and
the copies the other half (the per-byte 64-bit modulo in the closed-form
gather is the prime suspect).

The parse-only number ceilings **both** single-pass and any two-phase
design (a two-phase phase-1 runs the identical serial parse, so it cannot
exceed ~35 GB/s either). Two-phase's only lever is a faster phase-2 copy —
but single-pass's copy is equally optimizable, and doing so needs no table,
no barrier, and no extra memory traffic. **The decomposition question is
therefore settled for single-pass.** Perf pass 1 (#16) then measured the
two designed copy/parse micro-optimizations; both were rejected on hardware
(see "Perf pass 1" below), and the falsification trigger is evaluated there.

Occupancy readout (issue #14): 46 registers/thread, so ≥ 32 warps/SM is
achievable on sm_86.

### Perf pass 1 (issue #16): the designed micro-optimizations do not help

Both optimizations the #6 design panel grafted for perf pass 1 were
implemented and measured on Silesia against the ~18 GB/s baseline; **both
were rejected by measurement** (masterplan rule: accepted only on recorded
improvement):

| Attempt                                        | Result          | Why                                                                                                                                  |
| ---------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Incremental-mod match gather                   | ~15 GB/s (−16%) | the loop-carried `r += step` dependency pipelines worse than the independent per-iteration `i % offset`, which the compiler overlaps |
| Vectorized (16-byte) literal copy              | ~17 GB/s (−6%)  | Silesia's literal runs are mostly < 16 bytes, so the wide path rarely triggers and only adds setup/branch overhead                   |
| `__syncwarp` elision on zero-literal sequences | neutral         | the barriers are not the bottleneck                                                                                                  |

The empirical conclusion: the minimal-correct byte-per-lane kernel is at a
local optimum for this workload. The bottleneck is **structural** — the
redundant 32-lane parse sets the ~34 GB/s ceiling, and the copies are
latency-bound on the short literal/match runs typical of real data, where
neither a cheaper modulo nor wider copies help. Meaningful gains require
raising the parse ceiling itself (higher occupancy via register reduction,
or warp-specialization), which is larger than a micro-op pass — tracked as
a follow-up (issue #21).

**Falsification-trigger verdict (masterplan section 9).** The shipped
kernel is ~5× CPU (below the ~15× reopen threshold), but two-phase stays
**ruled out**: the parse-only ceiling (~10× CPU) bounds any two-phase
phase-1, which shares the identical serial parse, so two-phase cannot reach
~15× either. The trigger's numeric condition is met while its purpose — does
two-phase help? — is answered NO by the arithmetic. The path to higher
throughput is structural single-pass work, not a decomposition change.

### Perf pass 2 (issue #21): the occupancy lever does not help either

Perf pass 1 named the remaining structural lever — raise the ~34 GB/s parse
ceiling through higher occupancy. This pass profiled the kernel and measured
that lever directly; it too is rejected by measurement.

**Profile (pinned container, RTX 3080 sm_86, `nvcc -Xptxas -v`).** The shipped
`lz4_decode_batch` (Full) uses **46 registers/thread** on sm_86 (48 on sm_80).
On sm_86 — 65536 registers/SM, 256-register warp allocation granularity,
128-thread blocks — 46 rounds up to 1536 registers/warp, giving 10 resident
blocks → **40 warps/SM (~83% occupancy)**. Register granularity is the wall:
41–48 registers/thread all fall in the same 1536-register/warp bucket and all
yield 40 warps; the next occupancy step (48 warps/SM, 100%) requires
**≤ 40 registers/thread**.

**The only reachable path forces a spill.** The parser's live state is
dominated by the 64-bit stream cursors and the six-field `Lz4Sequence`. The
anti-pattern rule (masterplan section 9) forbids narrowing any of it to 32-bit
— the ABI's `size_t` capacities admit values above 2^32, pinned by the
`SIZE_MAX` and beyond-convention capacity tests — so a legitimate drop to 40
registers is not available. `__launch_bounds__(kBlockThreads, 12)` caps ptxas
at 40 registers only by **spilling to local memory** (sm_86: 16-byte stack
frame, 20 bytes spill stores, 20 bytes spill loads).

Measured on Silesia, same session and hardware, with the CPU-oracle row and
the parse-only row as controls (both run in the same invocation, so machine
state is shared):

| Configuration                                                     | Full decode                        | Parse-only (control) |
| ----------------------------------------------------------------- | ---------------------------------- | -------------------- |
| Baseline (`__launch_bounds__(kBlockThreads)`, 46 reg, 40 warps)   | p50 12.178 ms, **17.4 GB/s**       | 6.302 ms, 33.6 GB/s  |
| `__launch_bounds__(kBlockThreads, 12)` (40 reg + spill, 48 warps) | p50 12.801 ms, **16.6 GB/s (−5%)** | 6.244 ms, 33.9 GB/s  |

Forcing 100% occupancy **regresses** the full decode ~5%. Parse-only (still 28
registers, no spill, so unaffected by the register cap) stays flat across the
same runs — an internal control that attributes the regression to the spill,
not to run-to-run variance; the CPU oracle baseline was in fact slightly faster
during the variant run. The extra local-memory traffic in the already
latency-bound parse loop costs more than the eight added warps hide. All ten
ctest gates (`parser_twin` and `gpu_fixture` oracle parity, `stream_twin`
determinism, and the rest) stay green on the variant — the change is
measurement-rejected, not correctness-rejected. No code shipped; the kernel is
unchanged.

**Empirical conclusion.** The occupancy lever cannot be realized under the
current fail-closed architecture: more warps need ≤ 40 registers/thread; ≤ 40
registers needs either a forbidden 64-bit narrowing or a spill; and the spill
regresses. Raising the parse ceiling therefore requires the warp-specialization
rewrite that abandons the load-bearing redundant-lockstep-parse invariant — its
own design panel, not a measured micro-pass — and under "formats over
percentage points" (masterplan section 2) the next format outranks it. Of the
two levers issue #21 named, the register-reduction lever is measured and
rejected here; the warp-specialization lever is scoped as a larger design
change deferred behind the format ladder.

### Worst case: the worst-4Bmatch adversarial-but-valid corpus (issue #19)

A security-posture number. The Silesia rows are an average; the worst case
for this kernel's throughput is an adversarial-but-valid block of
back-to-back minimum matches (match length 4, offset 1) — the maximum
sequence density a valid LZ4 block can carry, one parsed sequence per 4
decoded bytes, which drives the redundant 32-lane lockstep parse and the
closed-form modular gather on every match byte. The standard compressor
never emits it (it extends any offset-1 run into a single long match, the
best case), so the harness constructs the block directly (`--worst4b`) and
the oracle validates it (LZ4_decompress_safe accepts and it round-trips)
before any timing. Recorded 2026-07-17, same container and RTX 3080 as the
M1 rows above; 3200 identical 64 KB chunks (~210 MB), enough to saturate
the device and sit at the Silesia scale for a direct comparison.
`--worst4b --gpu`.

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- cudec: 1 (the CPU rows time the liblz4 oracle baseline; the GPU rows below, when --gpu is set, time cudec's decoder)
- corpus: worst-4Bmatch, 3200 chunks, 209.72 MB original, 157.31 MB compressed (ratio 0.750), hand-constructed offset-1 minmatch worst case (oracle-validated; LZ4_compress_default never emits it)
- chunk sizes: min 65536 / median 65536 / max 65536 bytes
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is LZ4_decompress_safe only (no clears, no allocation); output byte-verified once before timing; percentiles are nearest-rank
- wall per run: p50 140.704 ms / p90 143.329 ms / p99 146.840 ms
- decode throughput: p50 1.490 GB/s / p90 1.463 GB/s / p99 1.428 GB/s
- GPU decode (device-resident, CUDA-event timed, 3 warmup + 30 runs): p50 25.917 ms, 8.1 GB/s
- GPU parse-only ceiling (copies elided): p50 13.710 ms, 15.3 GB/s - ceilings this design AND any two-phase phase-1 (shared parse)
```

**The degradation is linear and bounded — not an amplification vector.**

- Every path degrades by the same ~2.2–2.3× against the Silesia average:
  CPU 3.46 → 1.49 GB/s, GPU decode 18.1 → 8.1 GB/s, GPU parse-only ceiling
  34.6 → 15.3 GB/s. The uniform factor is the sequence density (one
  sequence per 4 bytes here versus Silesia's longer average matches): the
  cost is linear in the number of sequences — exactly the redundant parse
  the kernel design accepts, no super-linear blow-up. This is below the
  issue's pessimistic ~4× estimate.
- No size amplification. The block barely compresses (ratio 0.750, 157 MB →
  210 MB, ~1.33× expansion) and each chunk decodes to exactly 65536 bytes,
  capped by the caller's destination capacity — never more. The two
  adversarial axes are mutually exclusive for LZ4: a decompression bomb is
  one long match — LZ4's length encoding costs ~1 input byte per 255 output
  bytes, so a full-64 KB single-match block is ~260 bytes (~250× expansion,
  and single-match amplification is capped near 255×) — a single fast
  sequence, the opposite of this throughput worst case, and cudec's fixed
  per-chunk output cap fail-closes the size axis regardless.
- The GPU advantage holds under the worst input: 8.1 GB/s worst-case GPU is
  still ~5.4× the CPU worst case (1.49 GB/s) and ~2.3× the CPU's Silesia
  _average_ (3.46 GB/s). A second run confirmed the numbers within GPU
  jitter (8.2 GB/s decode, 15.7 GB/s parse-only).

Reproduce with `bench_lz4 --worst4b --gpu`; the construction is oracle-
validated in-harness and locked against rot by the `bench_worst4b_selfcheck`
ctest on the GPU-less runner.

## M2: reusable streaming context, end-to-end (Silesia)

The streaming decode path is now a reusable context
(`cudec_stream_ctx_create` / `cudec_lz4_decompress_stream_ctx` /
`cudec_stream_ctx_destroy`, issue #29) that owns one CUDA stream and grow-only
pinned/device staging, created once and reused across decodes. It replaces the
per-call N-stream ring of #24 (dropped — the overlap it provided is not worth
its complexity for LZ4; see below). The wall is CPU-clocked around the whole
synchronous decode call (pinned staging + H2D + decode +, for host output,
D2H). Recorded 2026-07-17, same container and RTX 3080 as the M1 rows.
`--gpu --gpu-stream-ctx`, 3 warmup + 30 measured. **Steady-state** is the
acceptance datum: repeated decodes on one reused context whose staging is
already grown. **Cold** is the first decode on a fresh context, which pays the
staging allocation.

```
- GPU decode (device-resident, CUDA-event timed): p50 12.1 ms, 17.5 GB/s
- GPU streaming, reusable context, end-to-end (host compressed in -> decoded out; wall clock around the whole synchronous decode call; 3 warmup + 30 runs):
    device out: steady-state (reused ctx) p50 229.5 ms, 0.92 GB/s ; cold (fresh ctx, first call) p50 237.8 ms, 0.89 GB/s
    host out (readback synchronous): steady-state p50 365.2 ms, 0.58 GB/s ; cold p50 373.5 ms, 0.57 GB/s
    amortized setup removed by the reusable context (cold - steady): device 8.3 ms, host 8.3 ms; 102.44 MB compressed in, 211.94 MB decoded out
```

**The measurement corrects #24's own diagnosis: the per-call allocation was
NOT the dominant cost.** #24 could not account for ~233 ms of its ~249 ms
one-shot wall, named the per-call `cudaHostAlloc`/`cudaMalloc` of the ring as
the prime suspect, and declared a reusable context "required, not optional." A
reusable context that allocates nothing in steady state now measures the
allocation directly — and it is only **~8 ms** (cold − steady), not ~233 ms.
The steady-state device wall is still **~230 ms**, against a ~12 ms
device-resident decode (one launch over all 3239 chunks) and a ~4 ms compressed
H2D (M1/#24). The ~213 ms residual is therefore neither allocation (excluded:
steady == cold to within 8 ms) nor copy/decode (~16 ms together) — it is the
**per-wave serial submission** of the batch in 64-chunk waves (~51 waves for
Silesia), one H2D + launch + event-gated reuse + result D2H per wave on this
WSL2/WDDM setup where each submission flush costs milliseconds. The
device-resident path is ~12 ms precisely because it submits **once**; the
streaming path submits ~51 times. Not separately isolated, but attributed by
exclusion. Raising the wave granularity so the path submits once is the real
lever, out of this change's scope — **issue #33**.

**So the reusable context is a simplification, not the speedup #24
anticipated** — and it still earns its place under correctness > measured
performance > minimal code: it deletes the entire N-stream ring (the overlap
machinery the analysis below shows LZ4 does not benefit from), removes a
`streams` ABI parameter the #24 measurement proved only degraded throughput,
and is the correct primitive issue #33 builds on. The `stream_twin` conformance
property now locks the reuse guarantee: the same input decoded on a reused
context — after any number of prior decodes, including one that grew the
staging — is bit-identical to a fresh-context decode.

**Why dropping the N-stream overlap was right — a better compression ratio
makes input-H2D overlap LESS valuable.** #24's overlap premise was that a
caller serializes copy-then-decode; but for LZ4 the compressed H2D is only
~4 ms against a ~12 ms decode (LZ4's ~2:1 ratio keeps the input small), so
perfect input-side overlap saves `16 − max(4, 12) = 4 ms` at best (~25 %
ceiling) and was never realized. The better a format compresses, the smaller
its input half relative to decode, and the less input-H2D overlap can ever pay
— LZ4's ratio is exactly why it does not. The genuine overlap lever for the
high-ratio M3+ formats (Zstd, GDeflate), where decode dwarfs the input
transfer, is overlapping decode against the decoded-**output** D2H (an output
ring on the host-output path), NOT the input H2D. That is the change that would
reverse this simplification, and it needs its own test when it lands — the
removed `stream_overlap` test locked the input-H2D‖kernel capability LZ4 no
longer uses, so it was removed rather than left as an orphaned lock.

The device-resident M1 row (~18 GB/s, H2D/D2H excluded) remains the kernel
throughput; the streaming numbers are a different, honest metric (copy +
per-wave submission included) and are not a regression of it.

## Baseline: CPU oracle (M0, pre-kernel)

The reference the GPU decoder is measured against: the single-threaded CPU
oracle on the development machine, recorded 2026-07-17.

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3239 chunks, 211.94 MB original, 102.44 MB compressed (ratio 0.483), compressed in-harness via LZ4_compress_default
- chunk sizes: min 8066 / median 65536 / max 65536 bytes
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is LZ4_decompress_safe only (no clears, no allocation); output byte-verified once before timing; percentiles are nearest-rank
- wall per run: p50 60.082 ms / p90 60.406 ms / p99 60.816 ms
- decode throughput: p50 3.528 GB/s / p90 3.509 GB/s / p99 3.485 GB/s
```

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- corpus: builtin, 12 chunks, 0.21 MB original, 0.10 MB compressed (ratio 0.511), compressed in-harness via LZ4_compress_default
- chunk sizes: min 1 / median 257 / max 65536 bytes
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is LZ4_decompress_safe only (no clears, no allocation); output byte-verified once before timing; percentiles are nearest-rank
- wall per run: p50 0.046 ms / p90 0.049 ms / p99 0.064 ms
- decode throughput: p50 4.480 GB/s / p90 4.221 GB/s / p99 3.206 GB/s
```
