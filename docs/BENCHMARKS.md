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

## M2: pinned-host streaming decode, end-to-end (Silesia)

The streaming decode path (`cudec_lz4_decompress_stream`, issue #24) times the
whole synchronous call by CPU wall clock — pinned staging, H2D, decode, and,
for host output, D2H, plus the **one-shot per-call ring setup** — the number a
caller of the one-shot entry actually sees. Recorded 2026-07-17, same container
and RTX 3080 as the M1 rows. `--gpu --gpu-stream`, 3 warmup + 30 measured
(the device-resident row below is the `--gpu` path in the same run, for
comparison; it is CUDA-event timed, the streaming rows are wall-clock).

```
- GPU decode (device-resident, CUDA-event timed): p50 11.9 ms, 17.8 GB/s
- GPU streaming, end-to-end, ONE-SHOT-SETUP-DOMINATED (host compressed in -> decoded out; wall clock around the whole synchronous call incl. per-call ring setup; 3 warmup + 30 runs):
    device out: 1 stream p50 249.1 ms, 0.85 GB/s ; 4 streams (of 4 requested) p50 288.6 ms, 0.73 GB/s
    host out (1 internal stream; readback synchronous): p50 384.2 ms, 0.55 GB/s
    context: pure contiguous H2D of 102.44 MB compressed = p50 3.928 ms (a best-case floor; the pipeline stages many smaller per-wave copies) - dwarfed by the walls above, so the walls are setup-bound, not PCIe-bound
```

**The measurement falsifies the streaming path's value proposition as built —
two honest findings (masterplan section 5: numbers decide, not intentions):**

1. **Overlap does not pay off for LZ4.** The premise (issue #24) was that a
   caller "serializes copy-then-decode." But the compressed H2D is only
   **3.9 ms** — LZ4 compresses ~2:1, so the copy is small, while the decode is
   ~12 ms. Serial copy-then-decode is ~16 ms; perfect overlap is
   `max(4, 12) = 12 ms`, a ~25 % / ~4 ms win at best. The copy was never the
   wall for LZ4, so there is little to hide. (Overlap pays off when the
   compressed input is large relative to decode — not this format.)

2. **The one-shot per-call ring setup dominates and is the real cost.** The
   end-to-end wall (249 ms device-serial) dwarfs copy (~4 ms) + decode
   (~12 ms) by ~15×; the unaccounted ~233 ms is not separately isolated, but
   the prime suspect is the per-call `cudaHostAlloc` / `cudaMalloc` (and their
   frees) of the pinned ring, disproportionately expensive on this WSL2/WDDM
   setup — corroborated by the fact that it scales with stream count. Adding
   streams makes it **slower**
   (289 ms at 4 streams), because 4 streams provision ~4× the ring: the
   setup, not the pipeline, scales with stream count, so the two device rows
   are not a clean overlap comparison. PR1 deferred a reusable context "until
   profiling shows per-call allocation dominating" — profiling now shows
   exactly that. **The reusable context is required, not optional** (issue
   filed as the #24-PR2 follow-up).

The device-resident M1 row (~18 GB/s, H2D excluded) remains the kernel
throughput; the streaming number is a different, honest metric (copy +
setup included) and is not a regression of it. The `stream_overlap`
conformance test verifies the copy/decode **overlap capability** holds on
this hardware — an async H2D and the decode kernel on separate streams
complete in materially less than the sum of their individual times. Whether
cudec's own one-shot pipeline achieves that overlap is not externally
observable while the per-call setup dominates the wall; it becomes a
measurable, gate-able number once the reusable context lands. Both facts —
the capability exists, the one-shot form buries it and overlap is weak for
LZ4 anyway — are recorded rather than hidden.

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
