# Benchmarks

The narrative write-up - how these numbers are read, reproduced, and
positioned - is [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md). This
file is the raw baseline record it is built on.

The baseline record. Every entry carries the full methodology block as
emitted by the harness (`bench/bench_lz4`, `--gpu` for the device path) - a
number without its methodology cannot be produced, by construction.
Regressions against the recorded baselines block merges unless explicitly
justified ([MASTERPLAN](MASTERPLAN.md) section 5). Corpora are fetched
hash-pinned via `bench/get-corpora.sh` and never committed.

Every entry below is an NVIDIA measurement. The mirror of the rule above is
that a missing number is stated rather than omitted, so the AMD side has its
own section near the end of this file
([Community AMD results](#community-amd-results)) whose current content is that
there are none.

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
- CPU decode throughput (liblz4 baseline; full methodology in "Baseline: CPU oracle" below): p50 3.41 GB/s
- GPU decode (cudec, device-resident): p50 11.7 ms, 18.1 GB/s (~5.3x the CPU baseline)
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
exceed ~35 GB/s either). Two-phase's only lever is a faster phase-2 copy -
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
local optimum for this workload. The bottleneck is **structural** - the
redundant 32-lane parse sets the ~34 GB/s ceiling, and the copies are
latency-bound on the short literal/match runs typical of real data, where
neither a cheaper modulo nor wider copies help. Meaningful gains require
raising the parse ceiling itself (higher occupancy via register reduction,
or warp-specialization), which is larger than a micro-op pass - tracked as
a follow-up (issue #21).

**Falsification-trigger verdict (masterplan section 9).** The shipped
kernel is ~5× CPU (below the ~15× reopen threshold), but two-phase stays
**ruled out**: the parse-only ceiling (~10× CPU) bounds any two-phase
phase-1, which shares the identical serial parse, so two-phase cannot reach
~15× either. The trigger's numeric condition is met while its purpose - does
two-phase help? - is answered NO by the arithmetic. The path to higher
throughput is structural single-pass work, not a decomposition change.

### Perf pass 2 (issue #21): the occupancy lever does not help either

Perf pass 1 named the remaining structural lever - raise the ~34 GB/s parse
ceiling through higher occupancy. This pass profiled the kernel and measured
that lever directly; it too is rejected by measurement.

**Profile (pinned container, RTX 3080 sm_86, `nvcc -Xptxas -v`).** The shipped
`lz4_decode_batch` (Full) uses **46 registers/thread** on sm_86 (48 on sm_80).
On sm_86 - 65536 registers/SM, 256-register warp allocation granularity,
128-thread blocks - 46 rounds up to 1536 registers/warp, giving 10 resident
blocks → **40 warps/SM (~83% occupancy)**. Register granularity is the wall:
41–48 registers/thread all fall in the same 1536-register/warp bucket and all
yield 40 warps; the next occupancy step (48 warps/SM, 100%) requires
**≤ 40 registers/thread**.

**The only reachable path forces a spill.** The parser's live state is
dominated by the 64-bit stream cursors and the six-field `Lz4Sequence`. The
anti-pattern rule (masterplan section 9) forbids narrowing any of it to 32-bit -
the ABI's `size_t` capacities admit values above 2^32, pinned by the
`SIZE_MAX` and beyond-convention capacity tests - so a legitimate drop to 40
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
same runs - an internal control that attributes the regression to the spill,
not to run-to-run variance; the CPU oracle baseline was in fact slightly faster
during the variant run. The extra local-memory traffic in the already
latency-bound parse loop costs more than the eight added warps hide. All ten
ctest gates (`parser_twin` and `gpu_fixture` oracle parity, `stream_twin`
determinism, and the rest) stay green on the variant - the change is
measurement-rejected, not correctness-rejected. No code shipped; the kernel is
unchanged.

**Empirical conclusion.** The occupancy lever cannot be realized under the
current fail-closed architecture: more warps need ≤ 40 registers/thread; ≤ 40
registers needs either a forbidden 64-bit narrowing or a spill; and the spill
regresses. Raising the parse ceiling therefore requires the warp-specialization
rewrite that abandons the load-bearing redundant-lockstep-parse invariant - its
own design panel, not a measured micro-pass - and under "formats over
percentage points" (masterplan section 2) the next format outranks it. Of the
two levers issue #21 named, the register-reduction lever is measured and
rejected here; the warp-specialization lever is scoped as a larger design
change deferred behind the format ladder.

### Perf pass 3 (issue #36): the non-overlap match fast path is rejected by the worst case

The last untried match-copy lever: when `offset >= match_len` the closed-form
modular gather `dst[m+i] = dst[m-off + (i mod off)]` degenerates to a straight
copy (`i mod off == i` for every `i < match_len <= off`, and the ranges are
disjoint, so the copy is bit-identical, hazard-free, and the branch is
warp-uniform). The fast path elides the per-byte 64-bit modulo on every
non-overlapping match - the common case in real data.

Measured 2026-07-18, same pinned container and RTX 3080, baseline and patched
binaries built from the same tree and A/B-interleaved in one session (two full
passes, 3 warmup + 30 CUDA-event-timed runs each; all twelve ctest gates -
oracle parity, determinism - green on the patched kernel first):

The Delta column is throughput speedup (`baseline_ms / fast_path_ms − 1`,
per-pass), one basis for all three rows so they compare on one scale:

| Corpus                    | Baseline p50 (2 passes) | Fast path p50 (2 passes) | Delta (throughput speedup) |
| ------------------------- | ----------------------- | ------------------------ | -------------------------- |
| Silesia `--gpu`           | 11.979 / 12.859 ms      | 10.872 / 11.822 ms       | **+10.2% / +8.8%**         |
| `--worst4b --gpu`         | 25.502 / 26.256 ms      | 27.938 / 27.654 ms       | **−8.7% / −5.1%**          |
| `--longmatch --gpu` (new) | 1.277 / 1.278 ms        | 0.922 / 0.898 ms         | **+38.5% / +42.3%**        |

The gain is real - but so is the regression, and it lands exactly where this
project refuses to pay: the adversarial worst case. `--worst4b` is offset-1
minmatch, always overlapping, so the fast-path arm is never taken; the cost is
the added per-match predicate and the second copy loop's code in the hottest
per-sequence path of the maximum-sequence-density input (one match per 4
bytes). The plan's prediction that this would be "one free warp-uniform
compare" is refuted by measurement - five independent patched `--worst4b`
sessions all landed at 26.8–27.9 ms against a 25.1–26.3 ms baseline (a 5–9%
throughput regression against the 25.1–26.3 ms baseline pair).

**Rejected under the pre-registered accept rule** (issue #36: improvement on
at least one corpus with zero regression on the others) and under the
security posture behind it: the worst-case number is the DoS-resistance
margin (issue #19), and trading it for average-case throughput inverts the
project's hostile-input-first ordering. No kernel code shipped; the
`--longmatch` harness corpus and its `bench_longmatch_selfcheck` ctest stay,
so the regime is one flag away for any future attempt (a formulation that
recovers the Silesia +9–10% without touching the worst case would be accepted -
none is known under the single-loop structure, since the predicate is
inherently per-match).

### Perf pass 4 (issue #58): narrowing the gather modulo to 32 bits is accepted

The match-copy gather computes `i mod offset` once per match byte per lane.
Both operands provably fit in 32 bits at run time - `offset` is the LZ4
2-byte offset field, and `i` is below `match_len` - but nvcc cannot prove the
second, so it emits the 64-bit software modulo. sm_86 has no integer-divide
hardware, and the width shows up in the loop body rather than in an estimate:

```
nvcc -arch=sm_86 -std=c++17 -Iinclude -Isrc -Xptxas -v -c src/batch.cu -o /tmp/batch.o
cuobjdump -sass /tmp/batch.o
```

| Gather loop (SASS, `lz4_decode_batch<false>`) | Body            | Instructions per iteration |
| --------------------------------------------- | --------------- | -------------------------- |
| Baseline, 64-bit modulo                       | `0x1520-0x17a0` | 41                         |
| Patched, 32-bit arm                           | `0x18c0-0x19e0` | 19                         |
| Patched, 64-bit fallback arm                  | `0x1530-0x17b0` | 41                         |

The literal-copy loop is 11 instructions on both builds, and ptxas reports
`Used 48 registers, 0 bytes stack frame` on both, so the narrowing buys the
gather without moving occupancy - which is what sank perf pass 2.

Measured 2026-08-09, same pinned container and RTX 3080. **Both binaries are
built once, up front, and then alternated**, baseline and patched, pass after
pass. That method is the finding underneath the numbers and is worth stating
before them: an A/B/A that rebuilds between arms was run first and it read
`--worst4b` at +7.7%, which the alternation does not reproduce. Over minutes
this GPU drifts by more than the effect on two of the three corpora, so
whichever arm is measured later carries the drift. Alternating moves both arms
through the same drift. 3 warmup + 30 CUDA-event-timed runs per number.

| Corpus              | Baseline p50, five passes                     | Patched p50, five passes                      |
| ------------------- | --------------------------------------------- | --------------------------------------------- |
| `--worst4b --gpu`   | 26.632 / 26.383 / 26.149 / 26.298 / 26.507 ms | 26.213 / 25.974 / 26.284 / 25.981 / 25.965 ms |
| `--longmatch --gpu` | 1.281 / 1.422 / 1.347 / 1.422 / 1.422 ms      | 0.984 / 1.032 / 1.087 / 1.089 / 1.087 ms      |

Silesia over four alternations, with its parse-only ceiling read from the same
invocations. The ceiling elides the copies, so the gather cannot reach it:

| Pass | Baseline full / parse | Patched full / parse |
| ---- | --------------------- | -------------------- |
| 1    | 12.597 / 6.787 ms     | 12.272 / 7.145 ms    |
| 2    | 12.669 / 7.188 ms     | 12.470 / 7.284 ms    |
| 3    | 13.014 / 6.785 ms     | 12.451 / 7.135 ms    |
| 4    | 12.575 / 7.258 ms     | 12.245 / 7.258 ms    |

Throughput speedup on the medians, and the pairwise count, because with an
effect this size the count says more than the ratio:

| Corpus              | Baseline / patched median | Delta      | Passes won |
| ------------------- | ------------------------- | ---------- | ---------- |
| `--longmatch --gpu` | 1.422 / 1.087 ms          | **+30.8%** | 5 of 5     |
| Silesia `--gpu`     | 12.633 / 12.361 ms        | **+2.2%**  | 4 of 4     |
| `--worst4b --gpu`   | 26.383 / 25.981 ms        | **+1.5%**  | 4 of 5     |

**The worst case gains least, and that refutes what this lever was expected to
do.** #58 named `--worst4b` as the primary target, on the reasoning that every
one of its match bytes takes the overlap modulo. Every one does, but there are
only four of them per match: at match length 4 the gather loop runs a single
iteration on four of the thirty-two lanes, so the cheaper iteration is
amortized against a parse that is 59% of that corpus's wall
(15.5 ms of 26.4 ms parse-only). `--longmatch`'s 255-byte matches run the loop
eight times per lane and there the gather is the wall. The lever is a
match-length lever, not an overlap lever, and the corpus that exposes it is
the one perf pass 3 built for the opposite purpose.

Silesia's copy half, full minus the ceiling from the same invocation, is
5.810 / 5.481 / 6.229 / 5.317 ms against 5.127 / 5.186 / 5.316 / 4.987 ms.
Perf pass 1 concluded that Silesia's short copies are latency-bound and that
a cheaper modulo would not help them; +2.2% on the whole decode is small
enough to leave that conclusion standing rather than overturn it.

**Accepted.** Every corpus improves or holds, nothing regresses, and the axis
perf pass 3 was rejected on is unmoved in the wrong direction: `--worst4b` is
the adversarial worst case and the DoS-resistance margin, and it gains here
rather than paying. The 32-bit arm is entered under a warp-uniform
`match_len <= kGather32BitMaxLen`, so it costs no divergence, and a match
longer than 2^32 still decodes through the unchanged 64-bit arm.

### Best case: the longmatch corpus (issue #36)

The shipped kernel's baseline on the new `--longmatch` corpus - long
non-overlapping matches (match length 255 at offset 512), the copy-dominated
regime that maximally exposes the per-byte modular gather. Hand-constructed
like `--worst4b` (the standard compressor emits long matches but this shape
pins `offset >= match_len` on every match), oracle-validated in-harness,
shape-locked by the `bench_longmatch_selfcheck` ctest. Recorded 2026-07-18,
same container and RTX 3080. `--longmatch --gpu`.

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- cudec: 1 (the CPU rows time the liblz4 oracle baseline; the GPU rows below, when --gpu is set, time cudec's decoder)
- corpus: longmatch, 3200 chunks, 209.72 MB original, 5.72 MB compressed (ratio 0.027), hand-constructed long non-overlapping matches (offset 512 >= match length 255; oracle-validated; LZ4_compress_default never emits it)
- chunk sizes: min 65536 / median 65536 / max 65536 bytes
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is LZ4_decompress_safe only (no clears, no allocation); output byte-verified once before timing; percentiles are nearest-rank
- wall per run: p50 5.783 ms / p90 5.847 ms / p99 5.932 ms
- decode throughput: p50 36.265 GB/s / p90 35.868 GB/s / p99 35.351 GB/s
- GPU decode (device-resident, CUDA-event timed, 3 warmup + 30 runs): p50 1.271 ms, 165.0 GB/s
- GPU parse-only ceiling (copies elided): p50 0.265 ms, 790.7 GB/s - ceilings this design AND any two-phase phase-1 (shared parse)
```

At 165 GB/s the copy-dominated best case runs ~9x the Silesia average and
within ~4.6x of the ~760 GB/s output-bandwidth ceiling - the modular gather,
not bandwidth, is the limiter in this regime (the rejected fast path reached
~228–234 GB/s here, from its 0.922 / 0.898 ms A/B passes above over the
209.72 MB corpus), consistent with the ~250–400 GB/s redundant-parse family
ceiling published in the masterplan.

### Worst case: the worst-4Bmatch adversarial-but-valid corpus (issue #19)

A security-posture number. The Silesia rows are an average; the worst case
for this kernel's throughput is an adversarial-but-valid block of
back-to-back minimum matches (match length 4, offset 1) - the maximum
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

**The degradation is linear and bounded - not an amplification vector.**

- Every path degrades by the same ~2.2–2.3× against the Silesia average:
  CPU 3.41 → 1.49 GB/s, GPU decode 18.1 → 8.1 GB/s, GPU parse-only ceiling
  34.6 → 15.3 GB/s. The uniform factor is the sequence density (one
  sequence per 4 bytes here versus Silesia's longer average matches): the
  cost is linear in the number of sequences - exactly the redundant parse
  the kernel design accepts, no super-linear blow-up. This is below the
  issue's pessimistic ~4× estimate.
- No size amplification. The block barely compresses (ratio 0.750, 157 MB →
  210 MB, ~1.33× expansion) and each chunk decodes to exactly 65536 bytes,
  capped by the caller's destination capacity - never more. The two
  adversarial axes are mutually exclusive for LZ4: a decompression bomb is
  one long match - LZ4's length encoding costs ~1 input byte per 255 output
  bytes, so a full-64 KB single-match block is ~260 bytes (~250× expansion,
  and single-match amplification is capped near 255×) - a single fast
  sequence, the opposite of this throughput worst case, and cudec's fixed
  per-chunk output cap fail-closes the size axis regardless.
- The GPU advantage holds under the worst input: 8.1 GB/s worst-case GPU is
  still ~5.4× the CPU worst case (1.49 GB/s) and ~2.3× the CPU's Silesia
  _average_ (3.41 GB/s). A second run confirmed the numbers within GPU
  jitter (8.2 GB/s decode, 15.7 GB/s parse-only).

Reproduce with `bench_lz4 --worst4b --gpu`; the construction is oracle-
validated in-harness and locked against rot by the `bench_worst4b_selfcheck`
ctest on the GPU-less runner.

### Text-heavy: the enwik8 corpus (issue #138)

`bench/get-corpora.sh` has pinned enwik8 since the corpora were pinned, and
this file carried no row for it. That is the gap #138 was opened on: a pinned
corpus we never measure is an unkept promise in the methodology document, so
it either gains a row or loses its pin. It gains a row.

The first 100 MB of an English Wikipedia dump, so pure marked-up text rather
than the mixture Silesia averages over. No harness work was needed:
`bench_lz4` takes corpus files positionally and splits any file into 64 KB
chunks, so this is the same code path the Silesia rows use, on a different
input. Recorded 2026-08-05 inside the digest-pinned dev container
(`nvidia/cuda:12.6.2-devel-ubuntu24.04`) on the same RTX 3080 as every row
above, `--warmup 3 --runs 30`. Fetched and verified before the run: zip
SHA-256 `547994d9...34bc` and extracted-file SHA-256 `2b49720e...24a8`, both
`OK` against `bench/get-corpora.sh`'s pins.

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- cudec: 1 (the CPU rows time the liblz4 oracle baseline; the GPU rows below, when --gpu is set, time cudec's decoder)
- corpus: enwik8, 1526 chunks, 100.00 MB original, 57.00 MB compressed (ratio 0.570), compressed in-harness via LZ4_compress_default
- chunk sizes: min 57600 / median 65536 / max 65536 bytes
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is LZ4_decompress_safe only (no clears, no allocation); output byte-verified once before timing; percentiles are nearest-rank
- wall per run: p50 33.835 ms / p90 33.999 ms / p99 34.142 ms
- decode throughput: p50 2.956 GB/s / p90 2.941 GB/s / p99 2.929 GB/s
- GPU decode (device-resident, CUDA-event timed, 3 warmup + 30 runs): p50 8.999 ms, 11.1 GB/s
- GPU parse-only ceiling (copies elided): p50 4.592 ms, 21.8 GB/s - ceilings this design AND any two-phase phase-1 (shared parse)
```

What the row is worth keeping for: enwik8 is **not** a restatement of the
Silesia average. Its GPU decode lands at 11.1 GB/s against Silesia's 18.1,
about 1.6× slower, and its parse-only ceiling at 21.8 against 34.6, about the
same factor. The CPU oracle moves much less, 2.956 against 3.41. So the
corpus separates the two paths rather than scaling both, and the GPU-over-CPU
factor falls from ~5.3× on Silesia to ~3.8× here. Had the row simply
reproduced Silesia, unpinning would have been the better answer.

What the row does **not** establish is why. enwik8 compresses worse than
Silesia (ratio 0.570 against 0.483), which under the cost model already
recorded above - cost linear in the number of sequences, not in output bytes -
would predict exactly this direction, since a less compressible input carries
more sequences per decoded byte. That is consistent with the number and is not
measured by it: the harness reports no sequence count, and nothing here
counted one. Stated as the reading it is.

Reproduce with `bench/get-corpora.sh` followed by
`bench_lz4 bench/corpora/enwik8 --gpu --warmup 3 --runs 30`.

### Cost of the termination fuel cap (issue #72)

The sequence loop in `src/lz4_decode.cuh` now carries an explicit decrementing
fuel cap, so a parser bug that stopped advancing the source cursor would
produce a rejected chunk instead of a warp that never returns. The cap is one
64-bit decrement per sequence, folded into the loop's existing exit branch.
**It is not free, and the number is recorded here rather than waved away.**

Measured back-to-back in one session, same container digest, same machine,
same corpora: `origin/main` at `e85194d` against this change, `--warmup 3
--runs 30`, GPU device-resident and CUDA-event timed. Every comparison below
is the mean of two interleaved passes (after, before, after, before) so
drifting clocks cannot favour one side, and every individual sample is listed
so the spread is visible rather than averaged away.

The methodology block below is **one standalone run of the "after" build**,
pasted whole because the rule here is that a number ships with its
methodology. Its GPU decode p50 is 12.427 ms, inside that build's observed
run-to-run range but not identical to the paired mean in the table - the same
build measured twice, not a discrepancy. The paired means are what the claim
rests on.

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- cudec: 1 (the CPU rows time the liblz4 oracle baseline; the GPU rows below, when --gpu is set, time cudec's decoder)
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3239 chunks, 211.94 MB original, 102.44 MB compressed (ratio 0.483), compressed in-harness via LZ4_compress_default
- chunk sizes: min 8066 / median 65536 / max 65536 bytes
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is LZ4_decompress_safe only (no clears, no allocation); output byte-verified once before timing; percentiles are nearest-rank
- wall per run: p50 60.259 ms / p90 63.591 ms / p99 64.079 ms
- decode throughput: p50 3.517 GB/s / p90 3.333 GB/s / p99 3.307 GB/s
- GPU decode (device-resident, CUDA-event timed, 3 warmup + 30 runs): p50 12.427 ms, 17.1 GB/s
- GPU parse-only ceiling (copies elided): p50 7.163 ms, 29.6 GB/s - ceilings this design AND any two-phase phase-1 (shared parse)
```

| Corpus        | Metric                     | Before (samples)          | After (samples)           | Change                   |
| ------------- | -------------------------- | ------------------------- | ------------------------- | ------------------------ |
| Silesia       | GPU decode p50             | 11.924 ms (11.873/11.974) | 12.163 ms (12.107/12.218) | **+2.0%**                |
| Silesia       | GPU parse-only ceiling p50 | 6.056 ms (5.924/6.188)    | 7.079 ms (7.010/7.147)    | **+16.9%**               |
| worst-4Bmatch | GPU decode p50             | 25.643 ms (25.364/25.922) | 25.973 ms (25.285/26.661) | +1.3%, inside the spread |
| worst-4Bmatch | GPU parse-only ceiling p50 | 13.482 ms (13.467/13.496) | 15.437 ms (15.367/15.506) | **+14.5%**               |

The shipped decode path pays **+2.0% on Silesia**; on the worst case the
difference (+1.3%) is smaller than the "after" side's own pass-to-pass spread
(25.285 / 26.661 ms), so it is **not a measurement**, only an upper bound. The
parse-only ceiling - a diagnostic kernel with the copies elided, never shipped -
pays 14–17% on both corpora, tightly and reproducibly, which is where the
per-sequence cost shows up undiluted by memory stalls.

#### Occupancy is the constraint, and it bounds what the guard may cost

The kernel uses **48 registers/thread** on `sm_86`, up from 46 on `origin/main`.
That matters more than the two-register difference suggests: at 128-thread
blocks and 256-register warp granularity, 41–48 registers all round to
1536 registers/warp → 6144/block → 10 resident blocks → **40 warps/SM**, while
**49 registers steps down to 36 warps/SM**. The shipped kernel therefore sits
on the last rung before an occupancy cliff, and that was measured, not assumed:

| Variant                                          | Registers | Warps/SM | Silesia GPU decode p50 |
| ------------------------------------------------ | --------- | -------- | ---------------------- |
| `origin/main`, no cap and no geometry guard      | 46        | 40       | 11.924 ms              |
| shipped (cap + two-clause geometry guard)        | 48        | 40       | 12.163 ms              |
| + widen `warp_in_grid` to 64-bit                 | 52        | 36       | 12.63 ms               |
| + refuse via `total_warps > 1 << 27`             | 52        | 36       | not timed              |
| + refuse via `gridDim.x > UINT_MAX / blockDim.x` | 52        | 36       | not timed              |

The last three rows are why the kernel's 32-bit `warp_in_grid` is **left
32-bit and documented as a limit rather than enforced**. Three spellings of the
2^32-thread bound were measured, including one that touches no 64-bit value at
all and compares two values already live in registers - nvcc 12.6 allocates the
same four extra registers for all three, so neither "widening is free" nor
"there must be a cheaper spelling" survives contact with the compiler. Only the
first was timed; the other two share its register count and therefore its
occupancy, and were not timed separately.

What settles it is not the cost but the asymmetry in the consequence. The guard
already forces `blockDim.x` to a multiple of the warp size, so a wrapped index
stays warp-aligned: an aliased block recomputes the **same** `warp_in_grid`
values with a full 0–31 lane range, decodes the same chunks, and writes
byte-identical values to the same addresses. Exceeding the documented bound
therefore costs duplicated work - not missing bytes, not a hang, and the output
stays bit-identical. That is categorically unlike the two refused geometries,
where bytes really do go unwritten or the kernel really does spin. Paying an
occupancy step on every launch to prevent redundant-but-correct output, on a
geometry needing 33 million blocks against a `decode_grid_blocks` cap of 8192
and unreachable through the public ABI, is the wrong trade.

Anything added to this kernel from here has a hard budget: **48 registers.**

#### The formulation comparison does not rank the formulations

Three spellings of the cap were tried, and the honest reading is that the
measurement does not separate them. Silesia GPU decode p50, all samples ever
taken, across sessions:

| Formulation                                    | Samples (ms)                                                 |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `origin/main`, no cap                          | 11.873 / 11.974 / 11.980 / 11.987 / 12.011 / 12.218 / 12.234 |
| folded into the existing exit branch (shipped) | 12.107 / 12.218 / 12.442 / 12.498 / 12.684                   |
| 32-bit counter                                 | 12.699                                                       |
| `while (fuel-- != 0)` at the top of the loop   | 12.744 / 12.822                                              |

Only the first two rows were measured under the paired interleaved protocol,
and only they separate by more than the shipped build's own run-to-run range -
so the with/without-cap difference is real and the between-formulation
differences are **not established**. The shipped formulation was chosen on a
structural argument, not a measured one: folding the test into the loop's
existing exit branch adds no branch at the top of the loop. The 32-bit counter
was dropped on correctness, not speed - keeping its budget unreachable would
need a 4 GiB per-chunk limit, an accept-set change.

The regression is accepted deliberately under the prime-directive ordering -
**correctness > measured performance**. A decoder that hangs on hostile input
has failed open in the availability direction; nvCOMP 5.3's release notes
record shipping exactly that bug class in its Snappy decoder. Recovering the
throughput later is legitimate measurement-gated perf work, but not at the
price of the guarantee.

## M2: reusable streaming context, end-to-end (Silesia)

The streaming decode path is now a reusable context
(`cudec_stream_ctx_create` / `cudec_lz4_decompress_stream_ctx` /
`cudec_stream_ctx_destroy`, issue #29) that owns one CUDA stream and grow-only
pinned/device staging, created once and reused across decodes. It replaces the
per-call N-stream ring of #24 (dropped - the overlap it provided is not worth
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
allocation directly - and it is only **~8 ms** (cold − steady), not ~233 ms.
The steady-state device wall is still **~230 ms**, against a ~12 ms
device-resident decode (one launch over all 3239 chunks) and a ~4 ms compressed
H2D (M1/#24). The ~213 ms residual is therefore neither allocation (excluded:
steady == cold to within 8 ms) nor copy/decode (~16 ms together) - it is the
**per-wave serial submission** of the batch in 64-chunk waves (~51 waves for
Silesia), one H2D + launch + event-gated reuse + result D2H per wave on this
WSL2/WDDM setup where each submission flush costs milliseconds. The
device-resident path is ~12 ms precisely because it submits **once**; the
streaming path submits ~51 times. Not separately isolated, but attributed by
exclusion. Raising the wave granularity so the path submits once is the real
lever, out of this change's scope - **issue #33**.

**So the reusable context is a simplification, not the speedup #24
anticipated** - and it still earns its place under correctness > measured
performance > minimal code: it deletes the entire N-stream ring (the overlap
machinery the analysis below shows LZ4 does not benefit from), removes a
`streams` ABI parameter the #24 measurement proved only degraded throughput,
and is the correct primitive issue #33 builds on. The `stream_twin` conformance
property now locks the reuse guarantee: the same input decoded on a reused
context - after any number of prior decodes, including one that grew the
staging - is bit-identical to a fresh-context decode.

**Why dropping the N-stream overlap was right - a better compression ratio
makes input-H2D overlap LESS valuable.** #24's overlap premise was that a
caller serializes copy-then-decode; but for LZ4 the compressed H2D is only
~4 ms against a ~12 ms decode (LZ4's ~2:1 ratio keeps the input small), so
perfect input-side overlap saves `16 − max(4, 12) = 4 ms` at best (~25 %
ceiling) and was never realized. The better a format compresses, the smaller
its input half relative to decode, and the less input-H2D overlap can ever pay -
LZ4's ratio is exactly why it does not. The genuine overlap lever for the
high-ratio M3+ formats (Zstd, GDeflate), where decode dwarfs the input
transfer, is overlapping decode against the decoded-**output** D2H (an output
ring on the host-output path), NOT the input H2D. That is the change that would
reverse this simplification, and it needs its own test when it lands - the
removed `stream_overlap` test locked the input-H2D‖kernel capability LZ4 no
longer uses, so it was removed rather than left as an orphaned lock.

The device-resident M1 row (~18 GB/s, H2D/D2H excluded) remains the kernel
throughput; the streaming numbers are a different, honest metric (copy +
per-wave submission included) and are not a regression of it.

## Community AMD results

**No community results yet.** Nothing in this section is a measurement; it
exists so that the absence of AMD numbers everywhere above is a recorded fact
rather than a silence.

**What has and has not happened on AMD, as of this entry.** Every number in
this file was taken on one NVIDIA device (RTX 3080, sm_86, CUDA 12.6). The
maintainer hardware is NVIDIA-only, so no AMD GPU has ever executed this
decoder for the project, and no AMD measurement exists to record. There is also
no AMD build. `CUDEC_ENABLE_HIP` now exists in `CMakeLists.txt`, and it refuses
on every machine: cudec has no HIP device sources yet, so the option holds the
fail-closed contract rather than producing a build, and no CI job compiles for
a `gfx` target. So the honest state is still neither measured nor compiled,
which is one rung below where this section will sit once the port and the HIP
compile gate land (issue #111) and the "compiled" half becomes a thing a
workflow run can be pointed at.

**What a community result is, and what it is not.** A third-party measurement
on hardware the maintainer does not have and cannot re-run. It is recorded as
exactly that. It is **not** promoted to a project baseline and it does **not**
gate merges: the rule that regressions against a recorded baseline block a
merge (MASTERPLAN section 5) applies to the CUDA baselines above, which the
maintainer can reproduce on demand, and extending it to a number nobody here can
reproduce would make an unreproducible figure into a merge veto. A community
result informs; it does not gate.

**The recording format, fixed before the first result rather than by it.** When
one arrives it is appended below as its own subsection, headed exactly like
this so the index values cannot be left out of the heading and filled in
somewhere less visible:

```
### <GPU name> / <gfx target> / ROCm <version> / wave <32|64> (#<issue>)
```

- The reporter's `bench_lz4` report block, pasted **whole and unedited**, in a
  fenced code block. Not summarised, not re-typed, and not trimmed to the lines
  that seemed interesting. The block is the evidence; a paraphrase of it is not.
- The four index values repeated in the heading above are what the entry is
  keyed by, because they are what makes two AMD results comparable or not. The
  wave width is load-bearing here in a way it is not on the CUDA rows: a 32-wide
  and a 64-wide build of the same kernel family are different executions of the
  same source, and a table that loses that distinction is a table that averages
  across it.
- The submitting issue linked, and the reporter credited by the name or handle
  they used to submit. If they asked not to be credited, that is written here
  too, in place of the name.
- Anything the reporter's environment did that the report block does not carry
  goes underneath in prose, marked as the reporter's statement rather than as a
  measured field.

The `bench_lz4` report block emits a `CUDA device:` line and carries no ROCm
version and no wave width today (`bench/bench_lz4.cpp`). Until the HIP build
lands and the harness emits them, those two index values come from the reporter
by hand and are recorded as reported, not as read out of the device.

No sentence in this section can be refuted by pointing out that no AMD GPU was
ever run, because that is what it says.

## Baseline: CPU oracle (M0, pre-kernel)

The reference the GPU decoder is measured against: the single-threaded CPU
oracle on the development machine. Re-measured 2026-07-17 (issue #48) with a
plain, GPU-less invocation (`bench_lz4 bench/corpora/silesia/*`, no `--gpu`)
to replace a stale figure that had drifted ~2% from the M1 block's CPU line
through ordinary run-to-run wall-clock jitter (single-thread CPU decode is
timed with `wall clock`, not CUDA events, so it is not jitter-free); this is
now the one authoritative Silesia CPU-oracle figure, cited by the M1 block
above as well.

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- cudec: 1 (the CPU rows time the liblz4 oracle baseline; the GPU rows below, when --gpu is set, time cudec's decoder)
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3239 chunks, 211.94 MB original, 102.44 MB compressed (ratio 0.483), compressed in-harness via LZ4_compress_default
- chunk sizes: min 8066 / median 65536 / max 65536 bytes
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is LZ4_decompress_safe only (no clears, no allocation); output byte-verified once before timing; percentiles are nearest-rank
- wall per run: p50 62.158 ms / p90 62.742 ms / p99 63.291 ms
- decode throughput: p50 3.410 GB/s / p90 3.378 GB/s / p99 3.349 GB/s
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
