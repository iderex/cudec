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

### Game-asset-like: the generated asset corpus (issue #139)

The regime the README leads with, and the only corpus here whose source data
is barely compressible. One 64 KB block tiled by four regions in the
proportions of a shipped asset package - BC1-shaped block-compressed texture,
interleaved 32-byte vertex records, a triangle-list index buffer, 16-bit
stereo PCM audio - replicated to 3200 chunks. Unlike `--worst4b` and
`--longmatch` the wire is ordinary `LZ4_compress_default` output: the shape
being modelled is the source data, and letting the standard compressor decide
the sequence structure is the point. Oracle-validated before any timing and
regime-locked by the CPU-only `bench_assetlike_selfcheck` ctest.

**It is a MODEL of the workload, not the workload.** A synthetic
texture/mesh/audio mixture reproduces the regime and is not a measurement on
real game data. No number below may be quoted as one.

Recorded 2026-08-09 in the digest-pinned dev container
(`nvidia/cuda:12.6.2-devel-ubuntu24.04`) on the same RTX 3080 as every row
above, `--assetlike --gpu --warmup 3 --runs 30`.

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- cudec: 100 (the CPU rows time the liblz4 oracle baseline; the GPU rows below, when --gpu is set, time cudec's decoder)
- corpus: asset-like, 3200 chunks, 209.72 MB original, 169.56 MB compressed (ratio 0.809), generated in-harness, a MODEL of a game asset package (BC1 texture, interleaved geometry, 16-bit PCM audio) and not a measurement on real game data; compressed by LZ4_compress_default and oracle-validated
- chunk sizes: min 65536 / median 65536 / max 65536 bytes
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is LZ4_decompress_safe only (no clears, no allocation); output byte-verified once before timing; percentiles are nearest-rank
- wall per run: p50 20.914 ms / p90 22.437 ms / p99 24.521 ms
- decode throughput: p50 10.028 GB/s / p90 9.347 GB/s / p99 8.552 GB/s
- GPU decode (device-resident, CUDA-event timed, 3 warmup + 30 runs): p50 4.748 ms, 44.2 GB/s
- GPU parse-only ceiling (copies elided): p50 2.574 ms, 81.5 GB/s - ceilings this design AND any two-phase phase-1 (shared parse)
```

Three consecutive invocations of that command gave GPU decode p50 4.748,
4.756 and 4.307 ms (44.2, 44.1, 48.7 GB/s). The block above is the first of
them; the spread is the session drift perf pass 4 measured on this host and
is why the paragraphs below read only what one invocation says about itself.

**No comparison against the Silesia row is drawn here, and the omission is
deliberate.** The Silesia figures in this document were taken before the
gather narrowing landed (perf pass 4, issue #58), so a ratio against them
would mix two kernels and one GPU's drift into a single number. What the
asset-like invocation says about itself, from its own timings:

- The copy half is 4.748 - 2.574 = 2.174 ms, about 46% of the decode. That is
  a copy-heavier split than the corpus's compressibility suggests: at ratio
  0.809 there is little to decode, and most of the output is literal bytes
  moved rather than matches gathered.
- The GPU-over-CPU factor is ~4.4× (44.2 against 10.028 GB/s), the two
  measured in the same invocation. The CPU oracle is fast here for the same
  reason: a barely-compressible input gives `LZ4_decompress_safe` little work
  per output byte.

What this row does **not** establish is where the asset-like regime sits
relative to the other corpora on today's kernel. That needs the other rows
re-measured in one session, which is a separate piece of work and is not
claimed here.

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

#### Recovery attempt: the counted trip count (issue #75)

**Measured-negative. No kernel code shipped.**

The cost above is a per-sequence decrement and its counter. The shape measured
here removes the counter and spends the same budget as the trip count of a
counted loop instead, so the bound becomes the loop's own induction variable:
`for (uint64_t left = budget; left != 0; --left)` around the sequence loop,
with the `done` test left where it was and the fuel test gone. The budget is
`src_size + 2` rather than `src_size + 1`, saturating at the top of the range,
because the decrementing spelling tested its counter after the call and so
admitted one more call than it was initialized with. Matching that exactly is
what keeps the accept set unmoved.

This is the fourth formulation measured against this cap. Three were measured
when it landed, and this is not among them: they were a top-of-loop test, the
folded exit-branch test that shipped, and a 32-bit counter, all of them counter
shapes. It is also the only remaining shape the configure-time loop scanner
admits, since that scanner requires a `while` or a condition-less `for` to name
an explicit decrementing `fuel` counter, and a counted `for` with a visible
condition satisfies it by construction rather than by naming anything.

Three interleaved passes, after / before / after / before / after / before in
one session, both binaries built from the same tree, recorded 2026-08-10 inside
the digest-pinned `nvidia/cuda:12.6.2-devel-ubuntu24.04` container on the RTX
3080 (sm_86, driver 560.94, CUDA 12.6, nvcc V12.6.77), device-resident and
CUDA-event timed, 3 warmup + 30 measured runs. Baseline is `40a9f54`.
Reproduce with `bench_lz4 --gpu bench/corpora/silesia/*` and
`bench_lz4 --gpu --worst4b`, both `--warmup 3 --runs 30`.

| Corpus        | Metric         | Before (samples)            | After (samples)             | Change |
| ------------- | -------------- | --------------------------- | --------------------------- | ------ |
| Silesia       | GPU decode p50 | 11.834 / 11.544 / 11.588 ms | 11.643 / 12.314 / 12.032 ms | +2.9 % |
| Silesia       | Parse-only p50 | 6.724 / 6.724 / 6.717 ms    | 6.682 / 6.685 / 6.686 ms    | -0.6 % |
| worst-4Bmatch | GPU decode p50 | 24.401 / 24.511 / 24.518 ms | 24.445 / 24.619 / 24.630 ms | +0.4 % |
| worst-4Bmatch | Parse-only p50 | 14.500 / 14.582 / 14.592 ms | 14.451 / 14.474 / 14.558 ms | -0.4 % |

Registers, from `cuobjdump --dump-resource-usage` on the built archive rather
than from a compile log: **sm_80 54 to 47, sm_86 48 to 47**. On sm_86 that is
inside the 41-48 bucket recorded above, so it moves no occupancy step and buys
nothing there. The change stays inside the 48-register budget this issue set.

**The verdict, and it is not the same on both rows.** The parse-only ceiling
does improve, by about half a percent, and it is the one row here worth
trusting: its four samples per side are within 0.005 ms of each other and the
two sides do not overlap. That says the counted form genuinely costs slightly
less per sequence.

It says nothing that reaches the shipped decoder. The decode rows show no
improvement on either corpus; Silesia reads +2.9 % off samples that overlap the
before side, and worst-4Bmatch +0.4 % well inside its own spread. The
pre-registered accept rule needs a recorded improvement on at least one corpus
with zero regression on the worst case, and there is no improvement to weigh,
so the change is refused.

**What this closes and what it does not.** The cap costs 14-17 % of the
parse-only ceiling; this recovers half a percent of it, roughly a thirtieth,
and none of the 2.0 % on the decode path. Together with the three formulations
measured when the cap landed, that is every counter shape the scanner admits,
and the per-block-precondition direction is separately closed off: the budget
is already derived once per chunk from the source size, and what remains per
iteration is enforcing a limit on iterations, which means counting them. So the
answer to the question this issue asked is that the cap's decode cost is not in
the counter's shape and cannot be reformulated away.

That is narrower than saying it is unrecoverable. Nothing here measures a
change to the parser's liveness argument, which is what would remove the need
for a budget rather than re-spell it, and nothing here touches the second
budget inside `AccumulateLength`, which runs only on a length extension and is
therefore invisible on a corpus without them.

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

### The wave is sized by staging, not by a chunk count (issue #33)

The attribution above named the lever and this is it taken. `kWaveChunks = 64`
is gone; the wave is now the largest number of chunks whose staging fits a
384 MiB budget, capped at 4096 chunks. The budget is charged against the
LARGEST chunk in the call rather than the mean, so peak staging stays inside it
for any distribution including a hostile one, and the chunk ceiling bounds the
metadata staging on its own - without it a batch of zero-length chunks divides
the budget by one. Silesia's 3239 chunks become ONE wave on the device-output
path and two on the host-output path, against ~51 before.

Measured 2026-08-09, same container and RTX 3080. **Both binaries built once
up front and then alternated**, baseline and patched, pass after pass - the
protocol perf pass 4 established, because this host drifts over minutes.
3 warmup + 30 runs per number, `--gpu-stream-ctx`.

| Pass   | Baseline device out | Patched device out |
| ------ | ------------------- | ------------------ |
| 1      | 232.5 ms            | 27.2 ms            |
| 2      | 235.6 ms            | 27.7 ms            |
| 3      | 235.0 ms            | 27.8 ms            |
| 4      | 235.5 ms            | 28.2 ms            |
| 5      | 235.4 ms            | 27.6 ms            |
| median | **235.4 ms**        | **27.7 ms**        |

| Pass   | Baseline host out | Patched host out |
| ------ | ----------------- | ---------------- |
| 1      | 377.5 ms          | 155.4 ms         |
| 2      | 391.1 ms          | 155.0 ms         |
| 3      | 374.3 ms          | 173.9 ms         |
| median | **377.5 ms**      | **155.4 ms**     |

Device output is **8.5× faster** (0.90 → 7.65 GB/s), won in 5 of 5 passes; host
output is **2.4× faster** (0.56 → 1.36 GB/s), won in 3 of 3. The steady-state
device wall lands at 27.7 ms against the ~16 ms floor the issue named (a ~12 ms
device-resident decode plus a ~4 ms compressed H2D), so the residual is now
~12 ms rather than ~213 ms. That residual is not isolated here and no claim is
made about it.

**Host output does not reach the same place, and the reason is not the wave.**
Its D2H targets pageable caller memory one chunk at a time, so it is 3239
synchronous copies whatever the wave size; the wave change removes the
submission cost around them and leaves the copies. That path is what issues
#133 and #135 are about.

**The price, stated rather than implied.** Peak staging rises from ~4 MB per
wave to at most the budget: device memory (compressed source plus, for host
output, the destination arena) stays within 384 MiB by construction, and pinned
host memory within the same bound for the source. For Silesia that is ~102 MB
pinned and ~102 MB device on the device-output path. One exception is
deliberate: a single chunk larger than the whole budget still decodes, alone in
its own wave, because the budget sizes a wave and is not a capacity limit on
the ABI.

Cold (first decode on a fresh context) moves the other way and is reported for
the same reason: 241.5 → 152.5 ms on the device path, so the larger allocation
costs more to make and still finishes ahead of the baseline's submission count.

`stream_twin` - the same input decoded on a reused context is bit-identical to
a fresh-context decode - stays green, along with the rest of the suite
(`100% tests passed, 0 tests failed out of 36`).

## M2: the frame path end to end, swept over block count (issue #132)

The frame entry point had no bench at all, so the per-block choreography
inside it was unobservable and none of the changes proposed against it could
be gated on a number. `bench_lz4 --frame` times `cudec_lz4f_decompress` with
the wall clock around the whole synchronous call, so the H2D, the decode, the
D2H, the assembly and both checksums are inside the measurement. Excluding the
transfers here would hide exactly what is being asked about.

The independent variable is the block count. The same Silesia bytes are
recompressed into one block-independent frame at each of the three block-max
sizes liblz4's frame format offers below 4 MB, so the block count moves 16x
over an unchanged corpus. Recorded 2026-08-09 inside the digest-pinned
`nvidia/cuda:12.6.2-devel-ubuntu24.04` container on the RTX 3080 (sm_86,
driver 12.6, runtime 12.6), 3 warmup + 30 measured runs, output byte-verified
against the original once per rung before anything is timed. Reproduce with
`bench_lz4 --frame bench/corpora/silesia/* --warmup 3 --runs 30`.

| Block max | Blocks decoded | Frame size | Wall p50 / p90 / p99        | End-to-end p50 |
| --------- | -------------- | ---------- | --------------------------- | -------------- |
| 64 KB     | 3234           | 102.46 MB  | 557.78 / 684.81 / 711.93 ms | 0.380 GB/s     |
| 256 KB    | 809            | 101.60 MB  | 455.19 / 611.15 / 782.14 ms | 0.466 GB/s     |
| 1 MB      | 203            | 101.06 MB  | 453.29 / 484.62 / 488.57 ms | 0.468 GB/s     |

**The block count costs about 34 microseconds a block.** Between the extreme
rungs, 3031 fewer blocks over the same 211.94 MB take 104.49 ms off the wall,
which is 19% of the 64 KB rung. That is a per-block cost rather than a
per-byte one, and it is the shape a per-block synchronous D2H produces. The
middle rung does not sit exactly on the line the two extremes draw - the fit
predicts 474 ms against a measured 455 - so 34 microseconds is a two-point
estimate and not a fitted constant. Its p90 of 611 ms against a p50 of 455 ms
says most of what the disagreement is: this path's run-to-run spread is wide,
and it narrows as the block count falls.

**The frame path runs at roughly a fortieth of the kernel it wraps.** The
device-resident batch row above decodes the same corpus at 16-18 GB/s; through
the frame entry point the same bytes come out at 0.38-0.47 GB/s. Most of that
is the transfers, which the device-resident number excludes by design and
which any end-to-end path has to pay. What the sweep separates out is the part
that is not the transfers: the ~450 ms all three rungs share, and the ~104 ms
the 64 KB rung pays on top for having 16x the blocks.

No decoder code moved for these numbers. This is the harness the frame-path
changes are to be measured against, and it exists so that they can be
rejected the way perf pass 3 was.

### Frame assembly: merging the per-block device-to-host copies (issue #133)

The sweep above priced the block count at roughly 34 microseconds a block, and
the shape that produces it is one blocking `cudaMemcpy` per compressed block in
the assembly loop. Assembly now issues those copies on one non-default stream
with a single terminal synchronization, and merges consecutive blocks that are
contiguous on both the device side and the output side into one copy before
issuing anything.

**Three shapes were measured and two were rejected.** The issue proposed the
first two and neither pays:

| Shape                                                            | 64 KB   | 256 KB  | 1 MB   |
| ---------------------------------------------------------------- | ------- | ------- | ------ |
| Async copies straight into the caller's `out`, one terminal sync | -1.9 %  | +3.0 %  | +1.8 % |
| Async copies through a 32 MiB pinned bounce buffer, in waves     | -2.5 %  | +25.5 % | +10.8% |
| Merge contiguous runs, async on one stream, one terminal sync    | -29.9 % | -11.1 % | -3.1 % |

The first is the naive swap, and it is flat because `out` is the caller's
buffer and is pageable: a device-to-pageable `cudaMemcpyAsync` is synchronous
with respect to the host by specification, so the stall it was meant to remove
is still paid. The second removes the stall for real, and loses anyway - it
adds one host memcpy of the whole output, and past the 64 KB rung there are too
few submissions left for the saved latency to cover that.

The third wins because it removes submissions instead of relocating them, and
what lets it is a property of the format rather than of this corpus: every data
block but the last decodes to exactly the frame's block max, which is also the
device slot stride, so a run of full blocks is one contiguous device range.
Silesia's 3234-block frame becomes a handful of copies. Both sides of the
adjacency are tested per block, so a frame that breaks either just gets more
copies.

**Numbers.** Three interleaved passes, after / before / after / before / after /
before in one session, both binaries built from the same tree, recorded
2026-08-10 inside the digest-pinned `nvidia/cuda:12.6.2-devel-ubuntu24.04`
container (`sha256:738fba0f...`) on the RTX 3080 (sm_86, driver 560.94, CUDA
12.6, nvcc V12.6.77), 3 warmup + 30 measured runs per rung, output
byte-verified against the original once per rung before timing. Reproduce with
`bench_lz4 --frame bench/corpora/silesia/* --warmup 3 --runs 30`. Every sample
is listed rather than averaged away.

| Block max | Before p50 (3 samples)         | After p50 (3 samples)          | Change      |
| --------- | ------------------------------ | ------------------------------ | ----------- |
| 64 KB     | 549.923 / 519.201 / 545.261 ms | 374.037 / 375.946 / 381.386 ms | **-29.9 %** |
| 256 KB    | 440.591 / 438.248 / 441.267 ms | 393.992 / 389.628 / 390.543 ms | **-11.1 %** |
| 1 MB      | 448.822 / 440.027 / 442.964 ms | 429.779 / 438.233 / 423.059 ms | -3.1 %      |

The two sides do not overlap at any rung, and the gap widens with the block
count, which is what a per-block cost being removed looks like. The 1 MB rung
has only 203 blocks and correspondingly little to win.

A first attempt at this A/B was discarded rather than reported: two of its
twelve rung measurements came out near three times their neighbours, which is
host contention and not a property of either binary. The table above is a
second session with no other container running.

**What this does not claim.** The single terminal synchronization is not what
paid - that shape is the first row of the first table, and it is flat. The win
is the merge. The ~450 ms floor all three rungs share is untouched and is still
the transfers plus the host-side checksum, so the frame path still runs far
below the kernel it wraps.

### Frame source gather: pinned staging does not pay (issue #135)

**Measured-negative. No decoder code shipped.**

The frame path gathers the non-contiguous compressed blocks into one host
buffer and pushes them up in a single H2D. That buffer is a pageable
`std::vector`, so the driver stages it through pinned memory of its own before
the DMA. The claim under test was that gathering straight into pinned staging
skips that bounce copy and shows up in the end-to-end wall time.

It does not. Three interleaved passes, after / before / after / before / after /
before in one session, both binaries built from the same tree, recorded
2026-08-10 inside the digest-pinned `nvidia/cuda:12.6.2-devel-ubuntu24.04`
container on the RTX 3080 (sm_86, driver 560.94, CUDA 12.6, nvcc V12.6.77),
3 warmup + 30 measured runs per rung, output byte-verified once per rung before
timing. Baseline is `2962282`. Reproduce with
`bench_lz4 --frame bench/corpora/silesia/* --warmup 3 --runs 30`.

| Block max | Before p50 (3 samples)         | After p50 (3 samples)          | Change |
| --------- | ------------------------------ | ------------------------------ | ------ |
| 64 KB     | 399.628 / 433.568 / 424.567 ms | 432.287 / 441.727 / 433.930 ms | +4.0 % |
| 256 KB    | 425.677 / 416.421 / 422.092 ms | 419.494 / 431.301 / 414.693 ms | +0.1 % |
| 1 MB      | 472.230 / 441.407 / 452.914 ms | 444.834 / 450.241 / 443.524 ms | -2.0 % |

The two sides overlap at every rung, and they overlap in both directions: the
64 KB column reads as a regression and the 1 MB column as a win, off samples
that interleave with each other. That is one distribution, not two, so the
honest reading is flat and the percentages are noise rather than effects.

Flat is enough to refuse it under this issue's pre-registered rule, and the
reason it is flat is worth keeping. The bounce copy this removes is one pass
over 102 MB. What the timed call also pays is the gather's own memcpy over the
same bytes, the content checksum over the 212 MB of output, and the transfers
in both directions, which is the ~450 ms floor every rung shares. Pinning
102 MB is not free either, and `cudaHostAlloc` is paid per call here because
the frame entry point owns no reusable context.

Only the shape this issue names was measured: one pinned buffer of `total_src`,
one H2D. A bounded pinned wave that overlaps the gather with the transfer is a
different mechanism and is not evaluated here; overlap belongs to the streaming
context work, which owns a buffer across calls and so does not pay the pinning
per call.

## M3: the Snappy CPU denominator (issue #164)

There was no Snappy kernel when this ran. This entry is the denominator, and
the harness says so in its own report rather than leaving a reader to infer it
from the absence of a GPU row. The device numbers read against it were
recorded later and are in the M3 GPU baselines section below.

Two corpus shapes, because the format and the batch API disagree about what a
unit is. Snappy's own framing is one stream per file; the shape this library
decodes is a page, which the columnar formats emit at 64 KiB. Both are built
from the same Silesia bytes by the pinned snappy 1.2.2 compressor, verified
stream by stream against the reference decoder before anything is timed, and
reported separately.

Each report carries a digest of the corpus that was actually built, folded
over the produced streams rather than over the inputs (the inputs are already
pinned by the manifest `bench/get-corpora.sh` writes). It is order-sensitive,
so the digests below belong to the file order a glob produces. Recorded
2026-08-10 inside the digest-pinned `nvidia/cuda:12.6.2-devel-ubuntu24.04`
container, on the host CPU named in the blocks. Reproduce with
`bench_snappy --warmup 3 --runs 30 bench/corpora/silesia/*`.

```
## bench_snappy report
- decoder: CPU oracle, snappy::RawUncompress (google/snappy 1.2.2), single thread. cudec has no Snappy kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- cudec: 100
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, whole-file streams, 12 streams, 211.94 MB original, 101.35 MB compressed (ratio 0.478), compressed in-harness by the pinned snappy oracle
- corpus digest: 7bd83d9e3b24fd44 (XXH64 over per-stream length and XXH64, little-endian, in corpus order)
- stream sizes: min 5345280 / median 10085684 / max 51220480 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is snappy::RawUncompress only (no allocation, no length parse); every stream round-trip-verified against the original once before timing; percentiles are nearest-rank
- wall per run: p50 177.088 ms / p90 181.655 ms / p99 186.636 ms
- decode throughput: p50 1.197 GB/s / p90 1.167 GB/s / p99 1.136 GB/s
```

```
## bench_snappy report
- decoder: CPU oracle, snappy::RawUncompress (google/snappy 1.2.2), single thread. cudec has no Snappy kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- cudec: 100
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 64 KiB-chunked streams, 3239 streams, 211.94 MB original, 101.36 MB compressed (ratio 0.478), compressed in-harness by the pinned snappy oracle
- corpus digest: be088850546d917c (XXH64 over per-stream length and XXH64, little-endian, in corpus order)
- stream sizes: min 8066 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is snappy::RawUncompress only (no allocation, no length parse); every stream round-trip-verified against the original once before timing; percentiles are nearest-rank
- wall per run: p50 177.413 ms / p90 184.445 ms / p99 187.988 ms
- decode throughput: p50 1.195 GB/s / p90 1.149 GB/s / p99 1.127 GB/s
```

**Cutting the corpus into 64 KiB pages costs the reference decoder nothing
measurable.** 177.088 ms whole against 177.413 ms chunked is 0.18% apart, well
inside the p50-to-p99 spread of either row, and the compressed size moves by
10 KB in 101 MB. That matters for what comes later: a device number quoted
against the chunked denominator is not being flattered by a handicapped
baseline, because the two baselines are the same number.

**Snappy's reference decoder is about 2.8x slower than liblz4's on these
bytes, at the same ratio.** The LZ4 CPU-oracle row further down this file
reports 3.410 GB/s p50 over the same 211.94 MB at ratio 0.483; this one
reports 1.197 GB/s at ratio 0.478. Both are single-thread wall clock on the
same host with the timed region held to the decode call alone, so the
comparison is between the two references and says nothing about either GPU
path.

## M3: the Snappy adversarial denominators (issue #119)

Two constructed corpora rather than one, because the format reaches its
maximum element rate in two different regimes and a lock on one says nothing
about the other. Both decode to the same 64 KiB block of a single repeated
byte, at the same 3200-chunk scale as the LZ4 worst-case rows, and both carry
one element per two compressed bytes. They differ only in what an element
costs to produce: the copy chain emits four decoded bytes per element, the
literal chain emits one.

Neither is anything the reference compressor emits. A block of one repeated
byte is exactly what it collapses into a single long copy, which is the best
case rather than the worst, so both streams are built byte by byte in the
harness and the reference passes verdict on every one of them before anything
is timed. The construction is locked twice besides: the copy chain on its
compressed share and its element rate, the literal chain on its element rate
and its decoded bytes per element. A corpus that drifts off its shape reds
`bench_snappy_worst_selfcheck` or `bench_snappy_worstlit_selfcheck` rather
than quietly reporting a worst case it no longer measures.

There was still no Snappy kernel when these ran, so they are denominators and
carry no cudec number; the device rows read against them are in the M3 GPU
baselines section below. Recorded 2026-08-10 inside the digest-pinned
`nvidia/cuda:12.6.2-devel-ubuntu24.04` container. Reproduce with
`bench_snappy --worstlit --warmup 3 --runs 30` and
`bench_snappy --worst --warmup 3 --runs 30`.

```
## bench_snappy report
- decoder: CPU oracle, snappy::RawUncompress (google/snappy 1.2.2), single thread. cudec has no Snappy kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- cudec: 100
- corpus: max parse work per output byte (constructed), 64 KiB-chunked streams, 3200 streams, 209.72 MB original, 419.44 MB compressed (ratio 2.000), hand-constructed in-harness as one length-1 literal element per output byte, which the reference compressor never emits; every stream validated by the pinned snappy oracle before timing
- corpus digest: a475ef89da01c0e4 (XXH64 over per-stream length and XXH64, little-endian, in corpus order)
- stream sizes: min 65536 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is snappy::RawUncompress only (no allocation, no length parse); every stream round-trip-verified against the original once before timing; percentiles are nearest-rank
- wall per run: p50 971.992 ms / p90 993.018 ms / p99 1138.249 ms
- decode throughput: p50 0.216 GB/s / p90 0.211 GB/s / p99 0.184 GB/s
- element density: 209715200 elements, 0.5000 per compressed byte, 1.0000 decoded bytes each
```

```
## bench_snappy report
- decoder: CPU oracle, snappy::RawUncompress (google/snappy 1.2.2), single thread. cudec has no Snappy kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- cudec: 100
- corpus: max element density (constructed), 64 KiB-chunked streams, 3200 streams, 209.72 MB original, 104.88 MB compressed (ratio 0.500), hand-constructed in-harness as back-to-back minimum-cost copies at offset 1, which the reference compressor never emits; every stream validated by the pinned snappy oracle before timing
- corpus digest: 1a7d7e76546da248 (XXH64 over per-stream length and XXH64, little-endian, in corpus order)
- stream sizes: min 65536 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is snappy::RawUncompress only (no allocation, no length parse); every stream round-trip-verified against the original once before timing; percentiles are nearest-rank
- wall per run: p50 1222.377 ms / p90 1328.367 ms / p99 1423.007 ms
- decode throughput: p50 0.172 GB/s / p90 0.158 GB/s / p99 0.147 GB/s
- element density: 52432000 elements, 0.4999 per compressed byte, 3.9998 decoded bytes each
```

**Both adversarial corpora cost the reference decoder between five and seven
times what Silesia does, per output byte.** 0.216 and 0.172 GB/s p50 against
1.195 GB/s on the 64 KiB-chunked Silesia rows above, over the same 64 KiB unit
and a comparable total. That is the margin these corpora exist to measure, and
it is the number a later device row on the same corpora is read against.

**The parse-bound corpus is the faster of the two per output byte, which is
the opposite of what its element count predicts.** It carries four times the
elements per decoded byte (209,715,200 against 52,432,000 for the same 209.72
MB) and still finishes 20% sooner. Per element the ordering reverses and the
gap is wide: 215.8 M elements/s on the literal chain against 42.9 M on the
copy chain, five times apart. The reference's offset-1 copy is the cost, not
its parse, and a stream of the cheapest possible copies is a stream of
byte-at-a-time overlapping copies.

**That ordering belongs to this decoder and must not be carried over to the
device rows.** A warp-cooperative decoder attacks the two regimes with
different machinery: the offset-1 gather is a closed-form modular read rather
than a serial copy, while the parse chain stays serial per chunk. Which corpus
floors the GPU is a measurement nobody has taken, and neither this section nor
the element counts settle it. It is recorded here so that whoever takes it has
the CPU ordering in front of them rather than an assumption.

## M3: first recorded Snappy GPU baselines and the ParseOnly ceiling (issue #167)

The first device numbers for the Snappy path, read against the CPU
denominators recorded in the two sections above. Every block below carries its
own CPU rows, so the ratio in each one is against that run's own denominator
rather than against a number quoted from another session.

What is timed. `cudec_snappy_decompress_batch`, device-resident: the
compressed batch is uploaded once and the timed region is the decode launch
alone, CUDA-event timed, so H2D and D2H are excluded. Before any timing the
batch is decoded once and every chunk is checked to return `CUDEC_OK` with
exactly its original byte count; the harness refuses to report a number for a
batch that did not.

The ParseOnly ceiling is instantiable for Snappy and is reported. It is the
same `chunk_decode_batch` with the copies elided, reached through the parser
template seam rather than through a second kernel, so the parse it measures is
the parse the shipped path runs and not a copy of it. That is what makes it a
ceiling on this design and on the phase 1 of any two-phase design that would
share the parse.

Recorded 2026-08-25 inside the digest-pinned
`nvidia/cuda:12.6.2-devel-ubuntu24.04` container
(`sha256:738fba0fbdb225b7a2931c58a5c8f03a84d3cd2f6a84975826a157339ef750b8`,
nvcc 12.6) on the RTX 3080 named in each block, at 3 warmup + 30 measured
runs. Reproduce with `bench_snappy --gpu bench/corpora/silesia/*`,
`bench_snappy --worst --gpu` and `bench_snappy --worstlit --gpu`. The Silesia
corpus digests are the ones the CPU denominator section above recorded, so the
two sections were built from byte-identical corpora.

```
## bench_snappy report
- decoder: CPU oracle, snappy::RawUncompress (google/snappy 1.2.2), single thread. The GPU rows below time cudec's own decoder through cudec_snappy_decompress_batch, and the CPU rows are the denominator they are read against
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 13.3, runtime 12.6
- cudec: 100
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 64 KiB-chunked streams, 3239 streams, 211.94 MB original, 101.36 MB compressed (ratio 0.478), compressed in-harness by the pinned snappy oracle
- corpus digest: be088850546d917c (XXH64 over per-stream length and XXH64, little-endian, in corpus order)
- stream sizes: min 8066 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is snappy::RawUncompress only (no allocation, no length parse); every stream round-trip-verified against the original once before timing; percentiles are nearest-rank
- wall per run: p50 191.581 ms / p90 202.740 ms / p99 206.972 ms
- decode throughput: p50 1.106 GB/s / p90 1.045 GB/s / p99 1.024 GB/s
- GPU decode (device-resident, CUDA-event timed, 3 warmup + 30 runs, 3239 chunks, every chunk verified to its original size before timing): p50 15.574 ms, 13.609 GB/s
- GPU parse-only ceiling (copies elided, the identical lockstep parse through the chunk-decoder template seam): p50 10.746 ms, 19.723 GB/s
- GPU vs the CPU denominator in this report: 12.30x (CPU p50 191.581 ms, GPU p50 15.574 ms)
```

```
## bench_snappy report
- decoder: CPU oracle, snappy::RawUncompress (google/snappy 1.2.2), single thread. The GPU rows below time cudec's own decoder through cudec_snappy_decompress_batch, and the CPU rows are the denominator they are read against
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 13.3, runtime 12.6
- cudec: 100
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, whole-file streams, 12 streams, 211.94 MB original, 101.35 MB compressed (ratio 0.478), compressed in-harness by the pinned snappy oracle
- corpus digest: 7bd83d9e3b24fd44 (XXH64 over per-stream length and XXH64, little-endian, in corpus order)
- stream sizes: min 5345280 / median 10085684 / max 51220480 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is snappy::RawUncompress only (no allocation, no length parse); every stream round-trip-verified against the original once before timing; percentiles are nearest-rank
- wall per run: p50 190.693 ms / p90 197.636 ms / p99 209.044 ms
- decode throughput: p50 1.111 GB/s / p90 1.072 GB/s / p99 1.014 GB/s
- GPU decode (device-resident, CUDA-event timed, 3 warmup + 30 runs, 12 chunks, every chunk verified to its original size before timing): p50 3381.903 ms, 0.063 GB/s
- GPU parse-only ceiling (copies elided, the identical lockstep parse through the chunk-decoder template seam): p50 1805.522 ms, 0.117 GB/s
- GPU vs the CPU denominator in this report: 0.06x (CPU p50 190.693 ms, GPU p50 3381.903 ms)
```

```
## bench_snappy report
- decoder: CPU oracle, snappy::RawUncompress (google/snappy 1.2.2), single thread. The GPU rows below time cudec's own decoder through cudec_snappy_decompress_batch, and the CPU rows are the denominator they are read against
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 13.3, runtime 12.6
- cudec: 100
- corpus: max element density (constructed), 64 KiB-chunked streams, 3200 streams, 209.72 MB original, 104.88 MB compressed (ratio 0.500), hand-constructed in-harness as back-to-back minimum-cost copies at offset 1, which the reference compressor never emits; every stream validated by the pinned snappy oracle before timing
- corpus digest: 1a7d7e76546da248 (XXH64 over per-stream length and XXH64, little-endian, in corpus order)
- stream sizes: min 65536 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is snappy::RawUncompress only (no allocation, no length parse); every stream round-trip-verified against the original once before timing; percentiles are nearest-rank
- wall per run: p50 1356.534 ms / p90 1420.085 ms / p99 1438.338 ms
- decode throughput: p50 0.155 GB/s / p90 0.148 GB/s / p99 0.146 GB/s
- GPU decode (device-resident, CUDA-event timed, 3 warmup + 30 runs, 3200 chunks, every chunk verified to its original size before timing): p50 26.082 ms, 8.041 GB/s
- GPU parse-only ceiling (copies elided, the identical lockstep parse through the chunk-decoder template seam): p50 17.912 ms, 11.708 GB/s
- GPU vs the CPU denominator in this report: 52.01x (CPU p50 1356.534 ms, GPU p50 26.082 ms)
- element density: 52432000 elements, 0.4999 per compressed byte, 3.9998 decoded bytes each
```

```
## bench_snappy report
- decoder: CPU oracle, snappy::RawUncompress (google/snappy 1.2.2), single thread. The GPU rows below time cudec's own decoder through cudec_snappy_decompress_batch, and the CPU rows are the denominator they are read against
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 13.3, runtime 12.6
- cudec: 100
- corpus: max parse work per output byte (constructed), 64 KiB-chunked streams, 3200 streams, 209.72 MB original, 419.44 MB compressed (ratio 2.000), hand-constructed in-harness as one length-1 literal element per output byte, which the reference compressor never emits; every stream validated by the pinned snappy oracle before timing
- corpus digest: a475ef89da01c0e4 (XXH64 over per-stream length and XXH64, little-endian, in corpus order)
- stream sizes: min 65536 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is snappy::RawUncompress only (no allocation, no length parse); every stream round-trip-verified against the original once before timing; percentiles are nearest-rank
- wall per run: p50 1078.347 ms / p90 1100.253 ms / p99 1107.216 ms
- decode throughput: p50 0.194 GB/s / p90 0.191 GB/s / p99 0.189 GB/s
- GPU decode (device-resident, CUDA-event timed, 3 warmup + 30 runs, 3200 chunks, every chunk verified to its original size before timing): p50 58.537 ms, 3.583 GB/s
- GPU parse-only ceiling (copies elided, the identical lockstep parse through the chunk-decoder template seam): p50 38.142 ms, 5.498 GB/s
- GPU vs the CPU denominator in this report: 18.42x (CPU p50 1078.347 ms, GPU p50 58.537 ms)
- element density: 209715200 elements, 0.5000 per compressed byte, 1.0000 decoded bytes each
```

**The parse is the bound on every corpus, and it is not close.** Eliding both
copy loops removes 31% of the chunked-Silesia decode time (15.574 ms to 10.746
ms), 31% on the copy chain (26.082 to 17.912) and 35% on the parse chain
(58.537 to 38.142). Two thirds of the kernel's time is the redundant 32-lane
lockstep parse in all three regimes. That is the same conclusion the LZ4 pass
reached, and it is now measured for Snappy rather than transferred: the lever
filed against these baselines that attacks the parse (#161) has a two-thirds
share to move, and the ones that attack the copy stage (#163, #165) have a
third.

**The device is floored by the parse-bound corpus, not by the copy chain, and
the CPU ordering is the opposite.** The adversarial-denominator section above
records the reference decoder finishing the parse chain 20% SOONER than the
copy chain, and says explicitly that which corpus floors the GPU is a
measurement nobody has taken. Taken here: the copy chain runs at 8.041 GB/s
and the parse chain at 3.583 GB/s, so the GPU is 2.2x apart in the direction
the CPU is 1.25x apart the other way. The reason is the machinery each regime
meets. The reference pays a serial byte-at-a-time overlapping copy for an
offset-1 element, which the kernel answers with the closed-form modular
gather; the parse stays serial per chunk on both. **So the DoS-resistance
margin for M3, and the number the pre-registered accept rule of #161, #163 and
#165 turns on, is the 3.583 GB/s parse-bound row rather than the copy chain.**

**Against LZ4 on the same device, the same protocol and a comparable corpus,
Snappy decodes slower and its ceiling is lower.** The M1 Silesia rows report
18.1 GB/s decode with a 34.6 GB/s parse-only ceiling; the chunked Silesia rows
here report 13.609 and 19.723. The two corpora are the same Silesia bytes at a
near-identical ratio (0.483 against 0.478), so what is left is the format: a
Snappy copy element carries at most 64 bytes where an LZ4 match is unbounded,
so the same output costs more elements and the serial parse chain per chunk is
longer. The adversarial rows land in the same place, LZ4's worst-4Bmatch at
8.1 GB/s with a 15.3 GB/s ceiling against Snappy's copy chain at 8.041 and
11.708.

**The whole-file shape is a statement about the batch shape rather than about
the decoder, and it is recorded here so that nobody quotes it as one.** Twelve
streams is twelve warps on a device with 68 SMs, so that row measures one warp
decoding up to 51 MB serially: 0.063 GB/s, 17x SLOWER than the
single-threaded reference. The chunked shape over the identical bytes is 216x
faster. That is the whole argument for the 64 KiB page unit the batch API is
built around, and it is why the chunked row and not this one is the M3
baseline. A caller holding whole-file Snappy streams has to cut them before
this API can help, and the format's own framing does not do it for them.

### M3 perf lever (issue #161): the bounded 5-byte header window is rejected by both worst cases

The one parse-side lever the format makes available. A Snappy element header
is a tag byte plus at most four length or offset bytes, so a single
fixed-width burst covers a whole header and field extraction becomes register
shifts instead of loads that wait on the tag. LZ4 has no such lever - its
token carries `0xFF` continuation chains, so no fixed lookahead covers one of
its headers - which is why no prior pass covers this.

What was measured. In `SnappyParser::Next`, `window` is filled with
`min(5, src_size - src_pos)` bytes in one counted loop of compile-time bound
five before the tag is examined, and the literal length and the copy offset
are then extracted from it by shift and mask, replacing the two data-dependent
byte loops. The two header-truncation checks stay exactly where they were and
say exactly what they said, so no byte above the fill is ever consumed and the
loop cannot touch a byte past the chunk. The masterplan's warning about a
blanket lookahead is about the CHECK, which this lever does not change; it is
the loads that were made blanket, and this pass is the measurement of that.

Measured 2026-08-25 in the digest-pinned
`nvidia/cuda:12.6.2-devel-ubuntu24.04` container on the RTX 3080, baseline and
patched binaries built from the same tree and A/B-interleaved in one session,
3 warmup + 30 CUDA-event-timed runs per invocation. All 49 ctest gates were
green on the patched parser first. Two interleaved passes on the three
64 KiB-chunked corpora; one on the whole-file shape, whose 3.4-second runs buy
no precision the verdict needs.

The Delta column is throughput speedup (`baseline_ms / window_ms − 1`,
per-pass), one basis for every row.

| Corpus                  | Baseline p50    | Header window p50 | Delta (throughput speedup) |
| ----------------------- | --------------- | ----------------- | -------------------------- |
| Silesia, 64 KiB-chunked | 16.754 / 15.793 | 17.144 / 16.860   | **−2.3% / −6.3%**          |
| `--worst` (copy chain)  | 26.654 / 26.216 | 29.904 / 29.315   | **−10.9% / −10.6%**        |
| `--worstlit` (parse)    | 61.336 / 57.424 | 86.649 / 84.934   | **−29.2% / −32.4%**        |
| Silesia, whole-file     | 3395.727        | 3198.165          | **+6.2%**                  |

The parse-only ceiling moves further in both directions, which is what
identifies the lever as acting where it was aimed rather than somewhere else:

| Corpus                  | Baseline ceiling p50 | Header window ceiling p50 | Delta               |
| ----------------------- | -------------------- | ------------------------- | ------------------- |
| Silesia, 64 KiB-chunked | 11.830 / 11.155      | 11.969 / 11.652           | **−1.2% / −4.3%**   |
| `--worst` (copy chain)  | 17.823 / 18.012      | 19.774 / 19.460           | **−9.9% / −7.4%**   |
| `--worstlit` (parse)    | 40.362 / 38.893      | 64.381 / 64.640           | **−37.3% / −39.8%** |
| Silesia, whole-file     | 1822.823             | 1604.276                  | **+13.6%**          |

**The window is fixed and the headers are not, and the loss is that
difference times the element rate.** The parse-bound corpus is one length-1
literal per output byte, so every header there is the tag alone: the lever
issues five loads where one is wanted, at the highest element rate the format
admits, and pays 29-32% of the decode and 37-40% of the parse. The copy chain
is all two-byte headers - tag plus one offset byte - so it pays 2.5 bytes of
traffic for every one it needs and loses 10.6-10.9%. Silesia's mixture of
inline literals and copy forms is the least dense of the three and loses the
least, 2.3-6.3%. The ordering across the three corpora is the ordering of
their header waste, which is the mechanism rather than a coincidence.

**The one win names its own condition and it is the condition the batch API
exists to avoid.** The whole-file shape is twelve streams, so twelve warps on
a device with 68 SMs, and there nothing else is competing for issue slots:
loads sent early hide latency that no other warp is covering, and the lever
gains 6.2% on the decode and 13.6% on the ceiling. Every corpus that occupies
the device loses. So the lever is a latency-hiding trick that only pays while
the device is idle, and a batch too small to occupy the device has a cheaper
answer than a parser change.

**Rejected under the pre-registered accept rule** (issue #161: recorded
improvement on at least one corpus with zero regression on the Snappy
worst-case corpus). Both worst cases regress, and the parse-bound one - which
the M3 baselines section above establishes as the row the DoS-resistance
margin is read on - regress the hardest. No kernel code shipped, and
`src/snappy_block.h` is untouched by this pass.

**What the negative closes and what it leaves open.** It closes the fixed-
width burst specifically: any variant that loads a constant number of header
bytes pays the same waste at the same rate, so there is nothing to retune
here. It does not close the parse chain, which the M3 baselines measured at
two thirds of the kernel's time on every corpus. A lever that reads only the
bytes the tag implies while still removing the dependency - the tag alone
first, then a single width-selected load - was not built or measured, and is
the shape a future attempt would have to take.

### M3 perf lever (issue #163): the short-copy predicated pass is not separable from noise

Snappy caps a copy element at 64 bytes, so at a 32-lane warp most elements are
at most two strided iterations and usually one. The hypothesis was that the
strided loop's setup and back edge, rather than the copy, are what the short
regime pays, and that a single predicated pass would remove them.

What was measured. In `chunk_decode_batch`, both copy stages gained a short
arm: `if (len <= kWarpSize) { if (lane < len) <one store> } else { <the
existing strided loop> }`, on the literal copy and on the 32-bit match gather.
Every test is warp-uniform - all lanes parsed the same element - so no lane can
leave early and strand the others at a `__syncwarp()`, and the closed-form
modular gather is unchanged in both arms. All 49 ctest gates were green on the
patched kernel first.

**The parse-only ceiling is the drift control, and it is what makes this
verdict readable.** ParseOnly elides both copy stages, so this lever cannot
touch it by construction: any difference in that row between the two sides is
session drift, measured in the same invocation as the number it is controlling
for. That is not a device this project can reserve, and the control says how
much of any delta is the machine.

Measured 2026-08-25 in the digest-pinned
`nvidia/cuda:12.6.2-devel-ubuntu24.04` container on the RTX 3080, baseline and
patched binaries built from the same tree and A/B-interleaved in one session,
3 warmup + 30 CUDA-event-timed runs per invocation. The worst cases ran FIRST,
which is the trap this lever walks into. Two interleaved passes on the three
64 KiB-chunked corpora, one on the whole-file shape. Every column is
throughput speedup, `baseline_ms / short_arm_ms − 1`.

| Corpus                  | Pass | Decode     | Ceiling (control) | Decode less control |
| ----------------------- | ---- | ---------- | ----------------- | ------------------- |
| `--worst` (copy chain)  | 1    | **+7.75%** | **+9.10%**        | **−1.35 pp**        |
| `--worst` (copy chain)  | 2    | −0.10%     | −0.38%            | +0.28 pp            |
| `--worstlit` (parse)    | 1    | +1.19%     | −0.21%            | +1.40 pp            |
| `--worstlit` (parse)    | 2    | +0.77%     | +0.34%            | +0.43 pp            |
| Silesia, 64 KiB-chunked | 1    | +0.96%     | +1.04%            | −0.08 pp            |
| Silesia, 64 KiB-chunked | 2    | −1.48%     | −1.88%            | +0.40 pp            |
| Silesia, whole-file     | 1    | +1.40%     | +1.04%            | +0.36 pp            |

**The first row is the reason the control is in the table.** Read alone it is
a 7.75% win on the adversarial copy corpus, which is exactly the result this
lever was built to find, on exactly the corpus the accept rule turns on. The
ceiling beside it moved 9.10% in the same direction, on a code path the lever
does not contain, and the second pass of the same corpus has both columns at
zero. That cell is the first invocation of the session on a cold clock, and
without a control measured in the same run it would have been recorded as the
lever's headline number.

**Corrected, every cell straddles zero: −1.35 to +1.40 percentage points,
against a control that itself swings ±1.9%.** There is no corpus on which an
improvement can be recorded, so the pre-registered accept rule (issue #163:
recorded improvement on at least one corpus with zero regression on the Snappy
worst-case corpus) is not met on its first clause. Rejected. No kernel code
shipped; `src/chunk_decode.cuh` is untouched.

**What this retires is the hypothesis rather than the lever.** Snappy's copy
stage is not loop-overhead-bound: removing the loop from the short regime
entirely changes nothing measurable, on a corpus whose every element is a
4-byte copy. The upside was bounded before the measurement anyway - the M3
baselines above put the whole copy stage at 31-35% of the kernel's time - and
this says the setup inside that share is not where it goes.

**And the #36 trap did not transfer, which is worth recording in its own
right.** Perf pass 1-3 rejected an LZ4 match-copy fast path because the added
per-element predicate cost 5-9% at maximum sequence density. The same shape of
predicate, added to the same kernel for Snappy, costs nothing measurable on
either Snappy worst case. The difference is what the predicate buys: LZ4's
arm was never taken on the offset-1 corpus, so it was pure cost, while
Snappy's short arm is taken by every element there. A future lever in this
stage does not inherit a debt from that pass.

**The second candidate named in the issue was not built, and the reason is a
settled design decision rather than effort.** Sub-warp fan-out - several
elements copying concurrently instead of idling lanes on a short one -
requires knowing the next element before the current copy retires, which means
parsing ahead of the copy. This kernel's parse is redundant across all 32
lanes in lockstep and single-pass, settled at the #85 panel and in the
masterplan's decomposition question; running the parse ahead of the copies is
the two-phase design that was measured and refused, not a fan-out variant. So
that candidate is a re-opening of #15 and not a lever this pass could take.

**One disclosure about the shape as built.** The short arm went into the
shared `chunk_decode_batch`, so it would have changed the LZ4 instantiation
too. Nothing ships, so no LZ4 corpora were re-measured and none of the M1 rows
above are touched or restated by this pass. A future variant that did win
would owe either those LZ4 numbers or a template gate that keeps the arm on
the Snappy instantiation alone.

### M3 perf lever (issue #165): the Snappy literal distribution, and the wide literal copy rejected a second time

Perf pass 1 rejected a vectorized literal copy for LZ4 at −6% and gave a
reason specific to a distribution: _"Silesia's literal runs are mostly < 16
bytes, so the wide path rarely triggers and only adds setup/branch
overhead."_ That is a statement about a trigger rate, so it does not transfer
to a different format's literal shape by assertion. It transfers or it does
not by measurement, and the distribution is what decides.

**The distribution first, because it is the durable half of this pass.**
Both harnesses gained a `--literals` mode (`bench/literal_hist.h`, shared so
that one set of bucket edges and one print format serve both formats - two
copies of the edges is how a comparison silently stops being one). It walks
each stream with the SHIPPED parser, so the elements counted are the elements
the decoder executes, and buckets the literal lengths. Reproduce with
`bench_snappy --literals bench/corpora/silesia/*` and
`bench_lz4 --literals bench/corpora/silesia/*`.

Silesia, 64 KiB-chunked, the same source bytes through the two compressors:

| Literal length | Snappy elements | Snappy bytes | LZ4 elements | LZ4 bytes |
| -------------- | --------------- | ------------ | ------------ | --------- |
| 1              | 38.5%           | 6.1%         | 36.0%        | 6.1%      |
| 2-3            | 29.4%           | 10.9%        | 28.1%        | 11.1%     |
| 4-7            | 17.3%           | 13.8%        | 18.9%        | 16.0%     |
| 8-15           | 11.1%           | 19.6%        | 11.7%        | 21.8%     |
| 16-31          | 2.9%            | 9.2%         | 3.8%         | 13.2%     |
| 32-63          | 0.6%            | 4.0%         | 1.0%         | 6.8%      |
| 64-255         | 0.2%            | 3.7%         | 0.4%         | 7.3%      |
| 256+           | 0.0%            | 32.7%        | 0.1%         | 17.7%     |

| Quantity                            | Snappy     | LZ4        |
| ----------------------------------- | ---------- | ---------- |
| Elements                            | 26,052,354 | 18,339,030 |
| Literal elements                    | 7,274,444  | 7,560,402  |
| Literal bytes                       | 45,773,418 | 44,916,934 |
| Wide-path trigger, literal elements | 3.7%       | 5.3%       |
| Wide-path trigger, literal bytes    | 49.6%      | 45.0%      |

The two adversarial corpora are one line each: `--worst` is 6,400 literal
elements in 52.4 M, all under four bytes, and `--worstlit` is 209,715,200
literals of exactly one byte. The wide path's trigger on both is **0.0% of
literal elements and 0.0% of literal bytes**, so those corpora can carry the
added predicate and nothing else.

**The premise this issue was written on is refuted.** It expected Snappy to
emit more and longer literals than LZ4 on the same data. It emits slightly
FEWER long ones by element share (3.7% against 5.3% at 16 bytes and up) and
about the same by byte share, and its literal bytes and literal element counts
are within 2% and 4% of LZ4's. The old rejection's premise therefore holds for
this format too.

**What Snappy's extra elements actually are, which the histogram settles as a
by-product.** Snappy carries 42% more elements than LZ4 for identical output
(26.05 M against 18.34 M) while its literal elements are 4% FEWER. Every extra
element is a copy: 18.78 M against 10.78 M, 74% more. That is the 64-byte copy
cap splitting what LZ4 spends one unbounded match on, and it is the mechanism
behind the parse-chain gap the M3 baselines above record between the two
formats. It was not the question this issue asked, and it is the answer to
the one the baselines section raised.

**The old sentence needs one correction, in the reader's favour.** "Rarely
triggers" is true of ELEMENTS and false of BYTES: at 16 bytes and up the
trigger is 3.7% of Snappy's literal elements but 49.6% of its literal bytes,
and a wide copy's work scales with bytes. So the trigger rate alone did not
settle it, and the A/B was owed rather than optional.

**The A/B.** A 4-byte-per-lane literal copy for literals of 32 bytes and up,
with a byte head and tail, guarded on the ABSOLUTE addresses rather than the
offsets - the caller's `dst` carries no alignment guarantee and a misaligned
32-bit access is undefined - so the wide arm runs where both sides share a
4-byte phase and the head aligns them. All 49 ctest gates green on the patched
kernel first. Same protocol as the two levers above: A/B-interleaved in one
session, both binaries from the same tree, 3 warmup + 30 CUDA-event-timed runs,
worst cases first, with the parse-only ceiling as the drift control - it elides
the copies, so this lever cannot touch it and any movement there is the
machine.

| Corpus                  | Pass | Decode | Ceiling (control) | Decode less control |
| ----------------------- | ---- | ------ | ----------------- | ------------------- |
| `--worst` (copy chain)  | 1    | −4.12% | +0.20%            | −4.32 pp            |
| `--worst` (copy chain)  | 2    | −3.93% | −0.05%            | −3.88 pp            |
| `--worstlit` (parse)    | 1    | −9.43% | +1.09%            | −10.52 pp           |
| `--worstlit` (parse)    | 2    | −9.50% | −0.04%            | −9.46 pp            |
| Silesia, 64 KiB-chunked | 1    | −4.72% | −1.28%            | −3.44 pp            |
| Silesia, 64 KiB-chunked | 2    | −7.33% | −1.28%            | −6.05 pp            |
| Silesia, whole-file     | 1    | −5.01% | −0.01%            | −5.00 pp            |

**Rejected under the pre-registered accept rule**, and this time with no cell
even ambiguous: every corpus regresses, both worst cases regress, and the two
that cannot trigger the wide arm at all regress the hardest. `--worstlit` at
−9.5% is the added predicate alone, paid 209.7 M times against zero wide
copies, which is the cleanest possible reading of what the arm costs. No
kernel code shipped; `src/chunk_decode.cuh` is untouched.

**The negative is stronger than perf pass 1's and it closes the question for
this format.** That pass could be read as a statement about one distribution;
this one has the distribution measured beside it, shows the byte-share trigger
at half of all literal bytes, and STILL loses on the corpus where the arm
triggers most. A wide literal path is not rejected here for failing to fire -
it fires - but for costing more than it saves, at 21.6% of the output being
literal bytes in the first place and the whole copy stage being 31-35% of the
kernel's time.

**What ships from this pass is the `--literals` mode.** The histogram is a
standing instrument rather than a one-off: the next attempt on this stage, for
any format on this seam, reads the trigger rate before writing a kernel
instead of arguing about it.

### M3 kernel shape (issue #292): the narrowed element buys occupancy and loses throughput

The last width question the Snappy seam left open. The arithmetic width was
settled with the gather (see the #58 pass above and the parser contract at the
head of `src/chunk_decode.cuh`): the copy engine already picks a 32-bit modulo
where the element admits one. What was not settled is the width the element is
CARRIED in. `DecodeSequence` is six `uint64_t`, 48 bytes, and Snappy provably
needs none of that range - its declared length is a varint32, so every position,
offset and length fits in 32 bits. The struct's size is register pressure per
lane, and the two instantiations of one kernel were already eight registers
apart.

Both arms were built and run rather than argued. The narrow arm is the same six
fields declared `uint32_t`, which halves the element to 24 bytes. It is a
measurement build and it is NOT in the tree: LZ4's `match_len` is bounded by the
caller's `size_t` capacity rather than by any field width, and the tree carries
a test that reaches past 2^32 through it, so a globally narrowed element is
wrong for the other format on this seam. What the experiment answers is whether
a SECOND, narrow element type for Snappy would be worth the panel decision it
would cost - and it is not, so the question stops here rather than at that
trade.

Read out through `cudaFuncGetAttributes` and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the shipped kernel:

```
device=NVIDIA GeForce RTX 3080 sm_86 maxWarpsPerSM=48 elementBytes=48
lz4      regs=48   blocks/SM=10  warps/SM=40  (of 48)
snappy   regs=56   blocks/SM=9   warps/SM=36  (of 48)

device=NVIDIA GeForce RTX 3080 sm_86 maxWarpsPerSM=48 elementBytes=24
lz4      regs=40   blocks/SM=12  warps/SM=48  (of 48)
snappy   regs=48   blocks/SM=10  warps/SM=40  (of 48)
```

So the narrowing does exactly what it was expected to do to residency: eight
registers back on both instantiations, one more resident block per SM for
Snappy, and LZ4 reaching the architectural ceiling of 48 warps/SM. On the
occupancy argument alone it wins.

It loses on the clock, on every corpus:

| corpus                 | 64-bit element | 32-bit element | delta     |
| ---------------------- | -------------- | -------------- | --------- |
| Silesia, 64 KiB chunks | 13.970 GB/s    | 12.899 GB/s    | **-7.7%** |
| worst (copy chain)     | 8.064 GB/s     | 7.553 GB/s     | **-6.3%** |
| worstlit (parse chain) | 3.612 GB/s     | 3.459 GB/s     | **-4.2%** |

Repeated on Silesia because a single pair at that spread is a claim about one
run. Three independent baseline samples against three narrow ones, alternating,
same binaries, same corpus:

```
baseline  13.970 / 13.823 / 13.471 GB/s
narrow    12.899 / 13.231 / 13.006 GB/s
```

Every baseline sample is above every narrow sample. GPU timing on this device
jitters 1-2% run to run, which is the size of the spread WITHIN each arm and
well under the gap between them.

**The parse-only ceiling does not move, and that is what locates the cost.**
Copies elided, the identical lockstep parse through the template seam:

```
baseline  19.911 / 19.693 / 18.734 GB/s
narrow    19.648 / 19.683 / 20.044 GB/s
```

The two sets interleave completely. So the narrowing does not slow the parse -
it slows the stage the parse feeds. The copy loops index global memory, where
every field is an address operand and a 32-bit field is widened again at the
point of use; the extra conversions are paid per lane per element, and they cost
more than the resident block gains. On the adversarial corpora the direction is
the same, which rules out an explanation that only holds for real data.

**Rejected, and the accept rule that rejects it is the one already registered
for this stage.** #161, #163 and #165 all turn on the parse-bound row as the
DoS-resistance margin; this arm moves it 4.2% the wrong way and the general
corpus 7.7% the wrong way. Occupancy is not the currency - the kernel was not
short of resident warps, and buying more of them by making every element access
more expensive is the trade this measurement refuses.

So the element stays 64-bit on both sides of the seam, there is no narrowing to
bounds-check, and the seam keeps ONE element contract - which is what the #85
panel decided the copy engine's safety rests on, now costing nothing rather than
costing an unmeasured amount. `src/decode_sequence.h` and `src/snappy_block.h`
carry the answer where their next reader meets it.

Recorded 2026-08-26 in the Ubuntu-24.04 WSL distribution, nvcc 13.3 (V13.3.73),
driver 610.88, RTX 3080 (sm_86), 3 warmup + 30 measured runs, device-resident
and CUDA-event timed, every chunk verified to its original size before timing.
Both arms were built and run on this one toolchain in one sitting: the M3
baselines above were taken under nvcc 12.6 in the pinned container, so the
numbers here are compared against their own arm and never against those.
Reproduce the baseline arm with `bench_snappy --gpu bench/corpora/silesia/*`,
`bench_snappy --worst --gpu` and `bench_snappy --worstlit --gpu`; the narrow arm
is those three over a build whose six `DecodeSequence` fields are `uint32_t`.

## M4: the GDeflate CPU denominator (issue #224)

There is no GDeflate kernel yet. This entry is the denominator a later device
number will be read against, and the harness says so in its own report rather
than leaving a reader to infer it from the absence of a GPU row.

Eight cells, two corpora crossed with the level set section 11.8 of the
[MASTERPLAN](MASTERPLAN.md) records. Silesia is the general-purpose half;
`asset-like` is the generated game-asset model (issue #139), which is the one
regime Silesia does not reach - most of its block is incompressible, so the
decode is dominated by literal transfer. Both harnesses read that block from
one generator, `bench/assetlike_source.h`, so the bytes a number is attested
against cannot drift between them.

The levels are 0, 1, 6 and 12, and each is there to reach something: 0 emits
uncompressed blocks by construction, which is the only block-type guarantee
the reference's own header gives; 1 is the fast end of the search; 6 is the
default; 12 is the densest table description. Section 11.8 also says plainly
that a level list is the PLAN for coverage and not the coverage argument, and
nothing here asserts which block types a family actually reached - that lock
needs a walk over the emitted page and is issue #225's.

The unit is the 64 KiB GDeflate page, and each page is compressed on its own,
which is what makes it independently decodable and what the batch surface
(#216) will take. The timed loop decodes one page per call. Before anything is
timed every page is decoded back by the reference and compared against the
source it was cut from, so a page set that dropped, reordered or truncated
part of the corpus fails instead of reporting a plausible ratio over different
bytes.

Each report carries a digest of the corpus that was actually built, folded
over the produced pages rather than over the inputs (the inputs are already
pinned by the manifest `bench/get-corpora.sh` writes). It is order-sensitive,
so the digests below belong to the file order listed in the corpus line.

Recorded 2026-08-26 in the Ubuntu-24.04 WSL distribution on the host CPU named
in the blocks, against the gdeflate fork pinned at
`8ba9502fb30d2bf728592d121f0d402e40c8cb05`. Reproduce with
`bench_gdeflate --warmup 3 --runs 30 bench/corpora/silesia/*` and
`bench_gdeflate --warmup 3 --runs 30 --assetlike`.

**Read the p50 and treat the tail with suspicion on this run.** The host was
carrying other work while these were taken - the Silesia level-0 cell spreads
from 0.692 GB/s at p50 to 0.227 GB/s at p99 - so the p90/p99 columns of the
blocks below carry that contention rather than a property of the decoder. The
p50 is the denominator this entry exists to record; the tail is reported
because the block reports it, not because it is clean.

| corpus     | level | compressed | ratio  | p50 decode | corpus digest      |
| ---------- | ----- | ---------- | ------ | ---------- | ------------------ |
| Silesia    | 0     | 212.38 MB  | 1.0021 | 0.692 GB/s | `88ed82d5ec9df3ad` |
| Silesia    | 1     | 75.89 MB   | 0.3581 | 0.313 GB/s | `131124d155e6672b` |
| Silesia    | 6     | 70.84 MB   | 0.3343 | 0.362 GB/s | `042b2473240db0b0` |
| Silesia    | 12    | 67.77 MB   | 0.3198 | 0.373 GB/s | `4b88e13a215ed884` |
| asset-like | 0     | 210.15 MB  | 1.0021 | 0.678 GB/s | `08ac6ec118b60189` |
| asset-like | 1     | 149.00 MB  | 0.7105 | 0.292 GB/s | `45690d97a5d3b054` |
| asset-like | 6     | 147.40 MB  | 0.7029 | 0.322 GB/s | `47dad3ea983dc577` |
| asset-like | 12    | 146.70 MB  | 0.6995 | 0.335 GB/s | `cb272ab3765d8378` |

The table is a reading aid. The eight blocks below are the record, and each one
carries the methodology its own numbers were taken under.

```
## bench_gdeflate report
- decoder: CPU oracle, libdeflate_gdeflate_decompress (the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05), single thread. cudec has no GDeflate kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 pages, 211.94 MB original, 212.38 MB compressed (ratio 1.0021), cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork; every page decoded back by the reference and compared against the source before timing
- granularity: 64 KiB pages, compression level 0
- corpus digest: 88ed82d5ec9df3ad (XXH64 over per-page length and XXH64, little-endian, in corpus order)
- page sizes: min 60692 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is libdeflate_gdeflate_decompress only (destinations allocated outside it); every page round-trip-verified against the source once before timing; percentiles are nearest-rank
- block-type composition: not asserted here; reading it needs a walk over the emitted page and that lock is issue #225's
- wall per run: p50 306.237 ms / p90 431.330 ms / p99 933.635 ms
- decode throughput: p50 0.692 GB/s / p90 0.491 GB/s / p99 0.227 GB/s
```

```
## bench_gdeflate report
- decoder: CPU oracle, libdeflate_gdeflate_decompress (the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05), single thread. cudec has no GDeflate kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 pages, 211.94 MB original, 75.89 MB compressed (ratio 0.3581), cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork; every page decoded back by the reference and compared against the source before timing
- granularity: 64 KiB pages, compression level 1
- corpus digest: 131124d155e6672b (XXH64 over per-page length and XXH64, little-endian, in corpus order)
- page sizes: min 60692 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is libdeflate_gdeflate_decompress only (destinations allocated outside it); every page round-trip-verified against the source once before timing; percentiles are nearest-rank
- block-type composition: not asserted here; reading it needs a walk over the emitted page and that lock is issue #225's
- wall per run: p50 676.159 ms / p90 1124.385 ms / p99 1393.539 ms
- decode throughput: p50 0.313 GB/s / p90 0.188 GB/s / p99 0.152 GB/s
```

```
## bench_gdeflate report
- decoder: CPU oracle, libdeflate_gdeflate_decompress (the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05), single thread. cudec has no GDeflate kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 pages, 211.94 MB original, 70.84 MB compressed (ratio 0.3343), cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork; every page decoded back by the reference and compared against the source before timing
- granularity: 64 KiB pages, compression level 6
- corpus digest: 042b2473240db0b0 (XXH64 over per-page length and XXH64, little-endian, in corpus order)
- page sizes: min 60692 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is libdeflate_gdeflate_decompress only (destinations allocated outside it); every page round-trip-verified against the source once before timing; percentiles are nearest-rank
- block-type composition: not asserted here; reading it needs a walk over the emitted page and that lock is issue #225's
- wall per run: p50 585.838 ms / p90 629.493 ms / p99 713.022 ms
- decode throughput: p50 0.362 GB/s / p90 0.337 GB/s / p99 0.297 GB/s
```

```
## bench_gdeflate report
- decoder: CPU oracle, libdeflate_gdeflate_decompress (the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05), single thread. cudec has no GDeflate kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 pages, 211.94 MB original, 67.77 MB compressed (ratio 0.3198), cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork; every page decoded back by the reference and compared against the source before timing
- granularity: 64 KiB pages, compression level 12
- corpus digest: 4b88e13a215ed884 (XXH64 over per-page length and XXH64, little-endian, in corpus order)
- page sizes: min 60692 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is libdeflate_gdeflate_decompress only (destinations allocated outside it); every page round-trip-verified against the source once before timing; percentiles are nearest-rank
- block-type composition: not asserted here; reading it needs a walk over the emitted page and that lock is issue #225's
- wall per run: p50 568.204 ms / p90 589.782 ms / p99 615.539 ms
- decode throughput: p50 0.373 GB/s / p90 0.359 GB/s / p99 0.344 GB/s
```

```
## bench_gdeflate report
- decoder: CPU oracle, libdeflate_gdeflate_decompress (the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05), single thread. cudec has no GDeflate kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: asset-like, 3200 pages, 209.72 MB original, 210.15 MB compressed (ratio 1.0021), generated in-harness, a MODEL of a game asset package (bench/assetlike_source.h, issue #139) and not a measurement on real game data; cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork
- granularity: 64 KiB pages, compression level 0
- corpus digest: 08ac6ec118b60189 (XXH64 over per-page length and XXH64, little-endian, in corpus order)
- page sizes: min 65536 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is libdeflate_gdeflate_decompress only (destinations allocated outside it); every page round-trip-verified against the source once before timing; percentiles are nearest-rank
- block-type composition: not asserted here; reading it needs a walk over the emitted page and that lock is issue #225's
- wall per run: p50 309.432 ms / p90 327.616 ms / p99 345.187 ms
- decode throughput: p50 0.678 GB/s / p90 0.640 GB/s / p99 0.608 GB/s
```

```
## bench_gdeflate report
- decoder: CPU oracle, libdeflate_gdeflate_decompress (the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05), single thread. cudec has no GDeflate kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: asset-like, 3200 pages, 209.72 MB original, 149.00 MB compressed (ratio 0.7105), generated in-harness, a MODEL of a game asset package (bench/assetlike_source.h, issue #139) and not a measurement on real game data; cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork
- granularity: 64 KiB pages, compression level 1
- corpus digest: 45690d97a5d3b054 (XXH64 over per-page length and XXH64, little-endian, in corpus order)
- page sizes: min 65536 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is libdeflate_gdeflate_decompress only (destinations allocated outside it); every page round-trip-verified against the source once before timing; percentiles are nearest-rank
- block-type composition: not asserted here; reading it needs a walk over the emitted page and that lock is issue #225's
- wall per run: p50 719.320 ms / p90 757.564 ms / p99 781.417 ms
- decode throughput: p50 0.292 GB/s / p90 0.277 GB/s / p99 0.268 GB/s
```

```
## bench_gdeflate report
- decoder: CPU oracle, libdeflate_gdeflate_decompress (the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05), single thread. cudec has no GDeflate kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: asset-like, 3200 pages, 209.72 MB original, 147.40 MB compressed (ratio 0.7029), generated in-harness, a MODEL of a game asset package (bench/assetlike_source.h, issue #139) and not a measurement on real game data; cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork
- granularity: 64 KiB pages, compression level 6
- corpus digest: 47dad3ea983dc577 (XXH64 over per-page length and XXH64, little-endian, in corpus order)
- page sizes: min 65536 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is libdeflate_gdeflate_decompress only (destinations allocated outside it); every page round-trip-verified against the source once before timing; percentiles are nearest-rank
- block-type composition: not asserted here; reading it needs a walk over the emitted page and that lock is issue #225's
- wall per run: p50 651.308 ms / p90 673.832 ms / p99 688.227 ms
- decode throughput: p50 0.322 GB/s / p90 0.311 GB/s / p99 0.305 GB/s
```

```
## bench_gdeflate report
- decoder: CPU oracle, libdeflate_gdeflate_decompress (the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05), single thread. cudec has no GDeflate kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: asset-like, 3200 pages, 209.72 MB original, 146.70 MB compressed (ratio 0.6995), generated in-harness, a MODEL of a game asset package (bench/assetlike_source.h, issue #139) and not a measurement on real game data; cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork
- granularity: 64 KiB pages, compression level 12
- corpus digest: cb272ab3765d8378 (XXH64 over per-page length and XXH64, little-endian, in corpus order)
- page sizes: min 65536 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is libdeflate_gdeflate_decompress only (destinations allocated outside it); every page round-trip-verified against the source once before timing; percentiles are nearest-rank
- block-type composition: not asserted here; reading it needs a walk over the emitted page and that lock is issue #225's
- wall per run: p50 626.935 ms / p90 666.084 ms / p99 682.146 ms
- decode throughput: p50 0.335 GB/s / p90 0.315 GB/s / p99 0.307 GB/s
```

## M4 perf lever: the block-type mix, retired on the census (issue #206)

**The lever, and why it is answered here instead of on a device.** Stored and
static-Huffman blocks are the cheap block types: a stored block carries no
table and a static one uses the fixed code. A decode path specialised for them
would be a second route through a security-critical round loop, and every
branch of it would have to be re-covered by the validation ladder. What it
could buy is bounded by how much of a real corpus is made of those blocks, and
that is a property of what the compressor emits rather than of any kernel. So
the count decides the lever before a kernel exists, which is the order #206
fixed: count first, then decide.

**The pre-registered rule this run was taken under**, written on #206 before
the measurement: only if the mix is materially non-trivial is the
specialisation implemented and measured, and "the mix does not justify a second
path" is a full result to be recorded as one. The forced stored/static corpora
in the M4 corpus set are decode-path coverage and are excluded from the
argument by the same rule, because they are synthetic by construction.

**Counting the openings would have answered the wrong question, and it would
have answered it in the direction that flatters the lever.** A GDeflate block
boundary is not findable by scanning (section 11.3), so the walk in
`bench_gdeflate --blocktypes` reads the first block header of each page and
stops. A stored block's length field is sixteen bits, and 65535 is one byte
short of a page, so a full page of stored data is at least two stored blocks,
always, while a page that uses a table can be one. A histogram of openings
therefore undercounts exactly the block type the lever is about. What reaches a
page's second block is a decode, so this census walks every page with cudec's
own page decoder (`src/gdeflate_block.h`) and reads the per-type counts it
produces.

The census is taken over the same corpora and the same compressor pin as the M4
CPU denominator above -- the eight corpus digests below reproduce the eight
recorded in that entry byte for byte, which is what makes this a census of that
corpus rather than of one resembling it. Recorded 2026-08-30. Reproduce with
`bench_gdeflate --blockmix bench/corpora/silesia/*` and
`bench_gdeflate --blockmix --assetlike`; nothing is timed and no figure here is
a denominator.

| corpus     | level | pages | blocks | blocks/page | stored blocks | static blocks | dynamic blocks | stored bytes | static bytes | dynamic bytes |
| ---------- | ----- | ----- | ------ | ----------- | ------------- | ------------- | -------------- | ------------ | ------------ | ------------- |
| Silesia    | 0     | 3234  | 6467   | 2.000       | 100.0000%     | 0.0000%       | 0.0000%        | 100.0000%    | 0.0000%      | 0.0000%       |
| Silesia    | 1     | 3234  | 6742   | 2.085       | 0.0148%       | 0.0148%       | 99.9703%       | 0.0057%      | 0.0059%      | 99.9883%      |
| Silesia    | 6     | 3234  | 6767   | 2.092       | 0.0296%       | 0.0443%       | 99.9261%       | 0.0115%      | 0.0255%      | 99.9630%      |
| Silesia    | 12    | 3234  | 7308   | 2.260       | 0.0137%       | 0.0137%       | 99.9726%       | 0.0057%      | 0.0096%      | 99.9847%      |
| asset-like | 0     | 3200  | 6400   | 2.000       | 100.0000%     | 0.0000%       | 0.0000%        | 100.0000%    | 0.0000%      | 0.0000%       |
| asset-like | 1     | 3200  | 9600   | 3.000       | 0.0000%       | 0.0000%       | 100.0000%      | 0.0000%      | 0.0000%      | 100.0000%     |
| asset-like | 6     | 3200  | 9600   | 3.000       | 0.0000%       | 0.0000%       | 100.0000%      | 0.0000%      | 0.0000%      | 100.0000%     |
| asset-like | 12    | 3200  | 9600   | 3.000       | 0.0000%       | 0.0000%       | 100.0000%      | 0.0000%      | 0.0000%      | 100.0000%     |

The table is a reading aid. The eight blocks below are the record.

```
## bench_gdeflate block-type mix
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 pages, 211.94 MB original, ratio 1.0021, cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork; every page decoded back by the reference and compared against the source before timing
- compressor: the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05, compression level 0
- corpus digest: 88ed82d5ec9df3ad
- read by: cudec's own page decoder (src/gdeflate_block.h), every page required to decode to its source bytes and every block and byte attributed to a type; this is a whole-page census and not a walk over openings
- blocks: 6467 over 3234 pages (2.000 per page), 211938580 output bytes
| type    | blocks | share of blocks | output bytes | share of bytes |
| ------- | ------ | --------------- | ------------ | -------------- |
| stored  |   6467 |       100.0000% |    211938580 |      100.0000% |
| static  |      0 |         0.0000% |            0 |        0.0000% |
| dynamic |      0 |         0.0000% |            0 |        0.0000% |
```

```
## bench_gdeflate block-type mix
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 pages, 211.94 MB original, ratio 0.3581, cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork; every page decoded back by the reference and compared against the source before timing
- compressor: the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05, compression level 1
- corpus digest: 131124d155e6672b
- read by: cudec's own page decoder (src/gdeflate_block.h), every page required to decode to its source bytes and every block and byte attributed to a type; this is a whole-page census and not a walk over openings
- blocks: 6742 over 3234 pages (2.085 per page), 211938580 output bytes
| type    | blocks | share of blocks | output bytes | share of bytes |
| ------- | ------ | --------------- | ------------ | -------------- |
| stored  |      1 |         0.0148% |        12100 |        0.0057% |
| static  |      1 |         0.0148% |        12604 |        0.0059% |
| dynamic |   6740 |        99.9703% |    211913876 |       99.9883% |
```

```
## bench_gdeflate block-type mix
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 pages, 211.94 MB original, ratio 0.3343, cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork; every page decoded back by the reference and compared against the source before timing
- compressor: the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05, compression level 6
- corpus digest: 042b2473240db0b0
- read by: cudec's own page decoder (src/gdeflate_block.h), every page required to decode to its source bytes and every block and byte attributed to a type; this is a whole-page census and not a walk over openings
- blocks: 6767 over 3234 pages (2.092 per page), 211938580 output bytes
| type    | blocks | share of blocks | output bytes | share of bytes |
| ------- | ------ | --------------- | ------------ | -------------- |
| stored  |      2 |         0.0296% |        24378 |        0.0115% |
| static  |      3 |         0.0443% |        53991 |        0.0255% |
| dynamic |   6762 |        99.9261% |    211860211 |       99.9630% |
```

```
## bench_gdeflate block-type mix
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 pages, 211.94 MB original, ratio 0.3198, cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork; every page decoded back by the reference and compared against the source before timing
- compressor: the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05, compression level 12
- corpus digest: 4b88e13a215ed884
- read by: cudec's own page decoder (src/gdeflate_block.h), every page required to decode to its source bytes and every block and byte attributed to a type; this is a whole-page census and not a walk over openings
- blocks: 7308 over 3234 pages (2.260 per page), 211938580 output bytes
| type    | blocks | share of blocks | output bytes | share of bytes |
| ------- | ------ | --------------- | ------------ | -------------- |
| stored  |      1 |         0.0137% |        12092 |        0.0057% |
| static  |      1 |         0.0137% |        20300 |        0.0096% |
| dynamic |   7306 |        99.9726% |    211906188 |       99.9847% |
```

```
## bench_gdeflate block-type mix
- corpus: asset-like, 3200 pages, 209.72 MB original, ratio 1.0021, generated in-harness, a MODEL of a game asset package (bench/assetlike_source.h, issue #139) and not a measurement on real game data; cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork
- compressor: the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05, compression level 0
- corpus digest: 08ac6ec118b60189
- read by: cudec's own page decoder (src/gdeflate_block.h), every page required to decode to its source bytes and every block and byte attributed to a type; this is a whole-page census and not a walk over openings
- blocks: 6400 over 3200 pages (2.000 per page), 209715200 output bytes
| type    | blocks | share of blocks | output bytes | share of bytes |
| ------- | ------ | --------------- | ------------ | -------------- |
| stored  |   6400 |       100.0000% |    209715200 |      100.0000% |
| static  |      0 |         0.0000% |            0 |        0.0000% |
| dynamic |      0 |         0.0000% |            0 |        0.0000% |
```

```
## bench_gdeflate block-type mix
- corpus: asset-like, 3200 pages, 209.72 MB original, ratio 0.7105, generated in-harness, a MODEL of a game asset package (bench/assetlike_source.h, issue #139) and not a measurement on real game data; cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork
- compressor: the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05, compression level 1
- corpus digest: 45690d97a5d3b054
- read by: cudec's own page decoder (src/gdeflate_block.h), every page required to decode to its source bytes and every block and byte attributed to a type; this is a whole-page census and not a walk over openings
- blocks: 9600 over 3200 pages (3.000 per page), 209715200 output bytes
| type    | blocks | share of blocks | output bytes | share of bytes |
| ------- | ------ | --------------- | ------------ | -------------- |
| stored  |      0 |         0.0000% |            0 |        0.0000% |
| static  |      0 |         0.0000% |            0 |        0.0000% |
| dynamic |   9600 |       100.0000% |    209715200 |      100.0000% |
```

```
## bench_gdeflate block-type mix
- corpus: asset-like, 3200 pages, 209.72 MB original, ratio 0.7029, generated in-harness, a MODEL of a game asset package (bench/assetlike_source.h, issue #139) and not a measurement on real game data; cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork
- compressor: the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05, compression level 6
- corpus digest: 47dad3ea983dc577
- read by: cudec's own page decoder (src/gdeflate_block.h), every page required to decode to its source bytes and every block and byte attributed to a type; this is a whole-page census and not a walk over openings
- blocks: 9600 over 3200 pages (3.000 per page), 209715200 output bytes
| type    | blocks | share of blocks | output bytes | share of bytes |
| ------- | ------ | --------------- | ------------ | -------------- |
| stored  |      0 |         0.0000% |            0 |        0.0000% |
| static  |      0 |         0.0000% |            0 |        0.0000% |
| dynamic |   9600 |       100.0000% |    209715200 |      100.0000% |
```

```
## bench_gdeflate block-type mix
- corpus: asset-like, 3200 pages, 209.72 MB original, ratio 0.6995, generated in-harness, a MODEL of a game asset package (bench/assetlike_source.h, issue #139) and not a measurement on real game data; cut into 64 KiB pages and each page compressed on its own by the pinned gdeflate fork
- compressor: the pinned NVIDIA/libdeflate gdeflate fork, commit 8ba9502fb30d2bf728592d121f0d402e40c8cb05, compression level 12
- corpus digest: cb272ab3765d8378
- read by: cudec's own page decoder (src/gdeflate_block.h), every page required to decode to its source bytes and every block and byte attributed to a type; this is a whole-page census and not a walk over openings
- blocks: 9600 over 3200 pages (3.000 per page), 209715200 output bytes
| type    | blocks | share of blocks | output bytes | share of bytes |
| ------- | ------ | --------------- | ------------ | -------------- |
| stored  |      0 |         0.0000% |            0 |        0.0000% |
| static  |      0 |         0.0000% |            0 |        0.0000% |
| dynamic |   9600 |       100.0000% |    209715200 |      100.0000% |
```

**The verdict: the mix does not justify a second path, and the corpus splits
into two populations rather than one.** Where the compressor compresses --
levels 1, 6 and 12, on both corpora -- stored and static blocks together are at
most **0.0740% of blocks and 0.0370% of the produced bytes**, and on the
asset-like corpus they are exactly zero at every one of those levels. A
specialised path acting on one part in two thousand seven hundred of the output
cannot pay for a second route through the round loop, and the accept rule #206
pre-registered asks for a recorded improvement on a non-synthetic corpus, which
that share cannot produce.

**Level 0 is 100% stored on both corpora and it is not the counter-example it
looks like.** It is the one block-type guarantee the reference's own header
gives -- level 0 emits uncompressed blocks by construction -- so that cell is
the compressor declining to compress rather than a workload with cheap blocks
in it. Two readings say it does not reopen the lever. First, the M4 CPU
denominator above records level 0 as the FASTEST family on both corpora, 0.692
and 0.678 GB/s against 0.292 to 0.373 for every compressed level, so a
specialisation there would accelerate the case that is already fastest and
leave the binding one untouched. Second, the stored block is not on the round
loop to begin with: `src/gdeflate_block.h` dispatches it to its own byte loop
that builds no table and enters no round, so the "cheap block paying the full
round-loop machinery" the lever is named for is not what the decoder does. What
the census leaves for a specialisation to act on is therefore the static blocks
alone, and they are 0.0137% to 0.0443% of blocks.

**What this closes and what it does not.** It closes #206: no kernel code ships
and the second decode path is not built. It does NOT relieve the M4 kernel of
giving stored and static blocks their own arms -- those arms are the format's
block-type dispatch and are mandatory, not a specialisation -- and it says
nothing about the round loop's cost on dynamic blocks, which is where 99.9% of
the work is and which #204, #205 and #207 measure.

**What this does not cover, stated as a bound rather than left to be assumed.**
Two corpora (Silesia and the asset-like model), one compressor (the pinned
NVIDIA/libdeflate gdeflate fork at the commit named in every block below), the
four levels the M4 corpus set names, and the 64 KiB page granularity the format
fixes. A different compressor, or a producer emitting stored blocks
deliberately, could carry a materially higher share, and neither was measured.

**A census that had collapsed back into a walk over openings would print the
same shape, so something that runs separates the two.**
`bench_gdeflate_blockmix_selfcheck` holds every page to decoding to its source
bytes, holds the per-type counts to summing to the page's own block count and
the per-type byte counts to summing to the page's output, and then holds the
level-0 cell to two things it cannot satisfy by accident: every block stored,
which is the compressor's own guarantee, and at least two blocks for every page
larger than a stored block's length field can express, which no walk over
openings can produce. Attributing every block to one type reds it on the first;
attributing bytes from the page start rather than the block start reds it on
the sum; counting one block per page -- the opening-walk reading -- reds it on
the second.

## M5: the Zstd CPU denominator (issue #227)

There is no Zstd kernel yet. This entry is the denominator a later device
number will be read against, and the harness says so in its own report rather
than leaving a reader to infer it from the absence of a GPU row.

Six cells, because the M5 batch model has two axes and a denominator taken at
one point on them does not transfer to another. The frame granularity is the
range that model names, 64 KiB to 512 KiB with every frame independent, and
both endpoints are recorded. The level set is fast, default and the high-search
family: level 19 is not optional here, because it is where the reference's own
search changes shape.

Every cell is cut from the same Silesia bytes by the pinned libzstd 1.5.7
through the corpus generator in `tests/zstd_corpus.h`, which the harness
consumes rather than duplicating. Before anything is timed, every frame is
decoded back by the reference and the concatenation of the decoded frames is
compared against the source, so a frame set that dropped, reordered or
truncated part of the corpus fails instead of reporting a plausible ratio over
different bytes.

Each report carries a digest of the corpus that was actually built, folded over
the produced frames rather than over the inputs (the inputs are already pinned
by the manifest `bench/get-corpora.sh` writes). It is order-sensitive, so the
digests below belong to the file order listed in the corpus line. Recorded
2026-08-11 inside the digest-pinned `nvidia/cuda:12.6.2-devel-ubuntu24.04`
container, on the host CPU named in the blocks. Reproduce with
`bench_zstd --warmup 3 --runs 30 bench/corpora/silesia/*`.

| granularity | level | compressed | ratio  | p50 decode | corpus digest      |
| ----------- | ----- | ---------- | ------ | ---------- | ------------------ |
| 64 KiB      | 1     | 76.70 MB   | 0.3619 | 1.281 GB/s | `f95ac5fe65483030` |
| 64 KiB      | 3     | 73.52 MB   | 0.3469 | 1.301 GB/s | `c7171e665a1888c6` |
| 64 KiB      | 19    | 65.04 MB   | 0.3069 | 1.086 GB/s | `748220fb33f7ee01` |
| 512 KiB     | 1     | 73.95 MB   | 0.3489 | 1.510 GB/s | `ed6b1bc51f66d81f` |
| 512 KiB     | 3     | 68.23 MB   | 0.3219 | 1.404 GB/s | `4983cbc86cd8255b` |
| 512 KiB     | 19    | 58.65 MB   | 0.2767 | 1.262 GB/s | `4242734974b3b0d5` |

The table is a reading aid. The six blocks below are the record, and each one
carries the methodology its own numbers were taken under.

```
## bench_zstd report
- decoder: CPU oracle, ZSTD_decompress (libzstd 1.5.7), single thread. cudec has no Zstd kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 frames, 211.94 MB original, 76.70 MB compressed (ratio 0.3619), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before timing
- granularity: 64 KiB frames, compression level 1
- corpus digest: f95ac5fe65483030 (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- frame sizes: min 60692 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is ZSTD_decompress only (destinations allocated outside it); every frame round-trip-verified and the concatenation compared against the source once before timing; percentiles are nearest-rank
- wall per run: p50 165.439 ms / p90 175.127 ms / p99 187.601 ms
- decode throughput: p50 1.281 GB/s / p90 1.210 GB/s / p99 1.130 GB/s
```

```
## bench_zstd report
- decoder: CPU oracle, ZSTD_decompress (libzstd 1.5.7), single thread. cudec has no Zstd kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 frames, 211.94 MB original, 73.52 MB compressed (ratio 0.3469), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before timing
- granularity: 64 KiB frames, compression level 3
- corpus digest: c7171e665a1888c6 (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- frame sizes: min 60692 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is ZSTD_decompress only (destinations allocated outside it); every frame round-trip-verified and the concatenation compared against the source once before timing; percentiles are nearest-rank
- wall per run: p50 162.906 ms / p90 180.539 ms / p99 184.608 ms
- decode throughput: p50 1.301 GB/s / p90 1.174 GB/s / p99 1.148 GB/s
```

```
## bench_zstd report
- decoder: CPU oracle, ZSTD_decompress (libzstd 1.5.7), single thread. cudec has no Zstd kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 frames, 211.94 MB original, 65.04 MB compressed (ratio 0.3069), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before timing
- granularity: 64 KiB frames, compression level 19
- corpus digest: 748220fb33f7ee01 (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- frame sizes: min 60692 / median 65536 / max 65536 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is ZSTD_decompress only (destinations allocated outside it); every frame round-trip-verified and the concatenation compared against the source once before timing; percentiles are nearest-rank
- wall per run: p50 195.213 ms / p90 199.905 ms / p99 204.529 ms
- decode throughput: p50 1.086 GB/s / p90 1.060 GB/s / p99 1.036 GB/s
```

```
## bench_zstd report
- decoder: CPU oracle, ZSTD_decompress (libzstd 1.5.7), single thread. cudec has no Zstd kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 405 frames, 211.94 MB original, 73.95 MB compressed (ratio 0.3489), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before timing
- granularity: 512 KiB frames, compression level 1
- corpus digest: ed6b1bc51f66d81f (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- frame sizes: min 126228 / median 524288 / max 524288 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is ZSTD_decompress only (destinations allocated outside it); every frame round-trip-verified and the concatenation compared against the source once before timing; percentiles are nearest-rank
- wall per run: p50 140.329 ms / p90 145.805 ms / p99 161.570 ms
- decode throughput: p50 1.510 GB/s / p90 1.454 GB/s / p99 1.312 GB/s
```

```
## bench_zstd report
- decoder: CPU oracle, ZSTD_decompress (libzstd 1.5.7), single thread. cudec has no Zstd kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 405 frames, 211.94 MB original, 68.23 MB compressed (ratio 0.3219), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before timing
- granularity: 512 KiB frames, compression level 3
- corpus digest: 4983cbc86cd8255b (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- frame sizes: min 126228 / median 524288 / max 524288 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is ZSTD_decompress only (destinations allocated outside it); every frame round-trip-verified and the concatenation compared against the source once before timing; percentiles are nearest-rank
- wall per run: p50 150.955 ms / p90 159.339 ms / p99 167.337 ms
- decode throughput: p50 1.404 GB/s / p90 1.330 GB/s / p99 1.267 GB/s
```

```
## bench_zstd report
- decoder: CPU oracle, ZSTD_decompress (libzstd 1.5.7), single thread. cudec has no Zstd kernel yet, so this report is the denominator and carries no cudec number
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 405 frames, 211.94 MB original, 58.65 MB compressed (ratio 0.2767), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before timing
- granularity: 512 KiB frames, compression level 19
- corpus digest: 4242734974b3b0d5 (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- frame sizes: min 126228 / median 524288 / max 524288 bytes uncompressed
- method: 3 warmup + 30 measured runs, wall clock per whole-corpus decode; the timed region is ZSTD_decompress only (destinations allocated outside it); every frame round-trip-verified and the concatenation compared against the source once before timing; percentiles are nearest-rank
- wall per run: p50 167.914 ms / p90 174.506 ms / p99 181.758 ms
- decode throughput: p50 1.262 GB/s / p90 1.215 GB/s / p99 1.166 GB/s
```

**Cutting the corpus to 64 KiB frames costs the reference decoder 8 to 14
percent, and that is the opposite of what the Snappy entry above found.** At
level 1 the same bytes decode at 1.510 GB/s in 512 KiB frames and 1.281 GB/s in
64 KiB ones; at level 3, 1.404 against 1.301; at level 19, 1.262 against 1.086.
Each gap is several times the p50-to-p99 spread of either row, where the Snappy
whole-versus-chunked pair differed by 0.18% and sat inside it. Zstd carries a
frame envelope and per-block entropy tables that Snappy has no equivalent of,
and cutting the corpus eight times finer pays for them eight times as often;
this measurement does not separate those two costs from each other, and no
claim is made here about which dominates.

What that means for later device numbers is a rule rather than an observation:
**a GPU figure must be quoted against the denominator at its own granularity.**
Reading a 64 KiB batch decode against the 512 KiB CPU row would credit the
device with a difference that is the corpus shape.

**Level 19 is the slowest cell at both granularities while compressing the
best.** 1.086 GB/s against 1.281 at 64 KiB, 1.262 against 1.510 at 512 KiB,
with the ratio moving 0.3619 to 0.3069 and 0.3489 to 0.2767. The ratio and the
decode cost move in opposite directions across the level set, so the
high-search family is where a device decoder has both the most to gain and the
most work per output byte. Why decode slows is not measured here; the per-phase
entropy-versus-execution split that would answer it is a separate entry.

**As a sanity check only**, the whole range 1.086 to 1.510 GB/s brackets the
order of magnitude zstd's own README suggests for single-thread decode. That
figure is not quoted as a number and nothing above is derived from it; these
rows were measured locally and stand on their own.

**Against the other two references on the same bytes and the same host**, at 64
KiB frames: liblz4 3.410 GB/s at ratio 0.483, snappy 1.197 GB/s at 0.478, and
libzstd 1.281 GB/s at 0.3619. Zstd is the only one of the three that is both
faster than snappy and substantially smaller. All three are single-thread wall
clock with the timed region held to the decode call alone, so this compares the
three references and says nothing about any GPU path.

**The forced-mode corpus is run as coverage and carries no number.** Nineteen
fixtures over five families (envelope, block, literals, tables, level) are
decoded by the reference and checked against the shape each was built to
demand, on every run of `bench_zstd --coverage`. They are sized to reach one
decode surface each, so a throughput figure over them would measure the fixture
list; the harness prints that in place of a number rather than leaving it to be
inferred.

## M5 perf lever: the single-stream literals shape, retired on incidence (issue #237)

**The lever, and why it is answered here instead of on a device.** The
single-stream Huffman literals form (`Size_Format=00`, RFC 8878 section
3.1.1.3.1.1) is the one spelling that carries a single backward stream; the
other three carry four. A decode shape dedicated to it would put the whole lane
team on that stream instead of running the four-stream machinery over one real
stream and three empty ones, and section 14.6 of the masterplan names it as one
of the two levers that would be tried first if the literals phase turned out to
dominate a frame's time. What it costs is a branch every literals section pays,
whatever spelling it wears. So its ceiling is set by how much literal work sits
in that form, and that is a property of what the compressor emits, not of any
kernel: it is measurable now, and it decides the lever before a kernel exists.

**The pre-registered rule this run was taken under**, written on #237 before the
measurement: a form appearing in a negligible fraction of the real corpora
retires the lever on the numbers alone, and that outcome is written up like any
other. cudec has retired a lever of exactly this shape once before -- perf pass
1 (issue #16) killed the vectorized literal copy because the path rarely
triggered and only added setup and branch overhead -- and issue #165 is where
the discipline of reading a trigger rate before writing the code was adopted.

**Two statistics, always both.** The share of literals SECTIONS wearing a
spelling and the share of literal BYTES sitting inside those sections answer
different questions, and here they cannot agree even in principle:
`Size_Format=00` caps `Regenerated_Size` at 1023 bytes while the wider
spellings reach 262143. The section share says how often the branch is paid;
the byte share bounds what it could buy. That separation is bench/literal_hist.h's
lesson from issue #165 carried across formats.

The census reads frame headers only, through the walker `tests/zstd_corpus.h`
already pins, over the same six cells and the same frames the M5 CPU
denominator above was taken on -- the six corpus digests below reproduce the six
recorded in that entry byte for byte, which is what makes this a census of that
corpus rather than of one resembling it. Recorded 2026-08-27. Reproduce with
`bench_zstd --literals bench/corpora/silesia/*`; nothing is timed and no entropy
stream is decoded, so no figure here is a denominator.

| granularity | level | huffman sections | 1-stream | of sections | huffman literal bytes | 1-stream bytes | of bytes |
| ----------- | ----- | ---------------- | -------- | ----------- | --------------------- | -------------- | -------- |
| 64 KiB      | 1     | 3149             | 0        | 0.0000%     | 63586548              | 0              | 0.0000%  |
| 64 KiB      | 3     | 3149             | 0        | 0.0000%     | 48451864              | 0              | 0.0000%  |
| 64 KiB      | 19    | 3096             | 0        | 0.0000%     | 26666688              | 0              | 0.0000%  |
| 512 KiB     | 1     | 1889             | 0        | 0.0000%     | 61165017              | 0              | 0.0000%  |
| 512 KiB     | 3     | 2173             | 6        | 0.2761%     | 34179495              | 1010           | 0.0030%  |
| 512 KiB     | 19    | 3621             | 174      | 4.8053%     | 17000854              | 28296          | 0.1664%  |

The table is a reading aid. The six blocks below are the record.

```
## bench_zstd literals-shape report
- what this is: an incidence census over the literals sections the pinned libzstd 1.5.7 emits, read from the frame headers by ParseZstdFrameShape in tests/zstd_corpus.h. NOT A THROUGHPUT MEASUREMENT: nothing is timed and no entropy stream is decoded, so no number here is a denominator
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 frames, 211.94 MB original, 76.70 MB compressed (ratio 0.3619), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before the walk
- granularity: 64 KiB frames, compression level 1
- corpus digest: f95ac5fe65483030 (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- blocks walked: 3234 over 3234 frames (raw 21 / rle 0 / compressed 3213)
- literals sections: 3213 (raw 64 / rle 0 / compressed 3149 / treeless 0)
- huffman sections by stream count: 3149 total, 1-stream (Size_Format=00) 0 (0.0000%), 4-stream 3149 (100.0000%)
- huffman literal bytes regenerated: 63586548 total, 1-stream 0 (0.0000%), 4-stream 63586548 (100.0000%)
- largest 1-stream section seen: 0 bytes regenerated (the spelling's own ceiling is 1023)
```

```
## bench_zstd literals-shape report
- what this is: an incidence census over the literals sections the pinned libzstd 1.5.7 emits, read from the frame headers by ParseZstdFrameShape in tests/zstd_corpus.h. NOT A THROUGHPUT MEASUREMENT: nothing is timed and no entropy stream is decoded, so no number here is a denominator
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 frames, 211.94 MB original, 73.52 MB compressed (ratio 0.3469), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before the walk
- granularity: 64 KiB frames, compression level 3
- corpus digest: c7171e665a1888c6 (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- blocks walked: 3234 over 3234 frames (raw 15 / rle 0 / compressed 3219)
- literals sections: 3219 (raw 70 / rle 0 / compressed 3149 / treeless 0)
- huffman sections by stream count: 3149 total, 1-stream (Size_Format=00) 0 (0.0000%), 4-stream 3149 (100.0000%)
- huffman literal bytes regenerated: 48451864 total, 1-stream 0 (0.0000%), 4-stream 48451864 (100.0000%)
- largest 1-stream section seen: 0 bytes regenerated (the spelling's own ceiling is 1023)
```

```
## bench_zstd literals-shape report
- what this is: an incidence census over the literals sections the pinned libzstd 1.5.7 emits, read from the frame headers by ParseZstdFrameShape in tests/zstd_corpus.h. NOT A THROUGHPUT MEASUREMENT: nothing is timed and no entropy stream is decoded, so no number here is a denominator
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 3234 frames, 211.94 MB original, 65.04 MB compressed (ratio 0.3069), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before the walk
- granularity: 64 KiB frames, compression level 19
- corpus digest: 748220fb33f7ee01 (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- blocks walked: 3234 over 3234 frames (raw 1 / rle 0 / compressed 3233)
- literals sections: 3233 (raw 137 / rle 0 / compressed 3096 / treeless 0)
- huffman sections by stream count: 3096 total, 1-stream (Size_Format=00) 0 (0.0000%), 4-stream 3096 (100.0000%)
- huffman literal bytes regenerated: 26666688 total, 1-stream 0 (0.0000%), 4-stream 26666688 (100.0000%)
- largest 1-stream section seen: 0 bytes regenerated (the spelling's own ceiling is 1023)
```

```
## bench_zstd literals-shape report
- what this is: an incidence census over the literals sections the pinned libzstd 1.5.7 emits, read from the frame headers by ParseZstdFrameShape in tests/zstd_corpus.h. NOT A THROUGHPUT MEASUREMENT: nothing is timed and no entropy stream is decoded, so no number here is a denominator
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 405 frames, 211.94 MB original, 73.95 MB compressed (ratio 0.3489), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before the walk
- granularity: 512 KiB frames, compression level 1
- corpus digest: ed6b1bc51f66d81f (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- blocks walked: 1940 over 405 frames (raw 6 / rle 0 / compressed 1934)
- literals sections: 1934 (raw 45 / rle 0 / compressed 1681 / treeless 208)
- huffman sections by stream count: 1889 total, 1-stream (Size_Format=00) 0 (0.0000%), 4-stream 1889 (100.0000%)
- huffman literal bytes regenerated: 61165017 total, 1-stream 0 (0.0000%), 4-stream 61165017 (100.0000%)
- largest 1-stream section seen: 0 bytes regenerated (the spelling's own ceiling is 1023)
```

```
## bench_zstd literals-shape report
- what this is: an incidence census over the literals sections the pinned libzstd 1.5.7 emits, read from the frame headers by ParseZstdFrameShape in tests/zstd_corpus.h. NOT A THROUGHPUT MEASUREMENT: nothing is timed and no entropy stream is decoded, so no number here is a denominator
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 405 frames, 211.94 MB original, 68.23 MB compressed (ratio 0.3219), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before the walk
- granularity: 512 KiB frames, compression level 3
- corpus digest: 4983cbc86cd8255b (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- blocks walked: 2506 over 405 frames (raw 88 / rle 0 / compressed 2418)
- literals sections: 2418 (raw 245 / rle 0 / compressed 1956 / treeless 217)
- huffman sections by stream count: 2173 total, 1-stream (Size_Format=00) 6 (0.2761%), 4-stream 2167 (99.7239%)
- huffman literal bytes regenerated: 34179495 total, 1-stream 1010 (0.0030%), 4-stream 34178485 (99.9970%)
- largest 1-stream section seen: 227 bytes regenerated (the spelling's own ceiling is 1023)
```

```
## bench_zstd literals-shape report
- what this is: an incidence census over the literals sections the pinned libzstd 1.5.7 emits, read from the frame headers by ParseZstdFrameShape in tests/zstd_corpus.h. NOT A THROUGHPUT MEASUREMENT: nothing is timed and no entropy stream is decoded, so no number here is a denominator
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- corpus: dickens+mozilla+mr+nci+ooffice+osdb+reymont+samba+sao+webster+x-ray+xml, 405 frames, 211.94 MB original, 58.65 MB compressed (ratio 0.2767), cut into independent frames and compressed by the pinned libzstd through the corpus generator in tests/zstd_corpus.h; every frame decoded back by the reference and the concatenation compared against the source before the walk
- granularity: 512 KiB frames, compression level 19
- corpus digest: 4242734974b3b0d5 (XXH64 over per-frame length and XXH64, little-endian, in corpus order)
- blocks walked: 4378 over 405 frames (raw 1 / rle 0 / compressed 4377)
- literals sections: 4377 (raw 735 / rle 21 / compressed 3213 / treeless 408)
- huffman sections by stream count: 3621 total, 1-stream (Size_Format=00) 174 (4.8053%), 4-stream 3447 (95.1947%)
- huffman literal bytes regenerated: 17000854 total, 1-stream 28296 (0.1664%), 4-stream 16972558 (99.8336%)
- largest 1-stream section seen: 255 bytes regenerated (the spelling's own ceiling is 1023)
```

**The lever is retired: no kernel code ships.** Across the whole recorded grid
the single-stream form carries at most **0.1664% of the Huffman literal bytes**,
and four of the six cells carry none at all. The branch that would select the
dedicated shape is taken by every literals section, so the trade is a cost paid
on up to 100% of sections against a ceiling of one part in six hundred of the
literal work -- and that ceiling is generous twice over, because it assumes the
dedicated shape decodes those bytes in zero time and because literal decode is
only one phase of a frame's cost. That is the pre-registered retirement
condition met on the numbers.

**The section share and the byte share diverge by a factor of thirty, and only
one of them is the argument.** At 512 KiB and level 19 the form is 4.8053% of
sections but 0.1664% of bytes, because the largest single-stream section in the
entire grid regenerates 255 bytes against a 262143-byte ceiling on the wide
spellings. A reader who took the section share for the size of the prize would
be reading the wrong statistic; issue #165 is where that mistake was made
visible on LZ4 and Snappy, and the census prints both figures on every run so it
cannot be made silently here.

**What drives the form is block splitting, not frame size,** and that is the
opposite of what the lever's framing assumed. Incidence is zero at 64 KiB
granularity across the whole level set and non-zero only at 512 KiB, where the
compressor emits many blocks per frame instead of one: 4378 blocks over 405
frames at level 19 against 3234 blocks over 3234 frames at 64 KiB. The form is
the tail of a block split small, so shrinking the frame -- which produces fewer,
larger blocks per frame here -- does not raise it. No cause is claimed beyond
what those counts show.

**What this does not cover, stated as a bound rather than left to be assumed.**
One corpus (Silesia), one compressor (the pinned libzstd 1.5.7), the two frame
granularities and three levels the M5 batch model names, and nothing below 64
KiB. A different compressor, a dictionary, or a producer emitting deliberately
tiny blocks could carry a materially higher share, and none of those was
measured. The forced-mode corpus does emit the form on demand, so the fallback
#237 named -- reach for the #185 fixtures if the natural corpora never exercise
it -- was not needed: the natural corpora exercise it, and the answer is that
they barely do.

**The census cannot report a zero it cannot see, and that is checked by
something that runs.** A near-zero incidence and a walk that stopped
recognising the single-stream spelling produce the identical table, so
`bench_zstd_literals_selfcheck` runs the census over the forced-mode corpus --
which carries a fixture of each spelling -- and fails unless both were counted,
in sections and in bytes. Removing the one-stream arm of the census turns that
ctest entry red with the reason named; the Silesia grid reporting 0.0000% leaves
it green. That is the separation the zeroes above need in order to be read as a
result.

## M5 perf lever: how big a Zstd literals section is, measured before the kernel (issue #236)

**Why this is here with no device in the room.** Issue #236's lever is the
4-stream Huffman literals path: the lane-to-stream mapping, and whether decoded
literals go straight to the destination or are staged in shared memory and
flushed coalesced. Which shape wins is a device A/B and it stays open. What does
not need a device is the size of the thing either shape operates on, and #236
names that measurement in its own body as the durable result whichever way the
A/B later goes, because it is a property of what the compressor emits.

**The question it settles is whether an M1 negative transfers.** Perf pass 1
(issue #16) above rejected the vectorized 16-byte literal copy at -6%, and the
reason recorded there is that Silesia's LZ4 literal runs are mostly shorter than
16 bytes, so the wide path rarely triggered. That is a statement about a
per-sequence literal RUN. A Zstd literals section is a whole block's literals
decoded in bulk. Different population, so the earlier number decides nothing
here, and the only honest way to say so is to measure the population it would
have to carry over to.

**One walk, not a second one.** The census is the one `bench_zstd --literals`
already runs for the entry above, extended with a size distribution rather than
duplicated: same headers, same walker in `tests/zstd_corpus.h`, same six cells,
and the six corpus digests below reproduce the ones recorded in that entry and
in the M5 CPU denominator byte for byte. Recorded 2026-08-27 with
`bench_zstd --literals bench/corpora/silesia/*`. Nothing is timed and no entropy
stream is decoded, so no figure here is a denominator.

**Two statistics, always both**, which is issue #165's lesson and is sharper
here than in the entry above: the section shares and the byte shares below
disagree by a wide margin in every cell, and only the byte share bounds what a
different write shape could buy.

Share of Huffman literal BYTES by section size, in regenerated bytes:

| granularity | level | digest           | 0-127   | 128-1023 | 1K-4K   | 4K-16K  | 16K-64K | 64K+    |
| ----------- | ----- | ---------------- | ------- | -------- | ------- | ------- | ------- | ------- |
| 64 KiB      | 1     | f95ac5fe65483030 | 0.0000% | 0.0041%  | 1.4681% | 15.047% | 73.380% | 10.101% |
| 64 KiB      | 3     | c7171e665a1888c6 | 0.0000% | 0.0141%  | 3.1227% | 35.510% | 61.083% | 0.2705% |
| 64 KiB      | 19    | 748220fb33f7ee01 | 0.0000% | 0.0713%  | 8.0726% | 60.237% | 31.620% | 0.0000% |
| 512 KiB     | 1     | ed6b1bc51f66d81f | 0.0000% | 0.0053%  | 0.4283% | 7.8709% | 63.352% | 28.344% |
| 512 KiB     | 3     | 4983cbc86cd8255b | 0.0007% | 0.1261%  | 3.4317% | 24.297% | 51.055% | 21.089% |
| 512 KiB     | 19    | 4242734974b3b0d5 | 0.0330% | 2.4833%  | 20.256% | 52.747% | 24.481% | 0.0000% |

Share of SECTIONS in the same buckets, with the section count and the mean:

| granularity | level | sections | 0-127   | 128-1023 | 1K-4K   | 4K-16K  | 16K-64K | 64K+    | mean bytes |
| ----------- | ----- | -------- | ------- | -------- | ------- | ------- | ------- | ------- | ---------- |
| 64 KiB      | 1     | 3149     | 0.0000% | 0.0953%  | 9.6221% | 33.122% | 54.049% | 3.1121% | 20192.6    |
| 64 KiB      | 3     | 3149     | 0.0000% | 0.2540%  | 15.783% | 53.033% | 30.867% | 0.0635% | 15386.4    |
| 64 KiB      | 19    | 3096     | 0.0000% | 0.8075%  | 21.705% | 64.470% | 13.017% | 0.0000% | 8613.3     |
| 512 KiB     | 1     | 1889     | 0.0000% | 0.2118%  | 4.8703% | 28.110% | 56.856% | 9.9524% | 32379.6    |
| 512 KiB     | 3     | 2173     | 0.0920% | 2.9452%  | 19.972% | 45.283% | 27.888% | 3.8196% | 15729.2    |
| 512 KiB     | 19    | 3621     | 1.6294% | 20.713%  | 39.740% | 32.698% | 5.2196% | 0.0000% | 4695.1     |

The two tables are a reading aid. The record is the `bench_zstd --literals` run
itself, whose six report blocks the entry above already carries; only the
distribution lines are new, both tables reproduce them in full, and the corpus
digests are what join a row here to the block it came out of. Those blocks are
not pasted a second time, because a second copy would go stale against the first
the next time either is re-taken.

**The M1 negative does not transfer, and the margin is three orders of
magnitude.** The mean Huffman literals section runs from 4695 to 32380
regenerated bytes across the grid. Sections below the 128-byte coalescing floor
-- 32 lanes times 4 bytes, the smallest store a warp can make whole -- hold
0.0000% of the literal bytes in four of the six cells and at most 0.0330% in the
other two, against LZ4 literal runs that were mostly under 16 bytes. So the
argument that killed the wide copy in perf pass 1, that the path would rarely
trigger, cannot be made against this one on this corpus. That is a negative
removed, not a lever accepted.

**What this does NOT establish, said here rather than left to be assumed.** A
regenerated section size is what the block header declares for the whole
section. It is an UPPER BOUND on what any one lane or any one flush handles and
not a measurement of either: the four-stream split divides a section four ways
before any lane assignment divides it again, and the lane assignment is the free
parameter the lever is about. So a byte share sitting above the coalescing floor
says the prize is not excluded by the size of the sections; it does not say a
staged flush would be coalesced, and nothing here measures a store pattern. That
is the device A/B, and it is what keeps #236 open.

**Where the mass sits also constrains the shape, and that is arithmetic rather
than a design decision.** Between 71% and 97% of the literal bytes in every cell
sit in sections of 4 KiB to 64 KiB, and the 64 KiB-and-above bucket carries
10.101% at 64 KiB level 1 and 28.344% at 512 KiB level 1. A staging buffer
holding a whole section would therefore have to be sized in tens of kilobytes,
which is the shared-memory budget masterplan section 14.2 sizes the resident
block count against. Any staging variant is a chunked flush rather than a
whole-section buffer, and its shared-memory cost and the blocks per SM it leaves
are reported beside its throughput number when the A/B is run, the way perf pass
2 (issue #21) did.

**Level and granularity move the distribution in opposite directions.** Raising
the level shrinks sections monotonically at both granularities (20192.6 to
15386.4 to 8613.3 at 64 KiB; 32379.6 to 15729.2 to 4695.1 at 512 KiB), which is
the compressor finding more matches and leaving fewer literals. Frame size does
not act in one direction: at level 1 the larger frame gives the larger sections
(32379.6 against 20192.6) and at level 19 the smaller ones (4695.1 against
8613.3). The entry above measured that same corner from the other side, counting
4378 blocks over 405 frames at 512 KiB level 19 against 3234 blocks over 3234
frames at 64 KiB, so one block-splitting behaviour is visible in both readings.
No cause is claimed beyond what the counts show.

**The bounds of the corpus, unchanged from the entry above.** One corpus
(Silesia), one compressor (the pinned libzstd 1.5.7), the two frame
granularities and three levels the M5 batch model names, and nothing below 64
KiB. A producer emitting deliberately tiny blocks, a dictionary, or a different
compressor could sit far to the left of every row here, and none of those was
measured.

**A distribution that had quietly stopped meaning what its labels say is
refused rather than hoped against.** Three things run. The bucket edges live in
one array that both the bucketing and the printed labels derive from, so a row
cannot be named from one set of numbers and filled from another. Every census
run -- the Silesia grid included, not only `--selfcheck` -- requires the
distribution to add up to the Huffman totals it was folded from, and requires
every occupied bucket to hold only sizes inside its own edges, checked against
the smallest and the largest section actually placed there rather than against
an average that would hide one misroute among hundreds. And `--selfcheck` walks
the edges directly, asserting that each inclusive upper edge lands in its own
bucket and each edge plus one in the next, because a fixture of an
exactly-1023-byte literals section is not something the reference can be asked
for: `Regenerated_Size` is whatever the compressor left as literals in a block,
and no public entry point sets it.

Four mutants, all red, each for a different failure:

| mutant                                                            | verdict                                                                                                                                                                                  |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the bucket scan compares `<` where `<=` belongs                   | `bench_zstd_literals_selfcheck` red; the edge walk names 127, 1023, 4095, 16383 and 65535 one after another, and the corpus-side checks stay silent, which is why the edge walk is there |
| a bucket's byte total is never folded                             | red on the run itself: a bucket counted sections carrying bytes and folded none of them                                                                                                  |
| a section is routed by a quarter of its size                      | red on the run itself: a bucket holds a section outside its own edges                                                                                                                    |
| the first edge moves from 127 to 255, the report still saying 128 | the build is red on a static assertion, so the trigger line cannot come to report a threshold nobody chose                                                                               |

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
