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
therefore settled for single-pass.** The perf pass (#16) targets the copy
first (vectorized multi-byte lane copies + incremental-mod, to pull
end-to-end toward the ~35 GB/s parse ceiling) and the parse itself second
(register-window staging, to raise the ceiling). The masterplan
falsification trigger — reopen two-phase only if the shipped kernel measures
below ~15x CPU after perf pass 1, or profiling attributes the majority of
stalls to copy starvation — is evaluated at #16.

Occupancy readout (issue #14): 46 registers/thread, so ≥ 32 warps/SM is
achievable on sm_86.

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
