# Benchmarks

The baseline record. Every entry carries the full methodology block as
emitted by the harness (`bench/bench_lz4`) — a number without its
methodology cannot be produced, by construction. Regressions against the
recorded baselines block merges unless explicitly justified
([MASTERPLAN](MASTERPLAN.md) section 5). Corpora are fetched hash-pinned
via `bench/get-corpora.sh` and never committed.

## Baseline: CPU oracle (M0, pre-kernel)

The reference the M1 GPU decoder is measured against: the single-threaded
CPU oracle on the development machine, recorded 2026-07-17 inside the
digest-pinned dev container (`nvidia/cuda:12.6.2-devel-ubuntu24.04`).
For context, the sm_86 output-bandwidth ceiling on this machine is
~760 GB/s.

```
## bench_lz4 report
- decoder: CPU oracle, LZ4_decompress_safe (liblz4 1.10.0), single thread
- host CPU: AMD Ryzen 9 5950X 16-Core Processor
- CUDA device: NVIDIA GeForce RTX 3080 (sm_86), driver 12.6, runtime 12.6
- cudec: 1 (stub era - the library decodes nothing yet)
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
- cudec: 1 (stub era - the library decodes nothing yet)
- corpus: builtin, 12 chunks, 0.21 MB original, 0.10 MB compressed (ratio 0.511), compressed in-harness via LZ4_compress_default
- chunk sizes: min 1 / median 257 / max 65536 bytes
- method: 3 warmup + 30 measured runs, wall clock per whole-batch decode; the timed region is LZ4_decompress_safe only (no clears, no allocation); output byte-verified once before timing; percentiles are nearest-rank
- wall per run: p50 0.046 ms / p90 0.049 ms / p99 0.064 ms
- decode throughput: p50 4.480 GB/s / p90 4.221 GB/s / p99 3.206 GB/s
```
