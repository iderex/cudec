# cudec

Open-source GPU decompression for the standard formats.

**Goal: batch-decode LZ4, Snappy, GDeflate and Zstd on an NVIDIA GPU at
memory-bandwidth speed — auditable, fail-closed, and fuzz-tested.**

## Why

GPU decompression matters wherever decode throughput is the bottleneck:
asset streaming, analytics scans, ML data loading, checkpoint restore. The
only production-grade library in this space, NVIDIA's nvCOMP, has been
proprietary since v2.3 — there is no maintained open-source library that
decodes the standard formats on the GPU. Meta's dietgpu is open but uses its
own rANS format; the GDeflate reference implementation is CPU-only; the
academic prototypes are unmaintained.

cudec fills that gap. Not on price — nvCOMP is free to use — but on the
properties a closed binary cannot offer:

- **Auditability.** Decompressors are classic attack surface. Every bounds
  check in cudec is readable, tested, and fuzz-diffed against the reference
  implementation — liblz4 today, with zlib and libzstd joining as the DEFLATE
  and Zstd formats land.
- **Portability.** CUDA first; a HIP port is a planned milestone. A
  vendor-locked binary can never follow.
- **Hackability.** Format quirks, tuning trade-offs, and kernel design are
  documented in the tree, not behind a support contract.

## Scope

Decode-only, batch-oriented. Compression stays on the CPU where it belongs;
the GPU wins when thousands of independent chunks decode in parallel. A
single small file on a cold PCIe bus is not the use case — the CPU wins that
one, and this README will never claim otherwise.

## Status

**M0 and M1 are complete; M2 is in progress.** cudec decodes real LZ4 on an
NVIDIA GPU today — batch block decode, the `.lz4` frame format
(block-independent subset), and a pinned-host streaming path — all fail-closed
and fuzz-diffed against liblz4. The design record is
[docs/MASTERPLAN.md](docs/MASTERPLAN.md); the measured LZ4 block and streaming
baselines, each carrying its full methodology, are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). Snappy, GDeflate, Zstd, and the HIP
port are planned, not yet implemented. Progress is tracked in the issues and
milestones:

| Milestone        | Deliverable                                         | Status      |
| ---------------- | --------------------------------------------------- | ----------- |
| M0 — Foundation  | Toolchain, CMake+CUDA skeleton, CI, test harness    | done        |
| M1 — LZ4 block   | Warp-cooperative LZ4 block decode, fuzz-diffed      | done        |
| M2 — LZ4 batch   | Frame format, batch API, streaming path, benchmarks | in progress |
| M3 — Snappy      | Snappy decode on the same kernel family             | planned     |
| M4 — GDeflate    | The first open GPU GDeflate decoder                 | planned     |
| M5 — Zstd        | Zstd decode (FSE/Huffman sequences)                 | planned     |
| M6 — Portability | HIP port                                            | planned     |

## Principles

- **Fail-closed.** A malformed or hostile bitstream produces a defined error,
  never an out-of-bounds access and never a guess. Every reject path has a
  negative test.
- **Deterministic.** Same input, same output — bit-exact on every code path.
- **Honest numbers.** Recorded baselines: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
  (`bench/bench_lz4`; corpora via `bench/get-corpora.sh`, hash-pinned).
  Every performance claim ships with GPU model, driver,
  CUDA version, corpus, and chunk-size distribution.
- **Minimal.** The least code that does the job; structural rules are locked
  in by conformance tests.

## Building

Two builds. The host-only build needs just a C compiler and compiles the ABI
and version surface — not the decoder — so CI has a real build gate without a
CUDA toolchain (CMake ≥ 3.24):

```sh
cmake -B build && cmake --build build
```

The CUDA build is the decoder (CUDA 12.x toolchain; the maintained path is the
pinned dev container — a GPU is required only for the gpu-labeled tests, not
the build: without a GPU, drop `--gpus all` and add `-LE gpu` to the ctest line
to build everything and run the host-side subset):

```sh
docker run --rm --gpus all -v "$PWD:/w" -w /w \
  nvidia/cuda:12.6.2-devel-ubuntu24.04@sha256:738fba0fbdb225b7a2931c58a5c8f03a84d3cd2f6a84975826a157339ef750b8 \
  sh -c "apt-get update -q && apt-get install -yq cmake >/dev/null && \
         cmake -B build-cuda -DCUDEC_ENABLE_CUDA=ON && \
         cmake --build build-cuda -j && \
         ctest --test-dir build-cuda --no-tests=error --output-on-failure"
```

## Contributing

Issue-driven: every change starts as an issue and lands as a gated PR — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## AI assistance

Claude (Anthropic) assisted in building cudec — with development, review, and
English phrasing — always on individual process steps, never producing
finished or unreviewed work. AI-assisted does not mean AI-run: a human
maintainer directs and oversees every step, reviews the result through the
project's adversarial review gate, has the final say on everything that
lands, and remains responsible for it at all times.

## License

[Apache-2.0](LICENSE)
