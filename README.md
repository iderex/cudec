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
  implementations (liblz4, zlib, libzstd).
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

**Design phase.** The architecture is being recorded in
[docs/MASTERPLAN.md](docs/MASTERPLAN.md); progress is tracked in the issues
and milestones:

| Milestone        | Deliverable                                         |
| ---------------- | --------------------------------------------------- |
| M0 — Foundation  | Toolchain, CMake+CUDA skeleton, CI, test harness    |
| M1 — LZ4 block   | Warp-cooperative LZ4 block decode, fuzz-diffed      |
| M2 — LZ4 batch   | Frame format, batch API, streaming path, benchmarks |
| M3 — Snappy      | Snappy decode on the same kernel family             |
| M4 — GDeflate    | The first open GPU GDeflate decoder                 |
| M5 — Zstd        | Zstd decode (FSE/Huffman sequences)                 |
| M6 — Portability | HIP port                                            |

## Principles

- **Fail-closed.** A malformed or hostile bitstream produces a defined error,
  never an out-of-bounds access and never a guess. Every reject path has a
  negative test.
- **Deterministic.** Same input, same output — bit-exact on every code path.
- **Honest numbers.** Every performance claim ships with GPU model, driver,
  CUDA version, corpus, and chunk-size distribution. Benchmarks live in the
  repo.
- **Minimal.** The least code that does the job; structural rules are locked
  in by conformance tests.

## Building

Requires CMake ≥ 3.24 and a C compiler; the CUDA kernels and their toolchain
requirements arrive with M1.

```sh
cmake -B build && cmake --build build
```

## Contributing

Issue-driven: every change starts as an issue and lands as a gated PR — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE)
