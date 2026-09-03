# cudec

Open-source GPU decompression for the standard formats.

**Goal: batch-decode LZ4, Snappy, GDeflate and Zstd on an NVIDIA GPU at
memory-bandwidth speed - auditable, fail-closed, and fuzz-tested.**

## Why

GPU decompression matters wherever decode throughput is the bottleneck:
asset streaming, analytics scans, ML data loading, checkpoint restore. The
only production-grade library in this space, NVIDIA's nvCOMP, has been
proprietary since v2.3 - there is no maintained open-source CUDA library that
decodes the standard formats on the GPU. Meta's dietgpu is open but uses its
own rANS format; the open GPU GDeflate and Zstd decoders that do exist are
shaders owned by a graphics runtime (DirectStorage, vkd3d-proton) rather than
libraries you can call; AMD's hipCOMP-core is an early-access HIP preview
downstream of a frozen nvCOMP fork; the academic prototypes are unmaintained.
[docs/MASTERPLAN.md](docs/MASTERPLAN.md) section 1 has the field table and
the sources.

cudec fills that gap. Not on price - nvCOMP is free to use - but on the
properties a closed binary cannot offer:

- **Auditability.** Decompressors are classic attack surface. Every bounds
  check in cudec is readable, tested, and fuzz-diffed against the reference
  implementations - liblz4 and snappy today, with zlib and libzstd joining as
  the DEFLATE and Zstd formats land.
- **Portability.** CUDA first. The HIP port compiles and has not run: the
  same sources build as HIP for four AMD architectures in a ROCm container,
  and no AMD GPU has executed the result. What that milestone claims is one
  kernel family, single-source across both vendors behind the same C ABI,
  auditable and fail-closed on either - not a first on AMD, for the reason
  the field table linked above gives.
- **Hackability.** Format quirks, tuning trade-offs, and kernel design are
  documented in the tree, not behind a support contract.

## Scope

Decode-only, batch-oriented. Compression stays on the CPU where it belongs;
the GPU wins when thousands of independent chunks decode in parallel. A
single small file on a cold PCIe bus is not the use case - the CPU wins that
one, and this README will never claim otherwise.

## Status

**M0 through M3 are complete; M4 is next.** cudec decodes real LZ4 and real
Snappy on an NVIDIA GPU today - LZ4 batch block decode, the `.lz4` frame
format (block-independent subset), a pinned-host LZ4 streaming path, and
Snappy batch decode of raw streams - all fail-closed and fuzz-diffed against
liblz4 and snappy. Snappy is batch-only by decision, not by omission: there is
no Snappy streaming entry and no milestone carries one. The design record is
[docs/MASTERPLAN.md](docs/MASTERPLAN.md); the measured LZ4 and Snappy
baselines, each carrying its full methodology, are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). GDeflate and Zstd are planned, not
yet implemented. The HIP port compiles and has never run on an AMD GPU (see
Building), so no AMD number and no AMD correctness claim exists. Progress is
tracked in the issues and milestones:

| Milestone        | Deliverable                                         | Status   |
| ---------------- | --------------------------------------------------- | -------- |
| M0 - Foundation  | Toolchain, CMake+CUDA skeleton, CI, test harness    | done     |
| M1 - LZ4 block   | Warp-cooperative LZ4 block decode, fuzz-diffed      | done     |
| M2 - LZ4 batch   | Frame format, batch API, streaming path, benchmarks | done     |
| M3 - Snappy      | Snappy decode on the same kernel family             | done     |
| M4 - GDeflate    | GDeflate decode as an auditable CUDA library        | planned  |
| M5 - Zstd        | Zstd decode (FSE/Huffman sequences)                 | planned  |
| M6 - Portability | HIP port, same sources, compiled and never run      | compiles |

## Principles

- **Fail-closed.** A malformed or hostile bitstream produces a defined error,
  never an out-of-bounds access and never a guess - and never a hang: every
  loop driven by a value read from the stream is capped, and termination is a
  tested invariant. Every reject path has a negative test.
- **Deterministic.** Same input, same output - bit-exact on every code path,
  in every supported launch configuration, on every supported GPU. The scope,
  the qualifiers, and the tested axes:
  [docs/DETERMINISM.md](docs/DETERMINISM.md).
- **Honest numbers.** Recorded baselines: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
  (`bench/bench_lz4`; corpora via `bench/get-corpora.sh`, hash-pinned).
  Every performance claim ships with GPU model, driver,
  CUDA version, corpus, and chunk-size distribution.
- **Minimal.** The least code that does the job; structural rules are locked
  in by conformance tests.

## Building

Two builds. The host-only build needs just a C compiler and compiles the ABI
and version surface - not the decoder - so CI has a real build gate without a
CUDA toolchain (CMake ≥ 3.24):

```sh
cmake -B build && cmake --build build
```

The CUDA build is the decoder (CUDA 12.x toolchain; the maintained path is the
pinned dev container - a GPU is required only for the gpu-labeled tests, not
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

The HIP build is the same decoder compiled for AMD: the same sources, driven
as CMake's HIP language (ROCm 7.0 or later; the maintained path is the
production `rocm/dev-ubuntu-24.04` container, 7.2.4 at the time of writing).
It compiles for `gfx90a`, `gfx942`, `gfx1100` and `gfx1201` unless
`CMAKE_HIP_ARCHITECTURES` says otherwise, and never auto-detects a device. It
has been compiled, not run: no AMD GPU has executed this decoder, so the
command below runs the host-side subset only, and the gpu-labeled tests exist
in that build as binaries nothing has yet executed. On-device AMD validation
is the community's to produce, and the runbook and result intake for it are
not written yet (#245, #246).

```sh
docker run --rm -v "$PWD:/w" -w /w \
  rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054 \
  sh -c "apt-get update -q && apt-get install -yq cmake >/dev/null && \
         cmake -B build-hip -DCUDEC_ENABLE_HIP=ON && \
         cmake --build build-hip -j && \
         ctest --test-dir build-hip -LE gpu --no-tests=error --output-on-failure"
```

Any of the three builds installs into a prefix that an outside project
consumes through `find_package` (`build-cuda` or `build-hip` in place of
`build` for the device flavours):

```sh
cmake --install build --prefix /some/prefix
```

```cmake
find_package(cudec 0.1.0 REQUIRED)
target_link_libraries(your_target PRIVATE cudec::cudec)
```

Point the consumer's configure at the prefix with
`-DCMAKE_PREFIX_PATH=/some/prefix`. The consumer writes nothing about the
backend: a prefix from a CUDA build carries its own
`find_dependency(CUDAToolkit)`, a prefix from a HIP build carries
`find_dependency(hip)` in the same shape, and a host-only prefix carries
none, so it stays consumable on a machine that has no device toolkit at all.
The host-only and CUDA flavours are installed and consumed in CI; no consumer
has been built against a HIP prefix yet.

Two limits on the CUDA flavour, both consumer-side. It needs the CUDA toolkit
present on the consuming machine, not merely the runtime library: the
dependency resolves through CMake's `FindCUDAToolkit`, which looks for `nvcc`.
And the consuming project must enable C++ (`project(... LANGUAGES C CXX)`),
because cudec's host orchestration is C++ and CMake picks the link driver from
the languages the consuming project enabled, not from the archive. A C-only
consumer links `cudec_version()` and fails on the first decode call with
undefined C++ runtime symbols. The HIP flavour carries the same two limits
with a ROCm installation in place of the toolkit, and nothing has exercised
them. The host-only prefix has neither limit; it is pure C and needs no
toolkit.

cudec builds as a static library only, and a top-level configure with
`BUILD_SHARED_LIBS=ON` is refused rather than producing an untested shared
build. The version compatibility rule is `SameMinorVersion`: while the version
is 0.x, this project treats a minor bump as the breaking one, so a consumer
asking for 0.0 is not handed 0.1.

## Usage

[examples/decode_frame.c](examples/decode_frame.c) decodes a `.lz4` frame using
the public header and libc, and nothing else. Every build compiles it, and the
CUDA and HIP builds link it into `build-cuda/examples/example_decode_frame`
and `build-hip/examples/example_decode_frame`.

[examples/decode_batch.cu](examples/decode_batch.cu) decodes four independent
LZ4 blocks in one batch call, one of them deliberately corrupt, so the
per-chunk contract is visible: the bad chunk reports a defined error with no
bytes written while its neighbours decode. It needs the CUDA build, which
links it into `build-cuda/examples/example_decode_batch`.

## Contributing

Issue-driven: every change starts as an issue and lands as a gated PR - see
[CONTRIBUTING.md](CONTRIBUTING.md).

## 🤝 AI-assisted, human-owned

Development here is AI-assisted. Claude (Anthropic) helps with individual
process steps - generating and analysing code, running the adversarial
security reviews, and translating documentation and comments into English. It
never hands over finished, unreviewed work: each step is only a proposal. A
human reviews, understands, edits where needed, and signs off on
every one - the AI proposes, a person decides, and a human stays responsible
for every line that ships, at all times. The review discipline is modelled,
as far as is practical for a volunteer project, on the change-control
expected of TÜV/BSI-certified software in a critical sector such as
healthcare - with no claim to actual certification. In short: nothing lands
because a tool suggested it; it lands because a person verified it.

## License

[Apache-2.0](LICENSE)

See NOTICE.md for the intended-use notice.
