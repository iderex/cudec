# Changelog

All notable changes to this project are recorded here, in the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format, and the
version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

What "notable" means here: a change a consumer of the C ABI, the build, or the
recorded numbers can observe. Internal refactors, test-net work and process
changes stay in the git history and in their issues, where they are already
readable.

0.1.0 is the first release, cut on 2026-09-03: releases were authorised that
day (#82), and the tag `v0.1.0` is set on the merge commit of the release pull
request, which carries the gate readings this entry summarises under Known
gaps. Everything in the entry is in the tree at that commit.

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-09-03

### Added

- **LZ4 block batch decode on the GPU.** `cudec_lz4_decompress_batch` decodes
  N independent LZ4 block-format chunks in one launch on a caller-provided
  stream, one warp per chunk, with a per-chunk `cudec_chunk_result`. The call is
  asynchronous: its return value covers argument validation and launch
  submission, and the per-chunk statuses are valid once the stream reaches the
  end of the launch.
- **Snappy raw-stream batch decode on the GPU.**
  `cudec_snappy_decompress_batch` decodes N independent raw Snappy streams -
  the varint-prefixed block `google/snappy`'s own `Compress` produces - in one
  launch on a caller-provided stream, and the batch contract holds argument for
  argument against the LZ4 entry above: the same device-side arrays, the same
  per-chunk `cudec_chunk_result`, the same synchronous argument rejects, the
  same launch limit, and the same asynchronous launch. The Snappy FRAMING
  format - stream identifier, framed chunks, masked CRC-32C - is not decoded
  here and no entry point decodes it, so a framed stream is a malformed raw one
  and is refused with `CUDEC_ERR_CORRUPT_INPUT` rather than partially decoded.
  The declared uncompressed length that opens every raw stream is
  attacker-controlled and sizes nothing: it is checked against that chunk's
  capacity before an element is parsed, a declaration above the capacity reports
  `CUDEC_ERR_OUTPUT_TOO_SMALL`, and every later overrun of it reports
  `CUDEC_ERR_CORRUPT_INPUT`. There is no Snappy streaming entry point and no
  milestone carries one: Snappy is batch-only by decision, not by omission.
- **`.lz4` frame decode.** `cudec_lz4f_decompress` decodes the frame format's
  block-independent subset, host memory to host memory, driving the batch
  decoder internally. The header, block and content checksums and the optional
  declared content size are verified fail-closed when present. A valid frame
  outside that subset - block-linked mode, which is liblz4's own frame-compressor
  default, or a dictionary id - is refused with `CUDEC_ERR_UNSUPPORTED` rather
  than decoded partially.
- **Streaming decode over pinned host memory.** `cudec_stream_ctx_create` /
  `cudec_lz4_decompress_stream_ctx` / `cudec_stream_ctx_destroy` decode from and
  to host memory through a reusable context that owns its CUDA stream and its
  pinned staging, plus `cudec_lz4_decompress_stream` as the one-shot wrapper for
  a caller that does not want to hold a context. A context that hits a CUDA
  fault is poisoned: it decodes nothing further and only its destructor is valid
  on it.
- **A defined status for every outcome.** `cudec_status` covers argument
  rejects, corrupt input, an output buffer too small, and CUDA faults, with
  `CUDEC_ERR_UNSUPPORTED` for a stream cudec understands and will not decode and
  `CUDEC_ERR_NOT_IMPLEMENTED` for an entry point that is declared here and not
  built in this configuration. No exception crosses the ABI and every CUDA
  call's error is checked.
- **The GDeflate batch entry, frozen ahead of its kernel.**
  `cudec_gdeflate_decompress_batch` takes the same arguments as the LZ4 and
  Snappy entries and refuses the same argument classes; its surface is the raw
  GDeflate page with a caller-supplied size and capacity, and no container is
  parsed. This build carries no GDeflate kernel, so a batch the entry accepts
  is answered `CUDEC_ERR_NOT_IMPLEMENTED` and no CUDA call is made. The symbol
  and the signature do not move when the kernel lands; only the answer to an
  accepted batch does.
- **`cudec_version()`** and the `CUDEC_VERSION_*` macros, single-sourced: the
  build system reads the version out of the public header, and a conformance
  test refuses a mismatch between the two.
- **An install surface.** `cmake --install` produces a prefix carrying the
  public header, the archive, and a package config, so an outside project links
  cudec through `find_package(cudec)` and the exported `cudec::cudec` target
  instead of vendoring the tree. The compatibility rule is `SameMinorVersion`,
  which at 0.x means a consumer asking for 0.0 is not handed 0.1, and CI drives
  a consumer in both directions of that rule. Both build flavours install and
  are consumed out of tree in CI: the consumer writes nothing about CUDA,
  because a prefix from a CUDA build carries its own
  `find_dependency(CUDAToolkit)` and a host-only prefix carries none, so it
  stays consumable where no CUDA toolkit exists. Consuming a CUDA-flavoured
  prefix does require the toolkit on the consuming machine and a project that
  enables C++, both stated in [README.md](README.md).

- **`CUDEC_ENABLE_HIP`, the HIP build of the same sources.** The option
  compiles the one set of device sources through CMake's HIP language for an
  explicit architecture list (`gfx90a`, `gfx942`, `gfx1100` and `gfx1201` by
  default, overridable), links `hip::host`, and refuses the configure where no
  HIP toolchain is found rather than falling back to a host-only build,
  because a build that quietly dropped the device engine and reported success
  would read exactly like a working port. `cudec_stream_t` keeps its CUDA
  driver pointer type on that build; a HIP caller passes its stream through it
  with a cast. It is mutually exclusive with `CUDEC_ENABLE_CUDA` in one build
  tree. Compiled in a ROCm container and never executed on an AMD device: no
  AMD numbers exist and none are claimed.

### Changed

- **`BUILD_SHARED_LIBS=ON` is refused in a top-level cudec build** rather than
  producing a shared library with no SOVERSION and no symbol-visibility policy.
  A project that vendors cudec as a subdirectory is unaffected: it gets a static
  cudec and keeps its own setting.

### Supported subset, stated as limits rather than implied

- **Formats, stated one at a time.** One sentence covering four formats is read
  as a promise in the direction that costs a consumer the most, so each format
  states its own limit and none of them can be refuted by resolving a symbol out
  of `include/cudec.h`.
  - **LZ4** decodes on the GPU: block batch, the `.lz4` frame subset bounded
    below, and the pinned-host streaming path.
  - **Snappy** decodes on the GPU, raw streams only, through the batch entry and
    nothing else. The framing format is not decoded by any entry point.
  - **GDeflate** has a declared entry point whose contract is frozen and no
    kernel behind it. A batch it accepts is answered
    `CUDEC_ERR_NOT_IMPLEMENTED`, and no CUDA call is made.
  - **Zstd** has host-side format parsers under `src/` and no public surface at
    all: `include/cudec.h` names no Zstd entry point, on any build.
  - **The HIP port** compiles and has not run. `CUDEC_ENABLE_HIP` produces a
    build that no AMD device has executed, so nothing about its behaviour on
    one is claimed.

  The milestone table in [README.md](README.md) carries the same list against
  the milestones that own each one.

- Frames must be block-independent and carry no dictionary id.
- Decode only. Nothing here compresses.
- NVIDIA GPUs. The default architecture list is `80` plus `86-real`, and the
  recorded numbers are all from an `sm_86` device. The CUDA
  engine is opt-in at configure time (`-DCUDEC_ENABLE_CUDA=ON`) and fails the
  configure without a toolchain rather than quietly building a host-only
  library.
- One deliberate divergence from liblz4, documented and tested: cudec refuses a
  match offset of 0, which liblz4 tolerates. cudec's accept set is a strict
  subset of the reference's on that one point.

### Guarantees, as tested

- **Fail-closed decode.** A malformed, truncated or hostile chunk produces a
  defined error with `bytes_written == 0`, never an out-of-bounds access and
  never partial output presented as success. Each format is held to its own
  reference as the authority on validity - liblz4 1.10.0 for LZ4, google/snappy
  1.2.2 for Snappy, including the three malformed streams snappy's own test
  suite carries - over the fixture corpus, its mutants and hand-crafted
  negatives; the parser twins run that comparison on the GPU-less CI runner and
  the device twins repeat it on hardware.
- **Determinism.** Same input, bit-identical output. The levels this holds at,
  and what each one is measured against, are in
  [docs/DETERMINISM.md](docs/DETERMINISM.md); `determinism_gpu` covers the
  launch-geometry axis a same-batch-twice comparison cannot see.
- **Termination.** Every parser call consumes at least one byte, asserted per
  call over the truncation ladder, the mutant corpus and the crafted streams, so
  a decode loop cannot hang the device on any input.
- **Fuzz-tested, read from a run rather than from a status.** Ten libFuzzer
  differential targets under `fuzz/` - LZ4 block and frame, Snappy block,
  GDeflate page, tables and tile stream, Zstd frame, FSE, literals and decode -
  replay a hash-pinned seed corpus bounded on every pull request and long on a
  schedule (`.github/workflows/fuzz.yml`). That workflow is not a required
  check, so the claim is cut from the run on the release pull request's head,
  which that pull request names with its conclusion and its link, and never
  from a run older than the commit it is made about.
- **Structural invariants locked as tests rather than as prose.** The C ABI
  carries no dependency beyond the device runtime, the decode path is
  integer-only, warp collectives may not take their participation metadata from
  parsed input, no bare wave-width literal is allowed outside the line that
  names it, and every ctest entry carries a timeout. These are configure-time
  refusals with their own liveness probes.

### Measured

Baselines are recorded, never quoted bare: every entry in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) carries the methodology block its
harness emitted - GPU, driver, CUDA version, corpus, chunk-size distribution and
run count - and
[docs/BENCHMARK-METHODOLOGY.md](docs/BENCHMARK-METHODOLOGY.md) is the write-up
that reads them. That file also records the perf passes whose optimizations were
measured and **rejected**, which is the part a changelog usually loses.

### Known gaps

- **The sanitizer block is empty, and says so.** Compute Sanitizer cannot
  attach to the device on my machine, which #258 closed with as its negative
  result, so the four-tool sweep has never produced a run there. The gate is
  parked and owed, not waived: the runner script is in the tree, #127 holds the
  block, and nothing in this release is claimed to have passed it. The release
  pull request records that absence with this reason rather than a coverage no
  run backs.
- **The GPU tests run on my hardware, not in CI.** CI builds the host-only,
  CUDA and HIP configurations and runs the host-side subset on the GPU-less
  runner; the device gate is a local run. At this release's commit the device
  was not reachable from the build environment (#418), so the device gate this
  release stands on is the last one recorded on #82, from 2026-08-25, and not a
  run on this commit; the release pull request says so in the same words.
