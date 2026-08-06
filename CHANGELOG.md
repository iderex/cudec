# Changelog

All notable changes to this project are recorded here, in the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format, and the
version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

What "notable" means here: a change a consumer of the C ABI, the build, or the
recorded numbers can observe. Internal refactors, test-net work and process
changes stay in the git history and in their issues, where they are already
readable.

No version has been released yet. `include/cudec.h` reports 0.0.1 and no tag
exists, so the entry below sits under Unreleased rather than under a number and
a date this project has not chosen: the released heading is the release's to
write, together with the version bump (#82). Everything in it is in the tree
today.

## [Unreleased]

### Added

- **LZ4 block batch decode on the GPU.** `cudec_lz4_decompress_batch` decodes
  N independent LZ4 block-format chunks in one launch on a caller-provided
  stream, one warp per chunk, with a per-chunk `cudec_chunk_result`. The call is
  asynchronous: its return value covers argument validation and launch
  submission, and the per-chunk statuses are valid once the stream reaches the
  end of the launch.
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
  `CUDEC_ERR_NOT_IMPLEMENTED` reserved for entry points that do not exist yet.
  No exception crosses the ABI and every CUDA call's error is checked.
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

- **`CUDEC_ENABLE_HIP`, a build option that currently refuses.** It is here so
  the HIP port has a fail-closed contract to land against, and it fails the
  configure on every machine: without a HIP toolchain it says so, and with one
  it says that cudec carries no HIP device sources yet. Neither case falls back
  to a host-only build, because a build that quietly dropped the device engine
  and reported success would read exactly like a working port. It is also
  mutually exclusive with `CUDEC_ENABLE_CUDA` in one build tree.

### Changed

- **`BUILD_SHARED_LIBS=ON` is refused in a top-level cudec build** rather than
  producing a shared library with no SOVERSION and no symbol-visibility policy.
  A project that vendors cudec as a subdirectory is unaffected: it gets a static
  cudec and keeps its own setting.

### Supported subset, stated as limits rather than implied

- LZ4 only. Snappy, GDeflate, Zstd and the HIP port are planned milestones with
  no code in this release; the milestone table in
  [README.md](README.md) is the current state.
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
  never partial output presented as success. Held to liblz4 1.10.0 as the
  authority on validity over the fixture corpus, its mutants and hand-crafted
  negatives; the parser twin runs that comparison on the GPU-less CI runner and
  the device twins repeat it on hardware.
- **Determinism.** Same input, bit-identical output. The levels this holds at,
  and what each one is measured against, are in
  [docs/DETERMINISM.md](docs/DETERMINISM.md); `determinism_gpu` covers the
  launch-geometry axis a same-batch-twice comparison cannot see.
- **Termination.** Every parser call consumes at least one byte, asserted per
  call over the truncation ladder, the mutant corpus and the crafted streams, so
  a decode loop cannot hang the device on any input.
- **Structural invariants locked as tests rather than as prose.** The C ABI
  carries no dependency beyond the CUDA runtime, the decode path is
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

- Compute Sanitizer cannot attach to the device on the maintainer's machine, so
  the four-tool sweep has never produced a clean run there. The runner script is
  in the tree and the blocker is tracked; nothing in this release is claimed to
  have passed it.
- The GPU tests run on the maintainer's hardware, not in CI, which has no GPU.
  CI builds both configurations and runs the host-side subset.
