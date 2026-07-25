# Determinism

cudec's decode path is **`gpu_to_gpu` deterministic**: the same compressed
input produces bit-identical output on any supported GPU, in any supported
launch configuration, in any run.

"Supported launch configuration" is a real qualifier, not a hedge. The kernel's
chunk-to-lane mapping requires a block size that is a whole number of warps and
a grid holding at least one whole warp; the shipped entry point always launches
`kBlockThreads`, and the kernel rejects any other geometry outright by
returning without decoding, rather than producing output nobody wrote. See
"What the level does not promise" below.

The level names come from NVIDIA's CCCL determinism vocabulary
(`not_guaranteed` / `run_to_run` / `gpu_to_gpu`), borrowed here so the promise
has a citable name and a testable scope instead of the adjective "deterministic".
Borrowing the terminology is all it is: cudec depends on nothing beyond the CUDA
runtime, and no CCCL dependency is added.

## What the level promises

For one compressed chunk and one destination capacity, the decoded bytes, the
per-chunk status, and `bytes_written` are identical:

| Axis                  | Promise                                                                                                                                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Run to run            | Identical, same process or a fresh one.                                                                                                                                                               |
| Launch configuration  | Identical for any grid size, any block size that is a whole number of warps, any chunk-to-warp mapping, and any number of concurrent streams. Other block sizes are refused, not decoded differently. |
| Batch composition     | Identical whether the chunk is decoded alone, in a batch, or in a differently ordered batch — chunks are independent.                                                                                 |
| Entry point           | Identical through the batch entry, the frame decoder, and the streaming context, fresh or reused.                                                                                                     |
| GPU to GPU            | Identical on any device the library supports (`sm_80` baseline and newer).                                                                                                                            |
| Driver / CUDA version | Identical: the arithmetic is integer and the output-byte mapping is fixed at the source level, not at the ISA level.                                                                                  |

## Why it holds

The reason is structural, not statistical — the pipeline has nothing
non-deterministic in it to control:

- **No floating point.** A lossless decompressor has no use for it. There is no
  FP type, no FP atomic, and therefore no reassociation, no fused-multiply-add
  contraction, and no dependence on the order in which partial results arrive.
  A configure-time check in `tests/CMakeLists.txt` reds the build if any FP type
  or FP atomic appears in the sources.
- **Every output byte is written exactly once**, by a statically determined
  lane — given a supported launch geometry, which the kernel enforces rather
  than assumes: the copy loops stride by the warp size, so a block that is not
  a whole number of warps would leave a fixed slice of every destination
  written by nobody. That geometry returns without decoding instead. An
  overlapping match is not a copy that chases itself but a modular gather
  `dst[d + i] = dst[d - off + (i mod off)]`, reading only the region a
  `__syncwarp()` has already frozen (masterplan section 9). There is no
  read-modify-write, no atomic accumulation, and no inter-warp communication, so
  no ordering exists to get wrong.
- **Chunks are independent.** A chunk's output depends on its own source bytes
  and its own capacity, never on its neighbours or on where in the batch it sits.
  That is what makes the launch-geometry and stream-count axes free.
- **No ambient input.** No clock, no random source, no device-property query,
  and no heuristic that switches algorithm by size feed the decode.

## What the level does not promise

- **Nothing about rejected chunks' destination contents.** On any non-OK status
  `bytes_written` is 0 and the destination is explicitly unspecified — the
  decoder never presents partial output as success, and the bytes it happened to
  write before rejecting are not part of the contract. The _status_ for a given
  input is deterministic; the leftover bytes are not.
- **Nothing about unsupported launch geometry.** A block size that is not a
  whole number of warps, or a grid holding less than one whole warp, is refused
  by the kernel: it returns without touching the destinations or the result
  records, so the caller sees whatever it primed them with. That is a defined
  refusal, not a decode — and it is only reachable through the internal kernel
  header, never through the public ABI, which always launches `kBlockThreads`.
  `tests/termination_gpu.cu` covers `<<<1, 16>>>`, `<<<2, 16>>>`, and
  `<<<2, 48>>>`. One further geometry bound is **documented rather than
  enforced**: the kernel's thread index is 32-bit, so it requires
  `gridDim.x * blockDim.x <= 2^32`. Exceeding it does **not** break this
  promise — because a block is already forced to be a whole number of warps,
  the wrapped index stays warp-aligned, so an aliased block recomputes the
  same chunks with a full lane range and writes byte-identical values to the
  same addresses. The output is unchanged; the work is done twice. That
  asymmetry is why this one is documented and the other two are refused:
  enforcing it costs a real sm_86 occupancy step on every launch (three
  spellings measured, all 48 → 52 registers — [BENCHMARKS.md](BENCHMARKS.md)),
  the shipped grid is capped at 8192 blocks, and no public entry point can
  approach it.
- **Nothing about timing.** Throughput varies with geometry, occupancy, clocks,
  and neighbours. Determinism here is about bytes, never about duration.
- **Nothing about memory addresses.** Allocation addresses and the pointer values
  a caller passes are the caller's business; the decoded content does not depend
  on them.
- **Nothing across format-behaviour changes.** A cudec version that decodes a
  stream a previous version rejected is a deliberate, documented change, not a
  determinism break. The invariant is fixed input, fixed version, fixed output.

## How it is tested

- `tests/determinism_gpu.cu` — the launch-geometry axis. One corpus of pristine
  and mutated chunks, one reference decode, then five runs each across five
  grid/block geometries (a single warp walking the whole batch through the
  grid-stride loop, block sizes of 32/64/96/128, a grid far larger than the
  batch, and the shipped sizing) plus a three-stream split of the same batch.
  Every geometry here is a supported one; the refused geometries are covered by
  `tests/termination_gpu.cu` instead.
  The whole destination arena is re-poisoned before every run and compared in
  full — decoded bytes and the untouched poison beyond `bytes_written` — along
  with every per-chunk result record.
- `tests/stream_twin.cu` — the entry-point and context-reuse axes: the streaming
  decoder's output is bit-identical to a single-stream reference decode, on a
  fresh context and on a reused one whose staging has grown.
- `tests/gpu_fixture.cu` and `tests/frame_twin.cu` — the oracle axis: decoded
  bytes match liblz4's own output for every accepted stream.
- `tests/CMakeLists.txt` — the structural axis: the floating-point ban above,
  checked at configure time, with probes that prove the check still bites.

The GPU-to-GPU axis is **reasoned, not measured**: the maintainer's gate runs one
device (an RTX 3080, `sm_86`). It rests on the integer-only argument above rather
than on a two-GPU comparison, and it is stated that way deliberately — a claim
without a measurement is not a measurement. A second device joins the gate when
one is available.
