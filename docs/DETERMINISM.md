# Determinism

cudec's decode path is **`gpu_to_gpu` deterministic**: the same compressed
input produces bit-identical output on any supported GPU, in any launch
configuration, in any run.

The level names come from NVIDIA's CCCL determinism vocabulary
(`not_guaranteed` / `run_to_run` / `gpu_to_gpu`), borrowed here so the promise
has a citable name and a testable scope instead of the adjective "deterministic".
Borrowing the terminology is all it is: cudec depends on nothing beyond the CUDA
runtime, and no CCCL dependency is added.

## What the level promises

For one compressed chunk and one destination capacity, the decoded bytes, the
per-chunk status, and `bytes_written` are identical:

| Axis                  | Promise                                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Run to run            | Identical, same process or a fresh one.                                                                               |
| Launch configuration  | Identical for any grid size, block size, chunk-to-warp mapping, or number of concurrent streams.                      |
| Batch composition     | Identical whether the chunk is decoded alone, in a batch, or in a differently ordered batch — chunks are independent. |
| Entry point           | Identical through the batch entry, the frame decoder, and the streaming context, fresh or reused.                     |
| GPU to GPU            | Identical on any device the library supports (`sm_80` baseline and newer).                                            |
| Driver / CUDA version | Identical: the arithmetic is integer and the output-byte mapping is fixed at the source level, not at the ISA level.  |

## Why it holds

The reason is structural, not statistical — the pipeline has nothing
non-deterministic in it to control:

- **No floating point.** A lossless decompressor has no use for it. There is no
  FP type, no FP atomic, and therefore no reassociation, no fused-multiply-add
  contraction, and no dependence on the order in which partial results arrive.
  A configure-time check in `tests/CMakeLists.txt` reds the build if any FP type
  or FP atomic appears in the sources.
- **Every output byte is written exactly once**, by a statically determined
  lane, as a pure function of the input and of bytes at lower addresses. An
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
