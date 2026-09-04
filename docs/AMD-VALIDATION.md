# AMD validation runbook

cudec compiles for AMD through HIP and **no AMD GPU has ever executed it**.
Every on-device AMD correctness result and every AMD performance number in
this project has to come from somebody else's hardware, so this document is
the apparatus for producing one: a clean clone to a pasteable result, in one
sitting, without asking anybody anything.

You need an AMD GPU, a container runtime, and disk for the container image
and the two benchmark corpora. You do not need to know anything about this
project.

## What has run and what has not

Read this before you read a command, because it decides how to take a
failure you hit.

- **The build and the host-side tests run green in exactly this image, on
  every pull request.** The `hip` job in `.github/workflows/ci.yml` uses the
  same digest quoted below, on a runner with no AMD GPU, and it compiles
  every test source and runs the label-excluded subset.
- **Nothing below has run on an AMD device.** The `gpu`-labelled tests and
  the benchmarks exist in the HIP build as binaries nothing has executed.
  You are the first.
- **The container's device-passthrough flags are the unverified part.** No
  machine in this project has an AMD GPU, so the `--device` lines in step 3
  have never been exercised here. If your setup needs a different set, use
  yours, and paste the invocation you actually ran with your result.

If something below is wrong, that is a defect in this document and worth an
issue on its own - it is meant to be executable without correspondence.

## 1. Clone

```sh
git clone https://github.com/iderex/cudec.git
cd cudec
```

## 2. Find your device's `gfx` target and wave width

Both go in your report, and the first one goes in the build command. On a
host with ROCm installed:

```sh
rocminfo | grep -E 'Marketing Name|^\s+Name:\s+gfx|Wavefront Size'
```

If ROCm is only inside the container, run the same line inside the container
from step 3 instead.

The wave width matters to this project more than it usually does: the decode
kernel is a `WaveSize` template and both instantiations ship in the same
binary, so a wave64 device (CDNA, `gfx90a`, `gfx942`) exercises an arm that a
wave32 device (RDNA, `gfx1100`, `gfx1201`) never reaches, and no device has
ever executed the wave64 one.

## 3. Build in the pinned container

The image is pinned by digest, and it is the same digest CI uses, so your run
and CI are one environment rather than two:

```sh
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  -v "$PWD:/w" -w /w \
  rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054 \
  bash
```

Everything from here to step 6 runs inside that shell. The image carries the
HIP toolchain; install what the build and the corpus fetcher need on top of
it - `git` and `cmake` are the step the CI job runs, `curl` and `unzip` are
what step 5 uses:

```sh
apt-get update -q && apt-get install -yq --no-install-recommends \
  git cmake curl unzip
```

Build for your own device, and name the architecture explicitly:

```sh
cmake -B build-hip -DCUDEC_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build-hip -j
```

Substitute the `gfx` target step 2 printed. Leaving
`CMAKE_HIP_ARCHITECTURES` off builds the four-architecture fat binary CI
builds (`gfx90a;gfx942;gfx1100;gfx1201`), which is slower and buys you
nothing here; the build never auto-detects a device, so an unset list is a
default rather than a probe.

## 4. Run the tests

The whole suite, device tests included. This is the line no AMD device has
run:

```sh
ctest --test-dir build-hip --no-tests=error --output-on-failure
```

`--no-tests=error` is what turns an empty selection red instead of green.
Keep the whole output; it is what you paste.

The entries that carry the `gpu` label are the ones that touch your device,
and they are the point of the exercise:

```sh
ctest --test-dir build-hip -L gpu -N
```

CI's own HIP listing names eight of them - `smoke`, `gpu_fixture`,
`frame_twin`, `stream_twin`, `termination_gpu`, `determinism_gpu`,
`snappy_block_device` and `wave_width_gpu` - and the CUDA build registers the
same eight plus `bench_frame_selfcheck`.
`frame_twin` and `stream_twin` are the oracle diffs
against liblz4, `determinism_gpu` is the bit-identical-across-geometry run,
`termination_gpu` is the hostile-corpus watchdog, and `wave_width_gpu`
decodes at the width your runtime reports. The rest of the suite -
`oracle_lz4`, `parser_twin`, `snappy_parser_twin`, `tilestream_host_negative`,
`termination`, `launch_fail`, `conformance_c` and the twin and probe sets -
is host-side, runs without a device, and is green in CI on both backends
already.

A failure in any of them is a result and worth reporting exactly as much as a
pass. Do not fix it, do not skip it, paste it.

## 5. Fetch the corpora and run the benchmarks

The corpora are SHA-256-pinned and never committed; the script refuses any
file whose hash does not match:

```sh
sh bench/get-corpora.sh
```

Then the four runs, each of which prints its own methodology block:

```sh
./build-hip/bench/bench_lz4 --gpu bench/corpora/silesia/*
./build-hip/bench/bench_lz4 --gpu --gpu-stream-ctx bench/corpora/silesia/*
./build-hip/bench/bench_lz4 --gpu --worst4b
./build-hip/bench/bench_lz4 --gpu --longmatch
```

The first two time the real corpus; `--worst4b` and `--longmatch` build their
adversarial corpora in-harness and validate them against the oracle before
timing, so they take no files. Every run is 3 warmup plus 30 measured
iterations by default - `--runs N` and `--warmup N` change that, and a report
that changed either says so.

## 6. What to paste, and where

**Paste every report block whole and unedited.** The harness emits its
methodology inseparably from its numbers, on purpose
([BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md)): a summarised or
hand-trimmed block is not a usable result and will not be recorded. The same
goes for the `ctest` output - the whole thing, passes included.

With it, four facts that no output above carries on its own:

- the GPU's marketing name and its `gfx` target, from step 2;
- the wave width, from step 2;
- the ROCm version: 7.2.4 if you used the image above unchanged, otherwise
  the one you built with;
- the `docker run` invocation you actually used, if it differs from step 3.

Open an issue on <https://github.com/iderex/cudec/issues> with all of it. A
result from wave64 silicon has an address of its own, issue #415, which is
open precisely because that dispatch arm has never executed anywhere; the
purpose-built intake form is #246.

## What an hour buys, measured where it could be measured

The build half is measured, in this image, on a GitHub `ubuntu-latest`
runner with no GPU - the `hip` job of CI run 33747806509, at `b1400ff`:

| Step                            | Duration |
| ------------------------------- | -------- |
| Pull and start the container    | 28 s     |
| `apt-get install git cmake`     | 7 s      |
| Configure and build, four `gfx` | 64 s     |
| Host-side `ctest` (`-LE gpu`)   | 16 s     |

Under two minutes for everything except the parts that need your device.
What is added on your side - the `gpu`-labelled tests, the corpus download,
and four benchmark runs at 33 iterations each - **has never been timed by
anybody**, because no AMD device has run it. The hour this runbook is written
to fit in is therefore a budget rather than a measurement, and a report
saying it took longer is itself a useful result.
