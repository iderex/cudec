# The fuzz targets

Differential libFuzzer targets over the format parsers. Each one drives
arbitrary bytes into the single-sourced parser from `src/`, runs the same host
driver the CPU twin test runs, and compares the result in process against the
hash-pinned reference decoder.

| target              | parser               | reference                                    |
| ------------------- | -------------------- | -------------------------------------------- |
| `fuzz_lz4_block`    | `src/lz4_block.h`    | `LZ4_decompress_safe`, liblz4 1.10.0         |
| `fuzz_snappy_block` | `src/snappy_block.h` | `snappy::RawUncompress`, google/snappy 1.2.2 |

The property each target asserts is the fail-open direction: where the parser
accepts, the reference must accept the same bytes and produce the identical
output and size. The opposite direction is not asserted, because a parser that
is stricter than its reference is the fail-closed contract working. Where a
deliberate strictness exists it is pinned in the twin test rather than here
(`tests/parser_twin.cpp` holds the LZ4 `offset == 0` family).

The Snappy target asserts one thing more, and it is the one a verdict cannot
carry: the driver counts every element the parser handed back that a caller
could not have executed without leaving its buffers, and that count must stay
zero whatever the verdict is. A reject that arrived only after an over-read is
still an over-read.

## Building and running

The build is opt-in, needs Clang, and refuses to build a fuzzer without
sanitizers:

```
cmake -B build-fuzz -DCUDEC_FUZZ=ON -DCUDEC_SANITIZE=address,undefined \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-fuzz
build-fuzz/fuzz/fuzz_lz4_block <corpus-dir> -max_total_time=60 -max_len=4096
```

No target is registered as a ctest entry. A fuzz run is unbounded and every
ctest entry in this tree owes a finite timeout.

## The selftest binaries

Every target is built twice. The `_selftest` twin is the same source with
`CUDEC_FUZZ_SELFTEST_BREAK` defined, which perturbs an accepted reference
output so the comparison must fire. It is what shows the comparison is live
without waiting for a real divergence to turn up, and it is never the binary
whose findings are read. A `_selftest` binary that runs to its time limit
instead of trapping means the harness has stopped comparing.

## Regression seeds

An input that ever crashed a target, tripped a sanitizer, or diverged from a
reference is kept forever. It is not enough to fix the parser and move on: the
input that found the defect is the only proof the fix holds, and a fuzzer
reseeded from scratch may take a very long time to find it again.

Two artifacts per finding, and one of them is not optional:

- The input becomes a permanent seed in the committed corpus.
- The same bytes become a negative in `tests/` pinning the exact status the
  parser now returns, so the non-fuzz gate covers it too. A corpus entry alone
  is covered only while the fuzz job runs; a pinned negative is covered on
  every pull request.

The corpus lives in `corpus/<target>/`, one directory per target, and
`corpus/SHA256SUMS` pins every seed by hash. `scripts/check-fuzz-corpus.sh`
refuses a changed seed, a seed the manifest does not name, an emptied target
directory and a target declared in `CMakeLists.txt` with no corpus at all; the
target list it checks against is read out of `CMakeLists.txt` rather than
written down a second time. Adding a seed means adding its line to the
manifest:

```
cd fuzz/corpus && find . -type f ! -name SHA256SUMS | sed 's|^\./||' \
  | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
```

`.github/workflows/fuzz.yml` runs the targets: every seed replayed on every
pull request, then a bounded exploration, with a longer scheduled run whose
corpus and crashing inputs are uploaded as artifacts. The seeds are replayed
before anything explores, which is what makes a kept regression seed cheap -
a defect that comes back is caught in seconds rather than by a fuzzer
rediscovering it.
