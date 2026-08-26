# The fuzz targets

Differential libFuzzer targets over the format parsers. Each one drives
arbitrary bytes into the single-sourced parser from `src/`, runs the same host
driver the CPU twin test runs, and compares the result in process against the
hash-pinned reference decoder.

The list below names what each target reads and what answers it. It is not the
authority for which targets exist - `CMakeLists.txt` is, and the job and the
corpus check both derive the list from it rather than from here:

```
sed -n 's/^cudec_add_fuzz_target(\([A-Za-z0-9_]*\).*/\1/p' fuzz/CMakeLists.txt \
  | grep -v '_selftest$' | LC_ALL=C sort -u
```

| target                     | parser                 | reference                                    |
| -------------------------- | ---------------------- | -------------------------------------------- |
| `fuzz_lz4_block`           | `src/lz4_block.h`      | `LZ4_decompress_safe`, liblz4 1.10.0         |
| `fuzz_lz4_frame`           | `src/lz4_frame.h`      | `LZ4F_decompress`, liblz4 1.10.0             |
| `fuzz_snappy_block`        | `src/snappy_block.h`   | `snappy::RawUncompress`, google/snappy 1.2.2 |
| `fuzz_gdeflate_tilestream` | `src/tilestream.h`     | none; invariants over what it accepts        |
| `fuzz_zstd_frame`          | `src/zstd_frame.h`     | `ZSTD_getFrameHeader`, zstd 1.5.7            |
| `fuzz_zstd_fse`            | `src/zstd_fse.h`       | `FSE_readNCount`, zstd 1.5.7                 |
| `fuzz_zstd_literals`       | `src/zstd_literals.h`  | `ZSTD_decompress`, zstd 1.5.7                |
| `fuzz_zstd_decode`         | the whole M5 host path | `ZSTD_decompress`, zstd 1.5.7                |

The property every target asserts is the fail-open direction: where the parser
accepts, the reference must accept the same bytes and produce the identical
output and size. The opposite direction is asserted only where a target can
show the two sides were given the same question to answer, because a parser
that is stricter than its reference is usually the fail-closed contract
working, not a defect. Three targets can. `fuzz_zstd_fse` hands the
reference the alphabet and accuracy-log ceilings the unit under test was given,
over a padded copy that keeps the reference's own end-of-buffer handling out of
the comparison, and holds both sides to each other from there.
`fuzz_zstd_literals` splices the fuzzer's bytes in as the literals section of a
real frame and asks `ZSTD_decompress` about the frame, so both sides answer for
the same section under the same window; it declares one strictness, the
eleven-bit cap RFC 8878 puts on a literals tree where the reference reads
twelve, and it identifies a refusal as that one by re-reading the description at
the reference's ceiling rather than by guessing from the rung. Where a
deliberate strictness exists instead, it is pinned in the twin test rather than
here (`tests/parser_twin.cpp` holds the LZ4 `offset == 0` family).
`fuzz_lz4_frame` is the third: both sides walk one `.lz4` container from its
magic number, so a frame `LZ4F_decompress` decodes end to end and the parser
calls corrupt is a valid container refused rather than a strictness. It
declares one, a block header whose 31-bit length masks to zero, identifies it
from the bytes rather than from the verdict, and it is pinned as a negative in
`tests/frame_host_negative.cpp` (`block-blen-zero`) so the exemption is not a
hole nothing covers. A skippable frame was the second until the walk stopped
calling it corrupt (#379); it is now refused as `UNSUPPORTED`, which this
target's stricter-direction check never sees.

`fuzz_zstd_decode` is the only target that enters more than one unit, and the
two things it can say follow from that. It runs the same host driver the CPU
twin test runs, so a stage order that is wrong, per-frame state that does not
survive a block boundary, or a section read at an offset the previous stage did
not leave are all visible to it and to none of the unit targets. It also runs
two passes: raw bytes as a whole frame, which is where the frame header, the
subset decisions, the checksum and the bytes-after-the-frame rule live; and a
structure-aware pass that holds a frame header, a block header and a Raw
literals section fixed and lets the fuzzer write the sequences section, which is
the only way the repeat-offset rules and the sequence execution are reached at
all. Its declared exception is a buffer holding a frame followed by anything
else, which the reference walks and the subset refuses (masterplan section
12.4); like the literals target's, it is identified rather than guessed at and
is printed once per run. What its second pass cannot compare, and why, is
argued in the target's own header rather than restated here.

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
