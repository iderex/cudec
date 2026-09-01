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
working, not a defect. Four targets can. `fuzz_zstd_fse` hands the
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

`fuzz_gdeflate_page` is the fourth, and it identifies a strictness from the
RUNG rather than from the bytes, which is the difference the reject ladder
(#183) bought. Both sides are handed the same page and the same capacity, so
the questions match; a refusal names one of the twenty-two branches of
`enum GDeflateReject`, and the target traps unless
`GDeflateRejectIsDeclaredDeparture` in `src/gdeflate_schedule.h` declares that
branch. Three are declared - the read past the last word of the page, the use
of an empty code, and the code-length repeat run that overruns HLIT + HDIST -
and each is argued at its own refusal site. The exemption is held by
`tests/gdeflate_departure_lock.cpp`, which requires a page the reference
decodes for every declared branch, so a list grown to silence a trap reds a
test instead. Over-strictness matters more here than for the other three:
GDeflate carries no checksum anywhere, so a stream the reference decompresses
and cudec calls corrupt reaches the caller with nothing to appeal to.

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

## The structure-aware layer over the two GDeflate targets

A GDeflate page is 32 least-significant-bit-first streams interleaved word by
word in the order a decoder's refills happen, so a block header is spread
across 32 lanes and every field in it sits at a bit offset that depends on
every field before it. Random bytes die in the code-length rounds, and the
table construction, the block loop and the in-tile LZ77 behind them are never
entered. `gdeflate_structure.h` is the answer: a generator of pages whose
envelope is correct by construction - BFINAL, BTYPE, HLIT, HDIST, HCLEN, a
precode that is a code, and literal/length and distance vectors that are codes

- and whose interior is whatever the engine's bytes say. It emits through
  `cudec_test::EmitPage`, the same mirror of `src/gdeflate_schedule.h` the header
  fixtures drive, so the round order exists once.

It is switchable off at run time, and the switch is what the numbers below were
taken with:

```
CUDEC_FUZZ_STRUCTURED=0 scripts/fuzz-coverage.sh build-fuzz fuzz_gdeflate_page 60 1
CUDEC_FUZZ_STRUCTURED=1 scripts/fuzz-coverage.sh build-fuzz fuzz_gdeflate_page 60 1
```

An environment switch rather than a build one, so both arms of the comparison
are the same binary. Unstructured mutation keeps running inside the layer as
well: only half of its mutations synthesise a page, a quarter perturb the words
behind an envelope, and a quarter are handed to libFuzzer's own mutator.

`scripts/fuzz-coverage.sh` is the route from a run to a figure. It reports the
`cov:` count off libFuzzer's `DONE` line - the instrumented edges the run
executed - and refuses a run that produced no such line, which is what a crash
or a kill leaves behind. What it cannot do is attribute an edge to a source
line, and it does not claim to.

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
