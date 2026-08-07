# The Zstd corpus

What `tests/zstd_corpus.cpp` generates, which decode surface each family
reaches, and how the corpus proves it reached it. The generator and its
self-proof are `tests/zstd_corpus.cpp` and
`tests/zstd_corpus_selfproof.cpp`; issue #185 is where the shape was argued.

Run it:

    ctest --test-dir build-cuda -R zstd_corpus_selfproof --output-on-failure

## Why the source bytes are built and not fetched

The self-proof has to be green on the CI runner, and that runner fetches no
corpus:

    grep -c -i "silesia\|get-corpora" .github/workflows/ci.yml
    0

So the two things cannot be one artifact. Every family below constructs its
own bytes, in the shape the existing bench self-checks already use, and none
of them reads a file.

The multi-level Silesia rung lives with the fetch instead. `MakeZstdBatchFrames`
takes the source bytes as an argument for exactly that reason: pointing it at
`bench/get-corpora.sh`'s Silesia needs no change here, and the batch geometry
it emits is the same one the self-proof checks on constructed bytes. What the
CI-gated proof therefore does NOT cover is Silesia's own level sweep at
Silesia's own scale. That is not covered anywhere yet, and no number or
property in this document should be read as covering it.

## What each family asks for, and what proves it got it

Each fixture carries a demand: the modes that must be present in the frame it
produced. The self-proof walks the emitted headers and fails the fixture when
a demanded mode is absent, so a compressor that declines a forced parameter
reds the suite instead of quietly costing coverage. The walker reads headers
only, and never entropy-decodes anything.

| Family   | Fixture                              | Surface reached                                |
| -------- | ------------------------------------ | ---------------------------------------------- |
| envelope | envelope-content-size-and-checksum   | content size present, XXH64 checksum present   |
| envelope | envelope-no-content-size-no-checksum | content size absent, checksum absent           |
| envelope | envelope-single-segment              | Single_Segment frame, no window descriptor     |
| envelope | envelope-windowed                    | windowed frame, many blocks                    |
| block    | block-raw                            | Raw block                                      |
| block    | block-rle                            | RLE block, not the frame's first               |
| block    | block-compressed                     | Compressed block                               |
| literals | literals-raw                         | Raw literals inside a compressed block         |
| literals | literals-rle-handbuilt               | RLE literals, no sequences                     |
| literals | literals-compressed-one-stream       | Huffman literals, 1 stream                     |
| literals | literals-compressed-four-stream      | Huffman literals, 4 streams and the jump table |
| literals | literals-treeless                    | Treeless literals reusing the previous tree    |
| tables   | tables-basic                         | Set_Basic for all three sequence fields        |
| tables   | tables-rle                           | Set_RLE for offsets and match lengths          |
| tables   | tables-compressed                    | Set_Compressed for all three fields            |
| tables   | tables-repeat                        | Repeat for all three fields                    |
| level    | level-18, level-19                   | the high-search family                         |
| level    | level-22-long-window                 | the high-search family with a 128 MiB window   |
| batch    | MakeZstdBatchFrames                  | independent frames over one source, rejoined   |

The self-proof also pins the list of cells above as a set. Removing a family
or weakening its demand reds the coverage check, so this table cannot drift
away from the corpus without the suite saying so.

## Two cells the compressor would not produce

**RLE literals.** A block needs a one-symbol literal alphabet AND enough
literals for a Huffman attempt to be made at all. Both together were not
reachable through the advanced API: the sources that make every literal
identical also make the run matchable, the match finder absorbs the literals
into an offset-1 match, and what comes back is Raw literals. Every variant
tried came back Raw, so the fixture is a hand-built frame instead, and the
oracle round-trip is what establishes it is legal rather than plausible. That
is the shape `tests/zstd_probes.cpp` already uses for behaviours no
compressor emits.

**Set_RLE for literal lengths.** On the one-symbol-per-field source the
compressor emitted Set_RLE for offsets and match lengths and a compressed
table for literal lengths. That is the measurement; no cause is claimed for
it. The cell itself is already covered in the tree, by the hand-built frames
in `tests/zstd_probes.cpp`, which set Symbol_Compression_Mode 1 for all three
fields:

    grep -n "RLE sequence tables" tests/zstd_probes.cpp

## The digest pin

The self-proof pins an FNV-1a digest over every fixture's name and frame
bytes, the instrument `tests/oracle_lz4.cpp` and `tests/snappy_corpus.cpp`
both carry. A zstd bump that moves the compressor's output moves every
fixture with it, and the bump then has to update the pin on purpose instead of
watching a changed corpus pass. FNV-1a is a drift tripwire, not a defence.
