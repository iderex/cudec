# The benchmark harness: honest numbers or none - every report emits its
# methodology block (docs/MASTERPLAN.md section 5). The CPU-oracle path is
# always timed; --gpu adds the device-resident GPU decode path (gpu_bench.cu,
# which shares the shipped kernel via src/chunk_decode.cuh). Consumer-only by
# rule: this directory links project targets and never modifies them (the
# conformance snapshot in tests/ runs before it).
find_package(CUDAToolkit REQUIRED)

add_executable(bench_lz4 bench_lz4.cpp gpu_bench.cu)
# lz4_frame_oracle is the frame COMPRESSOR for --frame (issue #132): the
# harness has to produce block-independent .lz4 frames to time
# cudec_lz4f_decompress on, and liblz4's frame compressor defaults to the
# linked mode cudec refuses. Nothing here decodes through it - the frame
# decode under test is cudec's.
target_link_libraries(bench_lz4 PRIVATE cudec cudec_test_fixtures lz4_oracle
                                        lz4_frame_oracle CUDA::cudart)
# gpu_bench.cu includes the internal kernel header from src/.
target_include_directories(bench_lz4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set_target_properties(
  bench_lz4 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                       CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                       CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
# -O2 unconditionally on the host path: an empty CMAKE_BUILD_TYPE (the
# documented container command) would otherwise time -O0 marshaling around
# the decoder. The CUDA path carries the project's strict device flags.
target_compile_options(
  bench_lz4
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;-O2>
          ${CUDEC_CUDA_STRICT_FLAGS})

# Rot protection: one short self-checked run on the built-in corpus as a
# host-side ctest entry, so the harness cannot silently break between
# milestones. (--selfcheck stays CPU-only so it runs on the GPU-less CI
# runner; the GPU path is exercised by the /bench command and BENCHMARKS.md.)
add_test(NAME bench_selfcheck COMMAND bench_lz4 --selfcheck)

# The worst-4Bmatch corpus is constructed in-harness and oracle-validated
# before timing; this runs that construction through the same self-checked
# path (a few chunks, CPU-only) so a broken generator - one whose stream the
# oracle no longer accepts, or that stops round-tripping - reds CI, not just
# a manual bench.
add_test(NAME bench_worst4b_selfcheck COMMAND bench_lz4 --worst4b --selfcheck)

# The long-non-overlap-match corpus (issue #36) is likewise hand-built and
# oracle-validated before timing, with a compression-ratio shape lock. This
# runs that construction through the self-checked path (a few chunks,
# CPU-only) so a generator that stops round-tripping, or drifts off the
# intended many-long-matches shape, reds CI rather than a manual bench.
add_test(NAME bench_longmatch_selfcheck COMMAND bench_lz4 --longmatch
                                                --selfcheck)

# The game-asset-like corpus (issue #139). Its source block is generated and
# its wire is ordinary LZ4_compress_default output, so the round-trip alone
# would pass on almost any block; the lock is over four quantities that
# together name the regime - compression ratio, literal share, sequence count
# and mean match length - because that regime is the whole reason the corpus
# exists. A generator that lost a noise source drifts toward the compressible
# corpora and one that lost its structure drifts toward pure noise; both leave
# a band and red CI rather than a manual bench. CPU-only, like the two above.
add_test(NAME bench_assetlike_selfcheck COMMAND bench_lz4 --assetlike
                                                --selfcheck)

# The frame sweep (issue #132). Unlike the four above this one cannot be
# CPU-only: --frame times cudec_lz4f_decompress, which decodes on the
# device, so the rot check that runs the same construction has to run there
# too. It therefore carries the gpu label and the GPU-less CI runner defers
# it to the local gate, exactly like frame_twin. RUN_SERIAL for the reason
# every gpu test here has it: one device, one occupant.
add_test(NAME bench_frame_selfcheck COMMAND bench_lz4 --frame --selfcheck)
set_tests_properties(bench_frame_selfcheck PROPERTIES LABELS gpu RUN_SERIAL
                                                      TRUE)

# The Snappy CPU denominator (issue #164). No CUDA in it at all: there is no
# Snappy kernel yet, so the only decoder this times is the reference, and the
# binary needs neither a device nor a kernel header. It links the same
# oracle target tests/ pins and the same fixture library, because the Snappy
# streams in this project have ONE provenance (tests/fixtures.h) and a bench
# that compressed its own would be a second one nobody reconciles.
add_executable(bench_snappy bench_snappy.cpp)
target_link_libraries(bench_snappy PRIVATE cudec cudec_test_fixtures
                                           snappy_oracle)
# The corpus digest reads src/xxhash64.h, the hash already in the tree.
target_include_directories(bench_snappy PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set_target_properties(bench_snappy PROPERTIES CXX_STANDARD 17
                                              CXX_STANDARD_REQUIRED ON)
# -O2 unconditionally for the reason bench_lz4 gives above: the documented
# container command sets no CMAKE_BUILD_TYPE, and an -O0 reference decoder
# would be timed instead of the reference decoder.
target_compile_options(bench_snappy PRIVATE -Wall -Wextra -Werror -O2)

# Same rot protection as the entries above, and one more thing besides. The
# selfcheck builds both corpus shapes from a fixed PRNG and asserts the
# digest of each, so a moved compressor pin, a changed chunk size or a
# generator that lost its noise source reds CI - none of which would stop the
# corpus round-tripping, which is why the round trip alone is not the check.
# CPU-only, so it runs on the GPU-less runner.
add_test(NAME bench_snappy_selfcheck COMMAND bench_snappy --selfcheck)

# The maximum-element-density corpus (issue #166). Kept separate from the
# Silesia path above because the two rot differently: that one rots on shape
# and digests, this one rots on adversariality. Its construction is checked
# twice before anything is timed - the oracle on validity, the density floors
# on whether it is still the worst case it reports - and this entry runs the
# identical construction on a handful of chunks. CPU-only, like the entry
# above.
add_test(NAME bench_snappy_worst_selfcheck COMMAND bench_snappy --worst
                                                   --selfcheck)

# The parse-bound corpus (issue #119). Separate from the entry above because
# the two lock different quantities: that one is bounded by elements per
# decoded byte at four bytes each, this one at one, and throughput is reported
# per decoded byte. A density lock on the copy chain says nothing about this
# margin, so it gets its own construction, its own two locks and its own
# entry. CPU-only, like the entries above.
add_test(NAME bench_snappy_worstlit_selfcheck COMMAND bench_snappy --worstlit
                                                      --selfcheck)

# The Zstd worst-case corpus (issue #229). No CUDA in it at all: the frames
# are constructed on the host, emitted through the reference's own
# sequence-compression entry point and decoded by the reference, so the
# binary needs neither a device nor the cudec kernel.
#
# It compiles tests/zstd_corpus.cpp rather than restating it: the frame
# walker there is what the shape lock reads modes and sequence counts out of,
# and the forced-mode fixture set beside it is what the table-reuse refusal
# is proven to still bite on. That file is consumed exactly as the tests
# consume it, which is the rule this directory follows - it links and
# compiles what tests/ defines and modifies none of it.
#
# zstd_full and never zstd_oracle: this corpus has to PRODUCE frames, and
# tests/CMakeLists.txt records why a binary reaching for both archives is a
# link the linker rightly refuses.
add_executable(bench_zstd bench_zstd.cpp ../tests/zstd_corpus.cpp)
target_link_libraries(bench_zstd PRIVATE zstd_full)
# ../src for the corpus digest, which reads src/xxhash64.h - the hash already
# in the tree rather than a new dependency for a bench.
target_include_directories(
  bench_zstd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../tests
                     ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set_target_properties(bench_zstd PROPERTIES CXX_STANDARD 17
                                            CXX_STANDARD_REQUIRED ON)
# -O2 unconditionally for the reason bench_lz4 gives above: the documented
# container command sets no CMAKE_BUILD_TYPE, and an -O0 decode loop times
# the marshaling rather than the decoder.
target_compile_options(bench_zstd PRIVATE -Wall -Wextra -Werror -O2)

# Same rot protection as the two above, and the same reasons: a generator
# whose frames the oracle stops accepting, or that drifts off the
# maximum-density / no-table-reuse / repeat-offset shape it is named for,
# reds CI rather than a manual bench. CPU-only, so it runs on the GPU-less
# runner.
add_test(NAME bench_zstd_worst_selfcheck COMMAND bench_zstd --worst
                                                 --selfcheck)

# The standard Zstd corpus path (issue #227). The recorded run reads the
# fetched Silesia files, which the CI runner does not have, so the selfcheck
# runs the identical grid - both granularity endpoints, the whole level set -
# over a source generated from a fixed PRNG, and asserts the digest of each
# cell. That is what makes it a rot check rather than a run: a moved
# compressor pin, a changed granularity or a generator that lost its noise
# source all move a digest, and none of them would stop the frames
# round-tripping. CPU-only, so it runs on the GPU-less runner.
add_test(NAME bench_zstd_selfcheck COMMAND bench_zstd --selfcheck)

# The forced-mode corpus as a coverage run (issue #227). It consumes the
# fixture generator tests/ pins rather than building a second one, and it
# reports which decode surfaces were reached instead of a throughput number -
# fixtures sized to reach a surface each cannot carry one. CPU-only.
add_test(NAME bench_zstd_coverage_selfcheck COMMAND bench_zstd --coverage
                                                    --selfcheck)

# Every ctest entry carries a finite TIMEOUT (issue #72): the self-checks run
# the same corpus construction and CPU decode the decoder is held to, so a
# non-terminating decode must red them rather than stall the run. The
# assertion below reds the configure step if an entry added here ever misses
# one.
set_tests_properties(
  bench_selfcheck bench_worst4b_selfcheck bench_longmatch_selfcheck
  bench_assetlike_selfcheck bench_zstd_worst_selfcheck bench_zstd_selfcheck
  bench_zstd_coverage_selfcheck bench_frame_selfcheck bench_snappy_selfcheck
  bench_snappy_worst_selfcheck bench_snappy_worstlit_selfcheck
  PROPERTIES TIMEOUT ${CUDEC_TEST_TIMEOUT_SECONDS})
cudec_assert_test_timeouts()
