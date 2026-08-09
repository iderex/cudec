# The benchmark harness: honest numbers or none - every report emits its
# methodology block (docs/MASTERPLAN.md section 5). The CPU-oracle path is
# always timed; --gpu adds the device-resident GPU decode path (gpu_bench.cu,
# which shares the shipped kernel via src/lz4_decode.cuh). Consumer-only by
# rule: this directory links project targets and never modifies them (the
# conformance snapshot in tests/ runs before it).
find_package(CUDAToolkit REQUIRED)

add_executable(bench_lz4 bench_lz4.cpp gpu_bench.cu)
target_link_libraries(bench_lz4 PRIVATE cudec cudec_test_fixtures lz4_oracle
                                        CUDA::cudart)
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
target_include_directories(bench_zstd
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../tests)
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

# Every ctest entry carries a finite TIMEOUT (issue #72): the self-checks run
# the same corpus construction and CPU decode the decoder is held to, so a
# non-terminating decode must red them rather than stall the run. The
# assertion below reds the configure step if an entry added here ever misses
# one.
set_tests_properties(
  bench_selfcheck bench_worst4b_selfcheck bench_longmatch_selfcheck
  bench_zstd_worst_selfcheck
  PROPERTIES TIMEOUT ${CUDEC_TEST_TIMEOUT_SECONDS})
cudec_assert_test_timeouts()
