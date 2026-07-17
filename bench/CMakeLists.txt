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
