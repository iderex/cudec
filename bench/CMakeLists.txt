# The benchmark harness: honest numbers or none - every report emits its
# methodology block (docs/MASTERPLAN.md section 5). CPU-oracle only until
# M1 lands a kernel to time. Consumer-only by rule: this directory links
# project targets and never modifies them (the conformance snapshot in
# tests/ runs before it).
find_package(CUDAToolkit REQUIRED)

add_executable(bench_lz4 bench_lz4.cpp)
target_link_libraries(bench_lz4 PRIVATE cudec cudec_test_fixtures lz4_oracle
                                        CUDA::cudart)
set_target_properties(bench_lz4 PROPERTIES CXX_STANDARD 17
                                           CXX_STANDARD_REQUIRED ON)
# -O2 unconditionally: an empty CMAKE_BUILD_TYPE (the documented container
# command) would otherwise time -O0 marshaling around the decoder.
target_compile_options(
  bench_lz4 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;-O2>)

# Rot protection: one short self-checked run on the built-in corpus as a
# host-side ctest entry, so the harness cannot silently break between
# milestones.
add_test(NAME bench_selfcheck COMMAND bench_lz4 --selfcheck)
