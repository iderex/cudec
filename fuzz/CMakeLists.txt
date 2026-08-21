# The fuzz targets (issue #140). Built only under -DCUDEC_FUZZ=ON; the
# top-level guard is what decides that, and it has already refused a
# non-Clang toolchain and a fuzz build with no sanitizers by the time this
# file is read.
#
# No ctest entry is registered here, deliberately. A fuzz target's natural run
# is unbounded, and every ctest entry in this tree must carry a finite TIMEOUT
# (issue #72) - a bounded -runs=N smoke entry is CI's shape and belongs to the
# CI wiring issue, not to the harness. This directory is also processed after
# the timeout sweep has run, so a test added here would escape it entirely.

include(${PROJECT_SOURCE_DIR}/cmake/CudecLz4Oracle.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/CudecSnappyOracle.cmake)

function(cudec_add_fuzz_target name)
  cmake_parse_arguments(F "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${name} ${F_SOURCES})
  target_link_libraries(${name} PRIVATE ${F_LIBS})
  # src/ for the single-sourced parser, tests/ for the shared host driver the
  # CPU twin test runs. Nothing here links the cudec archive: the parser is a
  # header and the target must not drag the CUDA runtime into a host-only
  # fuzz build.
  target_include_directories(
    ${name} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
                    ${PROJECT_SOURCE_DIR}/tests)
  set_target_properties(${name} PROPERTIES CXX_STANDARD 17
                                           CXX_STANDARD_REQUIRED ON)
  # -fsanitize=fuzzer on top of the requested sanitizers, on both the compile
  # and the link: the compile flag is what instruments the coverage the engine
  # steers on, the link flag is what supplies its main(). C/C++ only, the same
  # generator-expression guard the sanitizer flags carry, so no CUDA
  # translation unit could ever see them.
  target_compile_options(
    ${name}
    PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Werror;-fsanitize=fuzzer>
            $<$<COMPILE_LANGUAGE:C,CXX>:${CUDEC_SANITIZE_COMPILE_FLAGS}>)
  target_link_options(${name} PRIVATE -fsanitize=fuzzer
                      ${CUDEC_SANITIZE_LINK_FLAGS})
endfunction()

cudec_add_fuzz_target(fuzz_lz4_block SOURCES fuzz_lz4_block.cpp LIBS
                      lz4_oracle)

# The second binary of the pair, and the reason the first one's verdict can be
# believed: identical sources with CUDEC_FUZZ_SELFTEST_BREAK defined, which
# perturbs an accepted oracle output so the parity comparison MUST fire. A
# harness that had silently stopped comparing passes as this one, so building
# it is how the guard is shown to bite. It is never the binary whose findings
# are read.
cudec_add_fuzz_target(fuzz_lz4_block_selftest SOURCES fuzz_lz4_block.cpp LIBS
                      lz4_oracle)
target_compile_definitions(fuzz_lz4_block_selftest
                           PRIVATE CUDEC_FUZZ_SELFTEST_BREAK)

# The Snappy pair (issue #90), the same two binaries for the same two reasons.
cudec_add_fuzz_target(fuzz_snappy_block SOURCES fuzz_snappy_block.cpp LIBS
                      snappy_oracle)
cudec_add_fuzz_target(fuzz_snappy_block_selftest SOURCES fuzz_snappy_block.cpp
                      LIBS snappy_oracle)
target_compile_definitions(fuzz_snappy_block_selftest
                           PRIVATE CUDEC_FUZZ_SELFTEST_BREAK)

# The GDeflate TileStream envelope walker (issue #184), and the first target
# here with no reference behind it: the container has no oracle in this tree,
# so the property is an invariant over what the walker ACCEPTS rather than a
# comparison against another implementation. It links no oracle library for
# that reason. The twin below breaks the same way its siblings do - it
# perturbs an accepted result so the checks must fire - which is what keeps
# "no oracle" from also meaning "nothing proves the checks run".
cudec_add_fuzz_target(fuzz_gdeflate_tilestream SOURCES
                      fuzz_gdeflate_tilestream.cpp)
cudec_add_fuzz_target(fuzz_gdeflate_tilestream_selftest SOURCES
                      fuzz_gdeflate_tilestream.cpp)
target_compile_definitions(fuzz_gdeflate_tilestream_selftest
                           PRIVATE CUDEC_FUZZ_SELFTEST_BREAK)
