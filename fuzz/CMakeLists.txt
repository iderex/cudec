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
include(${PROJECT_SOURCE_DIR}/cmake/CudecGDeflateOracle.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/CudecZstdOracle.cmake)

function(cudec_add_fuzz_target name)
  cmake_parse_arguments(F "" "SHARE" "SOURCES;LIBS" ${ARGN})
  # SHARE is how much of the fuzz job's ONE wall-clock budget this target is
  # worth, relative to its siblings. Nothing here reads the number: it is
  # declared at the target because that is where a new target is written, and
  # scripts/fuzz-budget.sh divides the budget by it, argues there why the
  # shares are not equal, and refuses a target that declares none. The keyword
  # is named here all the same, because an undeclared keyword lands in
  # F_UNPARSED_ARGUMENTS and this function would silently accept a
  # misspelling of it.
  #
  # The _selftest twins declare none, and are exempt in the script for the
  # reason they exist: they are replayed once over the committed seeds to prove
  # the parity comparison is live, never fuzzed, so a budget for them would be
  # time spent on a binary whose findings are never read.
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

cudec_add_fuzz_target(fuzz_lz4_block SHARE 1 SOURCES fuzz_lz4_block.cpp LIBS
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

# The LZ4 frame envelope (issue #141): the container above the block target,
# and the one LZ4 surface a hostile .lz4 file reaches before any block is
# looked at. It links lz4_frame_oracle rather than lz4_oracle because the
# reference here is liblz4's frame API - LZ4F_decompress over the whole
# container, checksums included - and that lives in the frame translation
# units the block oracle does not carry. The target's own header says which
# of the two host-only shapes issue #141 asks about it took, and why.
cudec_add_fuzz_target(fuzz_lz4_frame SHARE 1 SOURCES fuzz_lz4_frame.cpp LIBS
                      lz4_frame_oracle)
cudec_add_fuzz_target(fuzz_lz4_frame_selftest SOURCES fuzz_lz4_frame.cpp LIBS
                      lz4_frame_oracle)
target_compile_definitions(fuzz_lz4_frame_selftest
                           PRIVATE CUDEC_FUZZ_SELFTEST_BREAK)

# The Snappy pair (issue #90), the same two binaries for the same two reasons.
cudec_add_fuzz_target(fuzz_snappy_block SHARE 1 SOURCES fuzz_snappy_block.cpp
                      LIBS snappy_oracle)
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
cudec_add_fuzz_target(fuzz_gdeflate_tilestream SHARE 1 SOURCES
                      fuzz_gdeflate_tilestream.cpp)
cudec_add_fuzz_target(fuzz_gdeflate_tilestream_selftest SOURCES
                      fuzz_gdeflate_tilestream.cpp)
target_compile_definitions(fuzz_gdeflate_tilestream_selftest
                           PRIVATE CUDEC_FUZZ_SELFTEST_BREAK)

# The GDeflate table pipeline (issue #186): the canonical construction and the
# dynamic block's code-length rounds that feed it. It links no oracle library
# and for a sharper reason than the walker above: the reference HAS a decode
# table builder, and it is static inside a translation unit compiled with
# HIDE_INTERFACE, so no comparison against it is possible at any boundary this
# tree links. The property is a codespace count computed beside the builder
# instead. SHARE 2 for the reason fuzz_zstd_fse carries the same number: this
# is a structure-aware target whose reachable state keeps growing with the time
# it is given, not an envelope that saturates in seconds.
cudec_add_fuzz_target(fuzz_gdeflate_tables SHARE 2 SOURCES
                      fuzz_gdeflate_tables.cpp)
cudec_add_fuzz_target(fuzz_gdeflate_tables_selftest SOURCES
                      fuzz_gdeflate_tables.cpp)
target_compile_definitions(fuzz_gdeflate_tables_selftest
                           PRIVATE CUDEC_FUZZ_SELFTEST_BREAK)

# The whole GDeflate page decode against the vendored oracle (issue #192): the
# schedule, the code-length rounds, the table construction, the block loop and
# the in-tile LZ77 in one entry point, diffed against
# libdeflate_gdeflate_decompress. It links gdeflate_oracle for that reason,
# which neither GDeflate target above does. SHARE 3 for the reason
# fuzz_zstd_decode carries the same number: this is the composition rather than
# a unit, and it is where the budget keeps buying reach.
cudec_add_fuzz_target(fuzz_gdeflate_page SHARE 3 SOURCES fuzz_gdeflate_page.cpp
                      LIBS gdeflate_oracle)
cudec_add_fuzz_target(fuzz_gdeflate_page_selftest SOURCES
                      fuzz_gdeflate_page.cpp LIBS gdeflate_oracle)
target_compile_definitions(fuzz_gdeflate_page_selftest
                           PRIVATE CUDEC_FUZZ_SELFTEST_BREAK)

# The Zstd frame and block header walk (issue #180), the one Zstd surface raw
# fuzzer bytes reach without a structure-aware layer. It links zstd_oracle -
# the decode-only translation units - because nothing here produces a frame;
# the seed corpus is committed bytes. ZSTD_STATIC_LINKING_ONLY is for
# ZSTD_getFrameHeader, which lives in libzstd's static API, the same reason
# tests/zstd_frame_twin.cpp defines it.
cudec_add_fuzz_target(fuzz_zstd_frame SHARE 1 SOURCES fuzz_zstd_frame.cpp LIBS
                      zstd_oracle)
target_compile_definitions(fuzz_zstd_frame PRIVATE ZSTD_STATIC_LINKING_ONLY)
cudec_add_fuzz_target(fuzz_zstd_frame_selftest SOURCES fuzz_zstd_frame.cpp
                      LIBS zstd_oracle)
target_compile_definitions(
  fuzz_zstd_frame_selftest PRIVATE ZSTD_STATIC_LINKING_ONLY
                                   CUDEC_FUZZ_SELFTEST_BREAK)

# The Zstd FSE table description decode and the table built from it (issue
# #190). It links zstd_oracle rather than zstd_full: this target only ever
# READS a description, and FSE_readNCount and FSE_buildDTable_wksp are both in
# the decode translation units. FSE_STATIC_LINKING_ONLY is what exposes
# FSE_buildDTable_wksp, FSE_decode_t and the two sizing macros; the non-wksp
# entry point is not exported by the pinned release. It is defined here rather
# than in the source for the reason issue #180 landed - the two spellings are a
# redefinition the strict-warning build refuses.
cudec_add_fuzz_target(fuzz_zstd_fse SHARE 2 SOURCES fuzz_zstd_fse.cpp LIBS
                      zstd_oracle)
target_compile_definitions(fuzz_zstd_fse PRIVATE FSE_STATIC_LINKING_ONLY)
cudec_add_fuzz_target(fuzz_zstd_fse_selftest SOURCES fuzz_zstd_fse.cpp LIBS
                      zstd_oracle)
target_compile_definitions(
  fuzz_zstd_fse_selftest PRIVATE FSE_STATIC_LINKING_ONLY
                                 CUDEC_FUZZ_SELFTEST_BREAK)

# The Zstd literals section: the header, the Huffman tree description in both
# spellings, the canonical table build, the jump table and the four streams
# (issue #191). It links zstd_oracle for the reason fuzz_zstd_fse gives - this
# target only ever DECODES - and it needs no STATIC_LINKING_ONLY define,
# because the reference it compares against is ZSTD_decompress on a whole
# frame rather than an internal entry point. The target's own header says why
# the frame is there: a literals section is not something libzstd decodes on
# its own, so an envelope is what gives the reference a question to answer.
cudec_add_fuzz_target(fuzz_zstd_literals SHARE 2 SOURCES fuzz_zstd_literals.cpp
                      LIBS zstd_oracle)
cudec_add_fuzz_target(fuzz_zstd_literals_selftest SOURCES
                      fuzz_zstd_literals.cpp LIBS zstd_oracle)
target_compile_definitions(fuzz_zstd_literals_selftest PRIVATE
                           CUDEC_FUZZ_SELFTEST_BREAK)

# The whole host Zstd frame decode (issue #194): every M5 unit in the order the
# format runs them, driven through the same host driver the CPU twin test runs.
# It links zstd_oracle for the reason fuzz_zstd_fse gives - this target only
# ever DECODES - and it needs no STATIC_LINKING_ONLY define, because both
# reference entry points it uses, ZSTD_decompress and
# ZSTD_findFrameCompressedSize, are in the stable API. Its structure-aware pass
# builds a frame twice, once without a declared content size to measure the
# body and once with, and the target's own header says why that is the only way
# the sequence execution is reachable inside a subset that requires the
# declaration.
cudec_add_fuzz_target(fuzz_zstd_decode SHARE 3 SOURCES fuzz_zstd_decode.cpp LIBS
                      zstd_oracle)
cudec_add_fuzz_target(fuzz_zstd_decode_selftest SOURCES fuzz_zstd_decode.cpp
                      LIBS zstd_oracle)
target_compile_definitions(fuzz_zstd_decode_selftest PRIVATE
                           CUDEC_FUZZ_SELFTEST_BREAK)
