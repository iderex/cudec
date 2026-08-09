# The test net: ctest is the runner, one executable per test group, the
# assertion header tests/require.h as the only "framework". Decision record
# and reassessment trigger: docs/MASTERPLAN.md section 5.

# Supply chain: the hash-verified tarball is the only acceptable oracle
# source - a system copy must never silently substitute for it.
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)

# liblz4 and its two oracle targets, lz4_oracle and lz4_frame_oracle. The
# declaration moved to cmake/CudecLz4Oracle.cmake when fuzz/ became a second
# consumer of the same pin (issue #140): FetchContent lets the first
# declaration win in silence, so two copies of the URL and hash could drift
# with nothing to say which one the build used.
include(FetchContent)
include(${PROJECT_SOURCE_DIR}/cmake/CudecLz4Oracle.cmake)

# google/snappy and its snappy_oracle target. The declaration moved to
# cmake/CudecSnappyOracle.cmake when fuzz/ became a second consumer of the
# same pin (issue #90), for the reason the liblz4 include above carries.
include(${PROJECT_SOURCE_DIR}/cmake/CudecSnappyOracle.cmake)

# The NVIDIA/libdeflate fork and its gdeflate_oracle target, the M4 oracle
# (issue #168). In its own module from the start rather than after a second
# consumer appears: the M4 fuzz rungs are that consumer, and the pin's three
# deviations from the standing policy are argued next to the pin instead of
# here.
include(${PROJECT_SOURCE_DIR}/cmake/CudecGDeflateOracle.cmake)

# facebook/zstd, the M5 oracle, pinned by the SHA-256 of the
# MAINTAINER-UPLOADED release asset - the shape the pinning policy prefers and
# lz4 already uses, rather than the auto-generated /archive/ tarball snappy
# had to fall back to.
#
# The pin, and how it was established at pin time (2026-08-06):
#
#   curl -sSL -o zstd-1.5.7.tar.gz \
#     https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
#   sha256sum zstd-1.5.7.tar.gz
#   eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
#
# self-computed, and it agrees with all of: the published
# zstd-1.5.7.tar.gz.sha256 asset beside it, and conan-center-index's
# recipes/zstd/all/conandata.yml for 1.5.7 - one second packaging ecosystem,
# as the policy asks. 2434947 bytes.
#
# A detached zstd-1.5.7.tar.gz.sig IS published (HTTP 200 at the same release)
# and it was DELIBERATELY NOT VERIFIED. Verifying it needs the signing key,
# and the only route to that key from here is the same host that serves the
# artifact and the signature, which makes the check circular rather than
# independent. The three hashes above come from two different parties, which
# is the property actually wanted. Said plainly so a later reader does not
# read this pin as signature-verified.
#
# Licence: zstd is dual-licensed BSD-2-Clause OR GPL-2.0 (LICENSE and COPYING
# in the tarball). cudec ELECTS BSD-2-Clause. Recorded here rather than only
# in a pull request because the election is a fact about this repository: the
# GPL arm is not taken, and nothing here may be read as taking it. cudec
# neither vendors nor redistributes these sources - the tarball is fetched at
# build time into the build tree and no cudec artifact contains it - so
# BSD-2-Clause's notice obligation attaches to no artifact shipped today. It
# attaches the moment a binary linking it is distributed, which is a thing
# this tree does not do and would have to decide to start.
FetchContent_Declare(
  zstd
  URL https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
  URL_HASH
    SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(zstd)

# The uploaded tarball carries no top-level CMakeLists (zstd's lives in
# build/cmake), so MakeAvailable only populates the source tree, exactly as
# with lz4 - and the same statement as a check rather than as a comment: if a
# future layout DID add one, MakeAvailable would build zstd's own project and
# these targets would collide with its.
if(TARGET libzstd_static)
  message(FATAL_ERROR "zstd's own build system was added to this tree - the "
                      "oracle rule is one target over the sources the decode "
                      "path needs, never the project's build system")
endif()

# The decode path, as an EXPLICIT translation-unit list rather than the
# amalgamated single-file decoder. Both were available; the list wins on the
# tree's own means test. The amalgamation is produced by running
# build/single_file_libs/combine.py at configure time, which puts a Python
# runtime on the build's critical path for every configure in CI and in the
# container - a language this tree does not otherwise carry - and hands the
# compiler a generated file no reviewer reads. The list below is nine names
# in a diff, and a tenth appearing is a reviewable event.
#
# The names are zstddeclib-in.c's own set, which is what the single-file
# decoder amalgamates, plus common/xxhash.c: a zstd frame may carry an XXH64
# content checksum, so the decoder reaches it. If that TU turns out to be
# unnecessary the link is what says so.
set(_zstd_decode_sources
    ${zstd_SOURCE_DIR}/lib/common/debug.c
    ${zstd_SOURCE_DIR}/lib/common/entropy_common.c
    ${zstd_SOURCE_DIR}/lib/common/error_private.c
    ${zstd_SOURCE_DIR}/lib/common/fse_decompress.c
    ${zstd_SOURCE_DIR}/lib/common/xxhash.c
    ${zstd_SOURCE_DIR}/lib/common/zstd_common.c
    ${zstd_SOURCE_DIR}/lib/decompress/huf_decompress.c
    ${zstd_SOURCE_DIR}/lib/decompress/zstd_ddict.c
    ${zstd_SOURCE_DIR}/lib/decompress/zstd_decompress.c
    ${zstd_SOURCE_DIR}/lib/decompress/zstd_decompress_block.c)
foreach(_zstd_source IN LISTS _zstd_decode_sources)
  if(NOT EXISTS ${_zstd_source})
    message(FATAL_ERROR "the zstd decode source ${_zstd_source} is missing: "
                        "the pinned tarball's layout has moved and this list "
                        "is stale")
  endif()
endforeach()

# ZSTD_LEGACY_SUPPORT=0 keeps the pre-v0.8 frame decoders out: cudec decodes
# the v1 subset and an oracle that accepts more than the thing under test
# would report a divergence that is not one. ZSTD_DISABLE_ASM=1 drops the
# BMI2 assembly path, which is x86-specific and irrelevant to a reference
# verdict. ZSTD_STRIP_ERROR_STRINGS because the harness reads
# ZSTD_getErrorCode, an enum, and never a message.
set(_zstd_defines ZSTD_LEGACY_SUPPORT=0 ZSTD_DISABLE_ASM=1
                  ZSTD_STRIP_ERROR_STRINGS)

add_library(zstd_oracle STATIC ${_zstd_decode_sources})
target_include_directories(zstd_oracle SYSTEM PUBLIC ${zstd_SOURCE_DIR}/lib)
target_compile_definitions(zstd_oracle PRIVATE ${_zstd_defines})
# SYSTEM include, none of the project's strict flags, -O2 unconditionally -
# the rule lz4_oracle and snappy_oracle both follow, for the same two reasons:
# third-party code is audited by hash rather than reformatted, and the
# documented container command sets no CMAKE_BUILD_TYPE.
target_compile_options(zstd_oracle PRIVATE -O2)

# The compressor as well, from the SAME populated tree, for corpus
# generation: an M5 corpus of level-swept, forced-mode frames cannot be
# written by hand. Only the oracle-diff target stays minimal, so the two are
# separate rather than one.
#
# They must never meet in one link. Every decode symbol is in both archives,
# so a binary reaching for both would resolve some from each and the linker
# would refuse it - correctly. A consumer takes zstd_oracle when it needs a
# verdict and zstd_full when it needs to produce a frame.
file(GLOB _zstd_full_sources CONFIGURE_DEPENDS ${zstd_SOURCE_DIR}/lib/common/*.c
     ${zstd_SOURCE_DIR}/lib/compress/*.c ${zstd_SOURCE_DIR}/lib/decompress/*.c)
if(NOT _zstd_full_sources)
  message(FATAL_ERROR "the zstd library sweep found no sources under "
                      "${zstd_SOURCE_DIR}/lib - the glob has stopped matching")
endif()
add_library(zstd_full STATIC ${_zstd_full_sources})
target_include_directories(zstd_full SYSTEM PUBLIC ${zstd_SOURCE_DIR}/lib)
target_compile_definitions(zstd_full PRIVATE ${_zstd_defines})
target_compile_options(zstd_full PRIVATE -O2)

# PRIVATE on purpose: the archive still links through, but the lz4, snappy and
# zstd include paths stay out of every consumer's command line - so an
# accidental #include <lz4.h>, <snappy.h> or <zstd.h> in a .cu test is a
# compile error, and the "nvcc never sees the oracle headers" rule is
# enforced, not just observed.
add_library(cudec_test_fixtures STATIC fixtures.cpp)
target_link_libraries(cudec_test_fixtures PRIVATE lz4_oracle snappy_oracle
                                                  zstd_oracle)
target_include_directories(cudec_test_fixtures PUBLIC .)
set_target_properties(cudec_test_fixtures PROPERTIES CXX_STANDARD 17
                                                     CXX_STANDARD_REQUIRED ON)
# -O2 for the harness's UNTIMED bulk work: corpus prep and the byte-verify
# pass run through this code over 200+ MB corpora (the timed region itself
# calls LZ4_decompress_safe in lz4_oracle directly and never enters
# fixtures code). Keeps setup time negligible under the build-type-less
# documented container command.
target_compile_options(
  cudec_test_fixtures
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;-O2>)
# Sanitizer gate (issue #42): the harness that builds and byte-verifies the
# corpora is host code on the untrusted-input path, so it runs under the
# sanitizer too. The lz4 oracle it links is deliberately left uninstrumented
# (third-party, audited by hash - the reference must not red the gate).
if(CUDEC_SANITIZE)
  target_compile_options(
    cudec_test_fixtures
    PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CUDEC_SANITIZE_COMPILE_FLAGS}>)
endif()

# GPU tests carry the label (the GPU-less CI runner selects -LE gpu) and
# run serially: one physical device, and deterministic device state is
# worth more than parallel wall time. A GPU test never self-skips - the
# banned pattern is recorded in docs/MASTERPLAN.md section 5.
function(cudec_add_test name)
  cmake_parse_arguments(T "GPU" "" "SOURCES;LIBS" ${ARGN})
  # The host-only sanitizer gate (issue #42) builds only the GPU-less,
  # host-instrumentable subset: a GPU test is a .cu binary linked by nvcc,
  # which cannot pull in the ASan-instrumented cudec without sanitizing the
  # device link. Skip them so the sanitizer build stays pure host.
  if(T_GPU AND CUDEC_SANITIZE)
    return()
  endif()
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE cudec ${T_LIBS})
  set_target_properties(
    ${name} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                       CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                       CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(
    ${name} PRIVATE ${CUDEC_CUDA_STRICT_FLAGS}
                    $<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Werror>)
  # Sanitizer flags on the host test's own TUs (C/C++ only) and its link, so
  # the instrumented cudec/fixtures objects pull in the runtimes. GPU tests
  # returned above, so this only ever reaches host executables.
  if(CUDEC_SANITIZE)
    target_compile_options(
      ${name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CUDEC_SANITIZE_COMPILE_FLAGS}>)
    target_link_options(${name} PRIVATE ${CUDEC_SANITIZE_LINK_FLAGS})
  endif()
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT
                                          ${CUDEC_TEST_TIMEOUT_SECONDS})
  if(T_GPU)
    set_tests_properties(${name} PROPERTIES LABELS gpu RUN_SERIAL TRUE)
  endif()
endfunction()

cudec_add_test(conformance_c SOURCES conformance_c.c)
set_target_properties(conformance_c PROPERTIES C_STANDARD 99
                                               C_STANDARD_REQUIRED ON)
# The top-level CMakeLists reads PROJECT_VERSION out of include/cudec.h; these
# hand the result back to the C side so that derivation is tested rather than
# only asserted (issue #80).
target_compile_definitions(
  conformance_c
  PRIVATE CUDEC_CMAKE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
          CUDEC_CMAKE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
          CUDEC_CMAKE_VERSION_PATCH=${PROJECT_VERSION_PATCH})
cudec_add_test(oracle_lz4 SOURCES oracle_lz4.cpp LIBS cudec_test_fixtures)
# The M3 oracle is only useful if it is present and callable; this is that,
# and nothing about cudec's own Snappy path, which does not exist yet.
cudec_add_test(oracle_snappy SOURCES oracle_snappy.cpp LIBS
               cudec_test_fixtures)
# The M4 oracle proving itself (issue #168), the counterpart of oracle_snappy
# one milestone later: the pin downloads and builds, both directions link, a
# stream it produced comes back byte-identical, and the three answering
# behaviours the M4 parity tests have to be written around are pinned rather
# than assumed. It links gdeflate_oracle alone - the shared fixture library
# carries no GDeflate anything yet, and adding this to it would put the fork's
# headers in front of every test that includes fixtures.h.
cudec_add_test(oracle_gdeflate SOURCES oracle_gdeflate.cpp LIBS
               gdeflate_oracle)
# The dossier's format facts executed against that oracle (issue #170), the
# GDeflate counterpart of snappy_probes below. It links gdeflate_oracle alone
# for the reason oracle_gdeflate gives above, and it is GPU-less because the
# facts are measured through the reference's public API rather than on a
# device.
cudec_add_test(gdeflate_probes SOURCES gdeflate_probes.cpp LIBS
               gdeflate_oracle)
# The reference's own decode behaviours, executed rather than read (issue
# #155). Host-side and GPU-less on purpose: it is the oracle answering, and
# the answers are what the M3 fail-closed matrix is allowed to lock against.
cudec_add_test(snappy_probes SOURCES snappy_probes.cpp LIBS
               cudec_test_fixtures)
# The Snappy corpus proving itself (issue #157), the counterpart of oracle_lz4:
# real pairs, a deterministic generator, and mutations that bite. Host-side, so
# it runs on the GPU-less runner.
cudec_add_test(snappy_corpus SOURCES snappy_corpus.cpp LIBS
               cudec_test_fixtures)
# The Snappy counterpart of parser_twin (issue #171): the single-source
# ladder from src/snappy_block.h executed on the host and held to snappy's
# verdicts. Host-side, so the GPU-less runner and the sanitizer gate both
# reach it - which is what makes the exactly-sized stream copy inside it
# worth having.
cudec_add_test(snappy_parser_twin SOURCES snappy_parser_twin.cpp LIBS
               cudec_test_fixtures)
target_include_directories(snappy_parser_twin
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
# The classes a mutation corpus cannot structurally build (issue #172): the
# literal length's extra bytes and the varint preamble's edge encodings, swept
# by hand against both reference calls. Its own binary rather than a section of
# the twin, because it is breadth over the same rungs the twin covers one
# stream each - and because a sweep that reds says which class moved.
cudec_add_test(snappy_edge_sweep SOURCES snappy_edge_sweep.cpp LIBS
               cudec_test_fixtures)
target_include_directories(snappy_edge_sweep
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
# snappy's own three negative vectors (issue #156), read out of the pinned
# archive rather than copied into this tree: the URL_HASH above already proves
# the bytes, and a copy would be three binary blobs restating it. The path is
# handed in here so the test names none of its own; provenance and the licence
# posture for everything under that directory are recorded in the test's header.
cudec_add_test(snappy_upstream_negatives SOURCES snappy_upstream_negatives.cpp
               LIBS cudec_test_fixtures)
target_include_directories(snappy_upstream_negatives
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_definitions(
  snappy_upstream_negatives
  PRIVATE CUDEC_SNAPPY_TESTDATA_DIR="${CUDEC_SNAPPY_SOURCE_DIR}/testdata")
# The other half of "single-source host AND device" (issue #171): the twin
# above compiles src/snappy_block.h with the host compiler and this one hands
# the same header to nvcc, then requires the two to produce an identical
# element trace. Without it the device half of the header is a comment. Not
# the device gate set (#154) - one thread, no warp cooperation, no kernel.
cudec_add_test(snappy_block_device SOURCES snappy_block_device.cu GPU)
if(TARGET snappy_block_device)
  target_include_directories(snappy_block_device
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()
# The M5 oracle proving itself (issue #181): the pin downloads, the two
# flavours build, and the compressor and the decode entry points agree on real
# bytes - plus one refusal per error class the fail-closed matrix will read.
# It links zstd_full and NOT cudec_test_fixtures: the fixtures archive carries
# the decode-only zstd, every decode symbol is in both archives, and a binary
# reaching for both would be refused by the linker. Host-side and GPU-less;
# nothing here is about cudec's own Zstd path, which does not exist yet.
cudec_add_test(zstd_oracle_smoke SOURCES zstd_oracle_smoke.cpp LIBS zstd_full)
# The M5 decode behaviours the research read out of RFC 8878 and the zstd
# source, executed instead of argued (issue #189). Same standalone shape and
# the same reason as zstd_oracle_smoke above: it drives the compressor for the
# one probe that needs a real Huffman-coded literals section, so it links
# zstd_full alone. ZSTD_STATIC_LINKING_ONLY is for ZSTD_frameHeaderSize, which
# is how that probe locates the first block without this test growing a second
# frame-header parser.
cudec_add_test(zstd_probes SOURCES zstd_probes.cpp LIBS zstd_full)
target_compile_definitions(zstd_probes PRIVATE ZSTD_STATIC_LINKING_ONLY)
# The Zstd backward bitstream reader, executed on the host before any FSE or
# Huffman code exists to run over it (issue #215). Every serial core in M5
# reads through this one unit, so it is proven alone: hand-written bit strings
# for expectations, one negative per reject rung. No oracle and no device.
# The M5 corpus proving itself before any Zstd decoder exists (issue #185),
# the counterpart of oracle_lz4 and snappy_corpus. It drives the compressor
# through its advanced API for the forced-mode families, so it links zstd_full
# alone for the reason zstd_oracle_smoke gives above. Host-side and GPU-less;
# the source bytes are constructed rather than fetched, because the runner
# this has to be green on fetches no corpus - docs/ZSTD-CORPUS.md carries that
# decision and the coverage matrix.
cudec_add_test(zstd_corpus_selfproof SOURCES zstd_corpus_selfproof.cpp
               zstd_corpus.cpp LIBS zstd_full)
# The Zstd mutation corpus proving itself (issue #187): the mutants are
# deterministic and digest-pinned, every fixture yields at least one the
# reference refuses, and the aimed header-field classes are asserted by name so
# a walker change cannot drop one in silence. It links zstd_full and compiles
# zstd_corpus.cpp for the reason zstd_corpus_selfproof gives above - the
# generator lives beside the corpus rather than in fixtures.cpp, because the
# decode-only and full zstd archives may never meet in one link.
cudec_add_test(zstd_mutants SOURCES zstd_mutants.cpp zstd_corpus.cpp LIBS
               zstd_full)
cudec_add_test(zstd_bitstream_twin SOURCES zstd_bitstream_twin.cpp)
target_include_directories(zstd_bitstream_twin
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
# The Zstd frame and block header parser and the v1 subset gate (issue #200):
# the first thing any Zstd decode path reads, and the only place a frame is
# judged admissible. Host-side and GPU-less, so the runner with no device
# reaches the gate and the sanitizer build instruments it. It links zstd_full
# alone for the reason zstd_oracle_smoke gives above, and
# ZSTD_STATIC_LINKING_ONLY is for ZSTD_getFrameHeader, which is the field-for-
# field oracle the positives are checked against rather than a second frame
# header parser written to check the first. It also holds src/xxhash64.h, the
# frame's content checksum (issue #202), to the checksums libzstd writes into
# real --check frames across every length class the algorithm branches on.
cudec_add_test(zstd_frame_twin SOURCES zstd_frame_twin.cpp LIBS zstd_full)
target_include_directories(zstd_frame_twin
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_definitions(zstd_frame_twin PRIVATE ZSTD_STATIC_LINKING_ONLY)
# The Zstd FSE table description decode and the table it builds (issue #217),
# held to FSE_readNCount and FSE_buildDTable_wksp cell by cell, plus the state
# cores that walk that table over a backward bitstream (issue #219), held to
# the reference's own decode of streams its compress pipeline wrote and of the
# Huffman weight descriptions the corpus carries. Host-side and
# GPU-less. It links zstd_full alone for the reason zstd_oracle_smoke gives
# above and because FSE_writeNCount lives on the compress side: the test
# writes the descriptions its negatives are mutations of, and proves its own
# writer against the reference's before using it. It compiles zstd_corpus.cpp
# for the same reason zstd_corpus_selfproof does - the #185 fixtures are where
# the Set_Compressed fields this sweeps over come from.
# FSE_STATIC_LINKING_ONLY is what exposes FSE_buildDTable_wksp and
# FSE_decode_t; the non-wksp entry point is not exported by the pinned
# release.
cudec_add_test(zstd_fse_twin SOURCES zstd_fse_twin.cpp zstd_corpus.cpp LIBS
               zstd_full)
target_include_directories(zstd_fse_twin
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_definitions(zstd_fse_twin PRIVATE FSE_STATIC_LINKING_ONLY)
cudec_add_test(parser_twin SOURCES parser_twin.cpp LIBS cudec_test_fixtures)
# The twin compiles the decoder core from src/ on the host - the
# fail-closed ladder runs on the GPU-less CI runner (masterplan section 9,
# ladder step 2).
target_include_directories(parser_twin
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
# The frame parse's reject branches and its host-only assembly path exercised
# with NO GPU: every case rejects (or decodes an all-uncompressed frame) before
# the first CUDA call, so it runs on the GPU-less runner and, under the
# sanitizer gate (issue #42), gives src/frame.cpp's hostile-input pointer
# arithmetic ASan/UBSan coverage that the GPU-labeled frame_twin cannot get in
# CI. It computes header checksums with the library's own xxhash32.h.
# Both entry points against the one shared launch limit (issue #39). Host-side:
# every call in it is an argument reject that returns before any CUDA call, so
# it runs on the GPU-less runner. It reads the limit from src/, like the twins.
cudec_add_test(batch_limit_parity SOURCES batch_limit_parity.cpp)
target_include_directories(batch_limit_parity
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

cudec_add_test(frame_host_negative SOURCES frame_host_negative.cpp)
target_include_directories(frame_host_negative
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
# The GDeflate TileStream envelope's reject branches (issue #174), the same
# shape and for the same reason: src/tilestream.h is pure host arithmetic over
# a byte range, so the whole container ladder is gated on the GPU-less runner
# and gets the ASan/UBSan pass over the pointer arithmetic a hostile envelope
# drives. No oracle: this is fail-closed coverage, and parity against the real
# DirectStorage vectors is issues #169 and #179.
cudec_add_test(tilestream_host_negative SOURCES tilestream_host_negative.cpp)
target_include_directories(tilestream_host_negative
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
# Termination as a tested invariant (issue #72): the parser's liveness
# contract asserted per call over the truncation ladder, the mutant corpus,
# the crafted hostile corpus, and the frame block-table walk - all of it
# rejecting (or assembling all-uncompressed) before the first CUDA call, so
# it runs on the GPU-less runner and under the sanitizer gate.
cudec_add_test(termination SOURCES termination.cpp LIBS cudec_test_fixtures)
target_include_directories(termination
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
cudec_add_test(launch_fail SOURCES launch_fail.cpp)
# Belt and suspenders with the setenv inside launch_fail.cpp: zero visible
# devices is the test condition, on the GPU machine too.
set_tests_properties(launch_fail PROPERTIES ENVIRONMENT
                                            "CUDA_VISIBLE_DEVICES=")
cudec_add_test(smoke SOURCES smoke.cu GPU)
cudec_add_test(gpu_fixture SOURCES gpu_fixture.cu LIBS cudec_test_fixtures
               GPU)
cudec_add_test(frame_twin SOURCES frame_twin.cu LIBS lz4_frame_oracle GPU)
# frame_twin verifies the library's own xxhash32.h against liblz4's XXH32.
# Guarded: the host-only sanitizer build (issue #42) skips the GPU targets, so
# this follow-up only runs when the target actually exists.
if(TARGET frame_twin)
  target_include_directories(frame_twin
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()
cudec_add_test(stream_twin SOURCES stream_twin.cu LIBS cudec_test_fixtures GPU)
# The same hostile corpus on the device, behind a host-side watchdog: a warp
# that never leaves its sequence loop holds the launch, so this one polls an
# event against a deadline instead of blocking on the stream (issue #72).
cudec_add_test(termination_gpu SOURCES termination_gpu.cu LIBS
               cudec_test_fixtures GPU)
# It also launches the kernel directly, to reach a geometry the shipped entry
# point never produces (fewer threads than one warp), so it needs the internal
# header from src/.
if(TARGET termination_gpu)
  target_include_directories(termination_gpu
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()
# Determinism across launch geometry - the axis the existing same-batch-twice
# compare cannot see. It launches the shipped kernel directly, so it needs the
# internal header from src/ (like bench/ does).
cudec_add_test(determinism_gpu SOURCES determinism_gpu.cu LIBS
               cudec_test_fixtures GPU)
if(TARGET determinism_gpu)
  target_include_directories(determinism_gpu
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()

# ---- Structural conformance, configure time: a violation reds every CI
# build. These are drift detectors, not tamper-proofing - the honest limits
# are recorded in docs/MASTERPLAN.md section 5.

# The library must depend on nothing beyond the CUDA runtime - checked on
# the private AND consumer-facing link surfaces (an INTERFACE dependency of
# a static library is exactly what its consumers inherit), and on link
# options, which could smuggle a -l dependency past the target lists.
foreach(_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES LINK_OPTIONS
              INTERFACE_LINK_OPTIONS)
  get_target_property(_entries cudec ${_prop})
  if(_entries)
    foreach(_entry IN LISTS _entries)
      # A PRIVATE link of an imported target lands in the interface list
      # wrapped as $<LINK_ONLY:CUDA::cudart>; strip that wrapper before
      # matching so CUDA::cudart (the CUDA runtime the rule permits) is not
      # a false positive.
      string(REGEX REPLACE "^\\$<LINK_ONLY:(.+)>$" "\\1" _bare "${_entry}")
      if(NOT _bare MATCHES "^CUDA::")
        message(
          FATAL_ERROR
            "cudec carries '${_entry}' in ${_prop} - the library must depend "
            "on nothing beyond the CUDA runtime")
      endif()
    endforeach()
  endif()
endforeach()

# Warnings-as-errors must be wired on the library...
get_target_property(_cudec_opts cudec COMPILE_OPTIONS)
if(NOT _cudec_opts MATCHES "Werror")
  message(FATAL_ERROR "warnings-as-errors is no longer wired on target cudec")
endif()

# ...and LIVE: a translation unit with a known warning must FAIL to compile
# under the exact flag list the targets use, invoked directly (try_compile
# does not forward per-language flags reliably - verified empirically). The
# presence check above cannot see a dead gate (an appended -Wno-error still
# contains the substring); these probes can. Honest scope limit: nvcc's
# front end diagnoses both probes (#177-D), so this proves the
# --Werror=all-warnings seam end to end; the -Xcompiler=-Werror suffix has
# no cheap gcc-only trigger (the front end shadows every candidate tried)
# and stays covered by the substring check only.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/probes/host_warning.cu
     "int main() {\n    int unused_host_variable;\n    return 0;\n}\n")
file(
  WRITE ${CMAKE_CURRENT_BINARY_DIR}/probes/device_warning.cu
  "__global__ void probe() { int unused_device_variable; }\nint main() { return 0; }\n"
)
# Positive control first: a warning-free TU must compile with exit 0 under
# the identical command. Without it, ANY probe breakage (arch renamed in a
# toolkit bump, unwritable dir, rejected flag) exits nonzero and would
# masquerade as "gate live" - the probes would go vacuous exactly on the
# toolchain-drift event they guard against.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/probes/control_clean.cu
     "int main() { return 0; }\n")
execute_process(
  COMMAND
    ${CMAKE_CUDA_COMPILER} ${CUDEC_CUDA_STRICT_FLAGS_RAW} -arch=sm_86 -c
    ${CMAKE_CURRENT_BINARY_DIR}/probes/control_clean.cu -o
    ${CMAKE_CURRENT_BINARY_DIR}/probes/control_clean.o
  RESULT_VARIABLE _control_exit
  OUTPUT_QUIET ERROR_QUIET)
if(NOT _control_exit EQUAL 0)
  message(
    FATAL_ERROR
      "warnings-gate probe invocation is broken (the warning-free control "
      "TU failed to compile) - fix the probe before trusting the gate")
endif()
foreach(_probe host_warning device_warning)
  execute_process(
    COMMAND
      ${CMAKE_CUDA_COMPILER} ${CUDEC_CUDA_STRICT_FLAGS_RAW} -arch=sm_86 -c
      ${CMAKE_CURRENT_BINARY_DIR}/probes/${_probe}.cu -o
      ${CMAKE_CURRENT_BINARY_DIR}/probes/${_probe}.o
    RESULT_VARIABLE _probe_exit
    OUTPUT_QUIET ERROR_QUIET)
  if(_probe_exit EQUAL 0)
    message(
      FATAL_ERROR
        "warnings-as-errors gate is dead: probe '${_probe}' compiled clean "
        "under the strict flags")
  endif()
endforeach()

# Single-source lock (issue #40): the host-side CUDA RAII owners live once, in
# src/cuda_raii.h, and every CUDA host translation unit reuses them instead of
# carrying its own copy. This detector reds the build if the header stops
# defining an owner, or if a host TU drops the include or re-introduces a local
# owner definition. Honest limit (as with the checks above): it catches a
# re-copy under the SAME struct names, not one smuggled in under a fresh name -
# a drift detector, not tamper-proofing.
set(_raii_header ${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda_raii.h)
if(NOT EXISTS ${_raii_header})
  message(FATAL_ERROR "src/cuda_raii.h is missing: the shared CUDA RAII owners "
                      "must be single-sourced there (issue #40)")
endif()
file(READ ${_raii_header} _raii_src)
foreach(_owner DevBuf PinnedBuf StreamOwner EventOwner)
  if(NOT _raii_src MATCHES "struct ${_owner}")
    message(FATAL_ERROR "src/cuda_raii.h no longer defines the shared owner "
                        "'${_owner}' (issue #40)")
  endif()
endforeach()
foreach(_tu frame.cpp stream.cpp)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/../src/${_tu} _tu_src)
  if(NOT _tu_src MATCHES "#include \"cuda_raii.h\"")
    message(FATAL_ERROR "src/${_tu} no longer includes cuda_raii.h: it must "
                        "reuse the shared CUDA RAII owners (issue #40)")
  endif()
  foreach(_owner DevBuf PinnedBuf StreamOwner EventOwner DevPtr)
    if(_tu_src MATCHES "struct ${_owner}")
      message(FATAL_ERROR "src/${_tu} defines a local 'struct ${_owner}': the "
                          "CUDA RAII owners are single-sourced in cuda_raii.h "
                          "(issue #40)")
    endif()
  endforeach()
endforeach()

# Single-source lock for the batch launch limit (issue #39): the bound both
# decode entry points enforce lives once, in src/batch_limits.h, and the two
# callers read it from there. It used to be written twice and held in agreement
# by a comment, which is a fail-open on a fail-closed boundary as soon as one
# copy moves: a chunk_count the streaming pre-check accepts and the batch entry
# then rejects per wave.
#
# The detector is the arithmetic itself. INT32_MAX is what the bound is built
# from, so a second source writing its own bound has to name it, and this
# refuses any source but the header doing so. Honest limit, as with the checks
# above: a copy that computes the same number a different way is not caught -
# a drift detector, not tamper-proofing.
set(_limits_header ${CMAKE_CURRENT_SOURCE_DIR}/../src/batch_limits.h)
if(NOT EXISTS ${_limits_header})
  message(FATAL_ERROR "src/batch_limits.h is missing: the batch launch limit "
                      "must be single-sourced there (issue #39)")
endif()
file(READ ${_limits_header} _limits_src)
if(NOT _limits_src MATCHES "kMaxBatchChunks")
  message(FATAL_ERROR "src/batch_limits.h no longer defines kMaxBatchChunks "
                      "(issue #39)")
endif()
foreach(_tu lz4_decode.cuh stream.cpp)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/../src/${_tu} _tu_src)
  if(NOT _tu_src MATCHES "#include \"batch_limits.h\"")
    message(FATAL_ERROR "src/${_tu} no longer includes batch_limits.h: the "
                        "batch launch limit is single-sourced there "
                        "(issue #39)")
  endif()
  # The comparison itself, not merely a mention of the name: a translation unit
  # can carry the include, keep the using-declaration and still bound
  # chunk_count by some other constant, which is the divergence this lock
  # exists for. Measured, not supposed - a streaming bound raised to twice the
  # canonical one passes every behavioural check available at that magnitude,
  # because no caller can hand the validator an array of tens of billions of
  # entries to walk.
  if(NOT _tu_src MATCHES "chunk_count > kMaxBatchChunks")
    message(FATAL_ERROR "src/${_tu} no longer bounds chunk_count by "
                        "kMaxBatchChunks: both entry points enforce the one "
                        "shared limit (issue #39)")
  endif()
endforeach()
# Every source, not a list somebody has to remember to extend: a new
# translation unit carrying its own bound is exactly the case this is for.
# One function, so the rule can be run against planted sources before it is
# trusted against the tree - the discipline issue #72's scanner established,
# and the reason a tightening is showable rather than asserted.
function(cudec_check_batch_limit_sweep src_dir out_violations)
  set(_violations "")
  file(GLOB_RECURSE _sources RELATIVE ${src_dir} CONFIGURE_DEPENDS
       ${src_dir}/*.c ${src_dir}/*.cc ${src_dir}/*.cpp ${src_dir}/*.cu
       ${src_dir}/*.cuh ${src_dir}/*.h ${src_dir}/*.hpp ${src_dir}/*.inl)
  if(NOT _sources)
    string(APPEND _violations
           "  the sweep found no sources under ${src_dir} - the glob has "
           "stopped matching (issue #39)\n")
  endif()
  foreach(_source IN LISTS _sources)
    if(_source STREQUAL "batch_limits.h")
      continue()
    endif()
    file(READ ${src_dir}/${_source} _source_src)
    # The whole identifier, never a substring. UINT32_MAX carries INT32_MAX
    # inside it, so a width test on an arithmetic operand was refused with a
    # message about the batch launch bound - naming something its author had
    # not done, and leaving "stop writing the standard constant" as the
    # shortest way past a rule whose subject the line was not (issue #333).
    if(_source_src MATCHES "(^|[^A-Za-z_0-9])INT32_MAX([^A-Za-z_0-9]|$)")
      string(APPEND _violations
             "  ${_source} builds its own bound from INT32_MAX - the batch "
             "launch limit is cudec_detail::kMaxBatchChunks in "
             "src/batch_limits.h (issue #39)\n")
    endif()
  endforeach()
  set(${out_violations} "${_violations}" PARENT_SCOPE)
endfunction()

# Both directions, planted first. The accepted probe must come back silent
# and the refused one is matched against the TEXT the rule emits rather than
# against a non-empty result: a rule narrowed without the second probe is a
# rule nobody can show still bites.
set(_batch_limit_probe_root ${CMAKE_CURRENT_BINARY_DIR}/batch-limit-probes)
file(REMOVE_RECURSE ${_batch_limit_probe_root})
file(WRITE ${_batch_limit_probe_root}/unsigned_constant/probe.cuh
     "static bool fits32(uint64_t n) { return n <= UINT32_MAX; }\n")
file(WRITE ${_batch_limit_probe_root}/own_bound/probe.cuh
     "static const size_t kChunks = (size_t)INT32_MAX * kWarpSize;\n")
set(_batch_limit_probes
    "unsigned_constant|"
    "own_bound|probe.cuh builds its own bound from INT32_MAX")
foreach(_entry IN LISTS _batch_limit_probes)
  string(REGEX REPLACE "^([^|]+)\\|(.*)$" "\\1" _probe "${_entry}")
  string(REGEX REPLACE "^([^|]+)\\|(.*)$" "\\2" _expected "${_entry}")
  cudec_check_batch_limit_sweep(${_batch_limit_probe_root}/${_probe}
                                _probe_result)
  if(_expected STREQUAL "")
    # Checked here rather than through string(FIND): an empty needle is found
    # at position 0 in every string, so the branch below would assert nothing.
    if(_probe_result)
      message(FATAL_ERROR "the batch-limit sweep refuses a compliant probe: "
                          "'${_probe}' must come back silent (issue #333). It "
                          "reported:\n${_probe_result}")
    endif()
  else()
    string(FIND "${_probe_result}" "${_expected}" _found_at)
    if(_found_at EQUAL -1)
      message(FATAL_ERROR "the batch-limit sweep is dead: probe '${_probe}' "
                          "did not report \"${_expected}\" (issue #333). It "
                          "reported:\n${_probe_result}")
    endif()
  endif()
endforeach()

# And now against the tree the rule is for.
cudec_check_batch_limit_sweep(${CMAKE_CURRENT_SOURCE_DIR}/../src
                              _batch_limit_violations)
if(_batch_limit_violations)
  message(FATAL_ERROR "a source under src/ builds its own batch launch bound "
                      "(issue #39):\n${_batch_limit_violations}")
endif()

# Reads a source and hands back its lines with C comments blanked in place,
# one list element per source line, so a rule can quote the line number a
# reader will open. Comments go first for the reason issue #72's scanner
# strips them: prose beside a rule must neither trip it nor mask it, and this
# header's rejects are documented in prose sitting right on top of them.
#
# FOUR characters CMake's list syntax eats are protected across the split and
# left protected in the result. A ';' would separate elements and a trailing
# '\' would escape the separator, which is what once numbered every line under
# a continuation short (issue #251). A '[' opens a bracket that swallows every
# separator until its ']', so ONE unbalanced bracket in a source - `src[from`
# wrapped across a line, or a lone one in a comment - joins the whole rest of
# the file into a single element and every line below it stops being a line.
# That is not hypothetical: it is measured on this very header, and issue #294
# carries it for the scanner that has the same reader.
#
# The placeholders are control bytes, and a source that carries one of those
# itself is refused rather than scanned - restoring it afterwards would be a
# way to smuggle a separator in.
function(cudec_code_lines path out_lines)
  string(ASCII 1 _tok_backslash)
  string(ASCII 2 _tok_semicolon)
  string(ASCII 3 _tok_open)
  string(ASCII 4 _tok_close)
  file(READ "${path}" _raw)
  if(_raw MATCHES "${_tok_backslash}|${_tok_semicolon}|${_tok_open}|${_tok_close}")
    message(FATAL_ERROR "${path} carries a control byte (0x01 to 0x04) and "
                        "cannot be read as text by this scan (issue #173)")
  endif()
  string(REPLACE "\\" "${_tok_backslash}" _raw "${_raw}")
  string(REPLACE ";" "${_tok_semicolon}" _raw "${_raw}")
  string(REPLACE "[" "${_tok_open}" _raw "${_raw}")
  string(REPLACE "]" "${_tok_close}" _raw "${_raw}")
  string(REPLACE "\r\n" "\n" _raw "${_raw}")
  string(REPLACE "\r" "\n" _raw "${_raw}")
  string(REGEX REPLACE "\n$" "" _raw "${_raw}")
  # What a whole read must produce, counted before the split rather than
  # inferred from it: one line per remaining newline, plus the last line,
  # which the strip above left without one.
  string(REGEX MATCHALL "\n" _newlines "${_raw}")
  list(LENGTH _newlines _expected)
  math(EXPR _expected "${_expected} + 1")
  if(_raw STREQUAL "")
    message(FATAL_ERROR "${path} is empty: a source this scan is asked to "
                        "read whole has at least one line (issue #173)")
  endif()
  string(REPLACE "\n" ";" _raw_lines "${_raw}")
  set(_out "")
  set(_in_comment FALSE)
  foreach(_line IN LISTS _raw_lines)
    if(_in_comment)
      if(_line MATCHES "\\*/")
        string(REGEX REPLACE "^.*\\*/" " " _line "${_line}")
        set(_in_comment FALSE)
      else()
        set(_line " ")
      endif()
    endif()
    string(REGEX REPLACE "//.*$" " " _line "${_line}")
    if(_line MATCHES "/\\*")
      string(REGEX REPLACE "/\\*.*$" " " _head "${_line}")
      if(_line MATCHES "/\\*.*\\*/")
        string(REGEX REPLACE "^.*\\*/" " " _tail "${_line}")
        set(_line "${_head} ${_tail}")
      else()
        set(_line "${_head}")
        set(_in_comment TRUE)
      endif()
    endif()
    # An empty element would vanish from the list and shift every line number
    # below it; a space holds the slot.
    if(_line STREQUAL "")
      set(_line " ")
    endif()
    list(APPEND _out "${_line}")
  endforeach()
  # A partial read must not look like a clean one. The protections above are
  # what make the counts agree, so this is the check that says they still do -
  # and the one that will red if a fifth character with list meaning turns up.
  list(LENGTH _out _produced)
  if(NOT _produced EQUAL _expected)
    message(FATAL_ERROR
            "${path} was read as ${_produced} lines and carries ${_expected}: "
            "this scan cannot read it whole, and a partial read reports the "
            "same silence as a clean one (issue #173)")
  endif()
  set(${out_lines} "${_out}" PARENT_SCOPE)
endfunction()

# ---- The Snappy parser core, locked (issue #173). Two rules over one source
# set, applied by ONE function so they can be run against planted probes
# before they are trusted against the tree - the discipline issue #72's
# scanner established, applied here because these two rules are the only
# thing standing between a second copy of the ladder and nobody noticing.
#
# Honest scope, the same as every check above: these are drift detectors
# under the KNOWN SPELLINGS, not tamper-proofing. A ladder re-copied under
# fresh type names, or a refusal spelled through a local variable instead of
# a `return`, is out of reach of a text scan and is what review is for.
#
# Rule 1, single source. src/snappy_block.h exists and defines the parser
# types; no other source under src/ defines them, and a source that names
# SnappyParser includes the header rather than restating the ladder. This is
# the cuda_raii.h lock (issue #40) applied to the format core: the M3 kernel
# instantiates this parser, and a kernel-local copy of a validation ladder is
# a fail-open the moment the two stop agreeing.
#
# Rule 2, the branch count. Every refusal in the header returns through
# SnappyRefuse naming a branch the enumeration declares, and each declared
# branch is named by EXACTLY ONE site. That makes the ladder's branch set
# derived from the code rather than restated beside it, and it is the static
# half of a pair: tests/snappy_parser_twin.cpp holds the other half at run
# time, requiring a negative to reach every declared branch. Adding a rung
# therefore costs a new enumerator here and a new negative there, and a rung
# that quietly reuses a neighbour's enumerator to avoid both is what the
# exactly-one count refuses.
function(cudec_check_snappy_locks src_dir out_violations)
  set(_violations "")
  set(_header ${src_dir}/snappy_block.h)
  if(NOT EXISTS ${_header})
    string(APPEND _violations
           "  ${src_dir}/snappy_block.h is missing: the Snappy validation "
           "ladder must be single-sourced there (issue #173)\n")
    set(${out_violations} "${_violations}" PARENT_SCOPE)
    return()
  endif()

  # Comments are stripped before any rule reads a line, so the prose above a
  # refusal can neither trip a rule nor mask one - and the line numbers stay
  # exact, because the stripping replaces comment text in place instead of
  # dropping lines.
  cudec_code_lines(${_header} _header_lines)
  foreach(_type SnappyParser SnappyElement)
    set(_defines FALSE)
    foreach(_line IN LISTS _header_lines)
      if(_line MATCHES "struct ${_type}")
        set(_defines TRUE)
      endif()
    endforeach()
    if(NOT _defines)
      string(APPEND _violations
             "  snappy_block.h no longer defines 'struct ${_type}': the "
             "Snappy parser core is single-sourced there (issue #173)\n")
    endif()
  endforeach()

  # Every other source under src/ - a glob rather than a list somebody has to
  # remember to extend, for the same reason the batch-limit sweep is one.
  file(GLOB_RECURSE _sources RELATIVE ${src_dir} CONFIGURE_DEPENDS
       ${src_dir}/*.c ${src_dir}/*.cc ${src_dir}/*.cpp ${src_dir}/*.cu
       ${src_dir}/*.cuh ${src_dir}/*.h ${src_dir}/*.hpp ${src_dir}/*.inl)
  foreach(_source IN LISTS _sources)
    if(_source STREQUAL "snappy_block.h")
      continue()
    endif()
    cudec_code_lines(${src_dir}/${_source} _source_lines)
    set(_names_parser FALSE)
    set(_includes_header FALSE)
    foreach(_line IN LISTS _source_lines)
      foreach(_type SnappyParser SnappyElement)
        if(_line MATCHES "struct ${_type}[ \t]*[{:]")
          string(APPEND _violations
                 "  ${_source} defines a local 'struct ${_type}': the Snappy "
                 "parser core is single-sourced in snappy_block.h "
                 "(issue #173)\n")
        endif()
      endforeach()
      if(_line MATCHES "SnappyParser|SnappyElement|SnappyDeclaredLength")
        set(_names_parser TRUE)
      endif()
      if(_line MATCHES "#include[ \t]+\"snappy_block.h\"")
        set(_includes_header TRUE)
      endif()
    endforeach()
    if(_names_parser AND NOT _includes_header)
      string(APPEND _violations
             "  ${_source} names the Snappy parser without including "
             "snappy_block.h: it must reuse the single-sourced ladder rather "
             "than restate it (issue #173)\n")
    endif()
  endforeach()

  # Rule 2. The declared branches first, read out of the enumeration between
  # its None and Count sentinels; then the refusal sites.
  set(_declared "")
  set(_in_enum FALSE)
  foreach(_line IN LISTS _header_lines)
    if(_line MATCHES "enum SnappyReject[ \t]*{")
      set(_in_enum TRUE)
      continue()
    endif()
    if(_in_enum)
      if(_line MATCHES "kSnappyRejectCount")
        set(_in_enum FALSE)
        continue()
      endif()
      if(_line MATCHES "^[ \t]*(kSnappyReject[A-Za-z0-9_]*)")
        if(NOT CMAKE_MATCH_1 STREQUAL "kSnappyRejectNone")
          list(APPEND _declared ${CMAKE_MATCH_1})
        endif()
      endif()
    endif()
  endforeach()
  if(NOT _declared)
    string(APPEND _violations
           "  snappy_block.h declares no reject ladder: 'enum SnappyReject' "
           "must enumerate one branch per reject reason (issue #173)\n")
  endif()

  set(_named "")
  set(_lineno 0)
  foreach(_line IN LISTS _header_lines)
    math(EXPR _lineno "${_lineno} + 1")
    # The choke point, refused first: a refusal that never reaches
    # SnappyRefuse is invisible to every count below, so it is the one shape
    # that could add a rung and leave both halves of this pair green.
    if(_line MATCHES "return[ \t]+CUDEC_ERR_")
      string(APPEND _violations
             "  snappy_block.h:${_lineno}: refuses without going through "
             "SnappyRefuse - every refusal names its ladder branch "
             "(issue #173)\n")
    endif()
    if(_line MATCHES "return[ \t]+SnappyRefuse\\(")
      # The branch has to be on the line the call opens on. A wrapped first
      # argument would hide the name from this scan exactly as a wrapped mask
      # hides from the collective rule.
      if(_line MATCHES "return[ \t]+SnappyRefuse\\((kSnappyReject[A-Za-z0-9_]*)")
        list(APPEND _named ${CMAKE_MATCH_1})
      else()
        string(APPEND _violations
               "  snappy_block.h:${_lineno}: a SnappyRefuse call with no "
               "ladder branch on its own line (issue #173)\n")
      endif()
    endif()
  endforeach()

  foreach(_branch IN LISTS _declared)
    set(_count 0)
    foreach(_site IN LISTS _named)
      if(_site STREQUAL _branch)
        math(EXPR _count "${_count} + 1")
      endif()
    endforeach()
    if(_count EQUAL 0)
      string(APPEND _violations
             "  ladder branch '${_branch}' is declared and no refusal site "
             "names it (issue #173)\n")
    elseif(NOT _count EQUAL 1)
      string(APPEND _violations
             "  ladder branch '${_branch}' is named by ${_count} refusal "
             "sites: a new rung takes a new branch, so that the twin's "
             "negative for it cannot be satisfied by its neighbour "
             "(issue #173)\n")
    endif()
  endforeach()
  foreach(_site IN LISTS _named)
    if(NOT _site IN_LIST _declared)
      string(APPEND _violations
             "  a refusal site names '${_site}', which 'enum SnappyReject' "
             "does not declare (issue #173)\n")
    endif()
  endforeach()

  set(${out_violations} "${_violations}" PARENT_SCOPE)
endfunction()

# The rules above are run against planted source trees before they are
# trusted against the real one, and every violating probe is matched against
# the TEXT its own rule emits rather than against a non-empty result - so one
# rule cannot cover for another that has quietly stopped matching. The probe
# headers are read, never compiled, which is why they are this terse.
set(_snappy_probe_root ${CMAKE_CURRENT_BINARY_DIR}/snappy-lock-probes)
file(REMOVE_RECURSE ${_snappy_probe_root})

# A compliant header: two rungs, two sites, one choke point. Each block is
# ONE string rather than a list of lines, because the variants below are cut
# from it with string(REPLACE) and a list would come back joined on
# semicolons - which is a stray separator written into a C header.
#
# The head is 21 lines, so an injected rung starts at line 22, which is the
# number the two line-scoped rules are asserted on below.
set(_probe_header_head "#ifndef PROBE_SNAPPY_H
#define PROBE_SNAPPY_H
namespace cudec_detail {
enum SnappyReject {
    kSnappyRejectNone = 0,
    kSnappyRejectAlpha,
    kSnappyRejectBeta,
    kSnappyRejectCount
};
inline cudec_status SnappyRefuse(SnappyReject branch,
                                 cudec_status status,
                                 SnappyReject* out) {
    return status;
}
struct SnappyElement {
    uint64_t from;
};
struct SnappyParser {
    SnappyReject reject;
    cudec_status Next(bool first) {
        if (first) {
")
set(_probe_header_tail "        }
        return SnappyRefuse(kSnappyRejectBeta, CUDEC_ERR_CORRUPT_INPUT,
                            &reject);
    }
};
}  // namespace cudec_detail
#endif
")
set(_probe_alpha_site "            return SnappyRefuse(kSnappyRejectAlpha,
                                CUDEC_ERR_CORRUPT_INPUT, &reject);
")

file(WRITE ${_snappy_probe_root}/clean/snappy_block.h ${_probe_header_head}
     ${_probe_alpha_site} ${_probe_header_tail})
# The tree with no core at all.
file(WRITE ${_snappy_probe_root}/missing_header/other.cu "void probe() {}\n")
# A header that stopped defining the parser type it exists to own.
string(REPLACE "struct SnappyParser {" "struct SomethingElse {"
               _probe_no_parser "${_probe_header_head}")
file(WRITE ${_snappy_probe_root}/no_parser/snappy_block.h ${_probe_no_parser}
     ${_probe_alpha_site} ${_probe_header_tail})
# A second source carrying its own copy of the parser type.
file(WRITE ${_snappy_probe_root}/local_copy/snappy_block.h
     ${_probe_header_head} ${_probe_alpha_site} ${_probe_header_tail})
file(WRITE ${_snappy_probe_root}/local_copy/kernel.cu
     "#include \"snappy_block.h\"\n" "struct SnappyParser {\n"
     "    uint64_t src_pos;\n" "};\n")
# A source driving the parser without reaching for the single source.
file(WRITE ${_snappy_probe_root}/no_include/snappy_block.h
     ${_probe_header_head} ${_probe_alpha_site} ${_probe_header_tail})
file(WRITE ${_snappy_probe_root}/no_include/kernel.cu
     "void probe(cudec_detail::SnappyParser* p) {\n" "    (void)p;\n" "}\n")
# The same source with the include: the permitted shape, which must stay
# silent, or the rule above would refuse the kernel this lock exists to admit.
file(WRITE ${_snappy_probe_root}/with_include/snappy_block.h
     ${_probe_header_head} ${_probe_alpha_site} ${_probe_header_tail})
file(WRITE ${_snappy_probe_root}/with_include/kernel.cu
     "#include \"snappy_block.h\"\n"
     "void probe(cudec_detail::SnappyParser* p) {\n" "    (void)p;\n" "}\n")
# A rung that refuses without naming itself.
file(WRITE ${_snappy_probe_root}/bare_return/snappy_block.h
     ${_probe_header_head} "            return CUDEC_ERR_CORRUPT_INPUT;\n"
     ${_probe_header_tail})
# A rung whose branch is wrapped onto the next line, out of a line-scoped
# scan's sight.
file(WRITE ${_snappy_probe_root}/wrapped_refuse/snappy_block.h
     ${_probe_header_head} "            return SnappyRefuse(\n"
     "                kSnappyRejectAlpha, CUDEC_ERR_CORRUPT_INPUT, &reject);\n"
     ${_probe_header_tail})
# A branch nothing refuses through: the shape a deleted rung leaves behind.
string(REPLACE "    kSnappyRejectBeta,
" "    kSnappyRejectBeta,
    kSnappyRejectGamma,
" _probe_extra_branch "${_probe_header_head}")
file(WRITE ${_snappy_probe_root}/branch_unused/snappy_block.h
     ${_probe_extra_branch} ${_probe_alpha_site} ${_probe_header_tail})
# A site naming a branch the enumeration does not carry: the shape a rung
# added without an enumerator leaves behind.
file(WRITE ${_snappy_probe_root}/site_undeclared/snappy_block.h
     ${_probe_header_head}
     "            return SnappyRefuse(kSnappyRejectGamma,\n"
     "                                CUDEC_ERR_CORRUPT_INPUT, &reject);\n"
     ${_probe_header_tail})
# Two rungs behind one branch - a new rung borrowing its neighbour's
# enumerator, which would let the twin's negative for the old rung stand in
# for the new one.
file(WRITE ${_snappy_probe_root}/branch_shared/snappy_block.h
     ${_probe_header_head} ${_probe_alpha_site}
     "        if (!first) {\n"
     "            return SnappyRefuse(kSnappyRejectAlpha,\n"
     "                                CUDEC_ERR_CORRUPT_INPUT, &reject);\n"
     "        }\n" ${_probe_header_tail})
# No enumeration at all.
string(REGEX REPLACE "enum SnappyReject" "enum SomethingElse" _probe_no_enum
                     "${_probe_header_head}")
file(WRITE ${_snappy_probe_root}/no_enum/snappy_block.h ${_probe_no_enum}
     ${_probe_alpha_site} ${_probe_header_tail})
# An unbalanced bracket above a violation - the shape that reads only the top
# of a file (issue #294). The rule below the bracket must still report, and at
# the line it is on.
file(WRITE ${_snappy_probe_root}/bracket_above/snappy_block.h
     ${_probe_header_head}
     "            /* Reading from src[from .. as the real header does. */\n"
     "            return CUDEC_ERR_CORRUPT_INPUT;\n" ${_probe_header_tail})
# Prose is not code. This header is compliant and its comments spell out both
# a bare refusal and a second parser definition; a scan that read comments
# would refuse the file that documents itself best.
file(WRITE ${_snappy_probe_root}/prose/snappy_block.h ${_probe_header_head}
     "            /* Not: return CUDEC_ERR_CORRUPT_INPUT; and not a second\n"
     "             * struct SnappyParser { either. */\n" ${_probe_alpha_site}
     ${_probe_header_tail})

set(_snappy_lock_probes
    "clean|"
    "with_include|"
    "prose|"
    "missing_header|snappy_block.h is missing"
    "no_parser|no longer defines 'struct SnappyParser'"
    "local_copy|kernel.cu defines a local 'struct SnappyParser'"
    "no_include|kernel.cu names the Snappy parser without including"
    "bare_return|snappy_block.h:22: refuses without going through SnappyRefuse"
    "wrapped_refuse|snappy_block.h:22: a SnappyRefuse call with no ladder branch"
    "branch_unused|ladder branch 'kSnappyRejectGamma' is declared and no refusal"
    "site_undeclared|a refusal site names 'kSnappyRejectGamma', which"
    "branch_shared|ladder branch 'kSnappyRejectAlpha' is named by 2 refusal"
    "no_enum|declares no reject ladder"
    "bracket_above|snappy_block.h:23: refuses without going through")
foreach(_entry IN LISTS _snappy_lock_probes)
  string(REGEX REPLACE "^([^|]+)\\|(.*)$" "\\1" _probe "${_entry}")
  string(REGEX REPLACE "^([^|]+)\\|(.*)$" "\\2" _expected "${_entry}")
  cudec_check_snappy_locks(${_snappy_probe_root}/${_probe} _probe_result)
  if(_expected STREQUAL "")
    # Checked here rather than through string(FIND): an empty needle is found
    # at position 0 in every string, so the branch below would assert nothing.
    if(_probe_result)
      message(FATAL_ERROR "the Snappy lock refuses a compliant probe: "
                          "'${_probe}' must come back silent (issue #173). It "
                          "reported:\n${_probe_result}")
    endif()
  else()
    string(FIND "${_probe_result}" "${_expected}" _found_at)
    if(_found_at EQUAL -1)
      message(FATAL_ERROR "the Snappy lock is dead for one rule: probe "
                          "'${_probe}' did not report \"${_expected}\" "
                          "(issue #173). It reported:\n${_probe_result}")
    endif()
  endif()
endforeach()

# And now against the tree the rules are for.
cudec_check_snappy_locks(${CMAKE_CURRENT_SOURCE_DIR}/../src _snappy_violations)
if(_snappy_violations)
  message(FATAL_ERROR "the Snappy parser core is not single-sourced, or its "
                      "reject ladder and its refusal sites have drifted "
                      "apart (issue #173):\n${_snappy_violations}")
endif()

# ---- Termination, warp-collective integrity, and determinism as structural
# rules (issue #72). Same honest scope as the checks above: drift detectors
# under the known spellings, not tamper-proofing. Every rule below is applied
# by ONE scanning function, which is then run over inline probe files that
# must and must not trip it - and each violating probe is matched against the
# TEXT its rule emits, not merely against a non-empty result, so one rule
# cannot cover for another that has quietly stopped matching.

# Scans one source and returns every structural violation it finds. Comments
# are stripped first, so prose ("never double-frees", "decode loops
# terminate") can neither trip a rule nor mask one. `check_loops` selects the
# termination rule, which is scoped to the decode path rather than to every
# helper.
#
# Rules:
#  - loops: a counted `for (init; cond; incr)` is accepted on its visible
#    induction variable; every other loop form (`while`, `do {`, a `for`
#    header with an empty condition) must name an explicit decrementing
#    `fuel` counter on its own line, the line above it, or the three below,
#    because a text scan cannot verify a monotone-cursor argument and a
#    non-terminating decode hangs the device (nvCOMP 5.3's release notes
#    record shipping exactly that on malformed Snappy input);
#  - warp collectives: the legacy non-_sync intrinsics are banned outright
#    (since Volta's independent thread scheduling they cannot express which
#    lanes participate at all), and a collective's mask must be a full-warp
#    constant or __activemask(), never a value named after a parsed bitstream
#    field - corrupting collective participation metadata is a published
#    attack on CUDA kernels that leaves control flow untouched. A collective
#    wrapped across lines is a violation in itself: the scan is line-scoped,
#    so a wrapped call would hide its mask from every rule here;
#  - determinism: no floating-point type and no floating-point atomic, which
#    is what makes the gpu_to_gpu level in docs/DETERMINISM.md hold by
#    construction rather than by measurement;
#  - wave width: no bare 32 or 64 outside the one line that binds it to a
#    name. The maintainer's hardware is NVIDIA, so a hardcoded 32 left in a
#    lane-stride expression is invisible to every test that can be run here
#    and wrong on every wave64 device (issue #211). A width must arrive as the
#    `WaveSize` template parameter or as a constant named after it, and the
#    only place the digits themselves may appear is a `constexpr` definition
#    or a `#define` whose entire initializer they are. The rule already caught
#    one: the host copy of the batch limit in src/stream.cpp multiplied
#    INT32_MAX by a bare 32 while its device twin used kWarpSize.
#
# Limits, stated as the neighbouring checks state theirs: constructs hidden
# inside string literals are not stripped; a mask computed into a variable on
# an earlier line is judged by its name at the call site; a `fuel` mention on
# the line above an unrelated loop satisfies the loop rule; and a loop or
# intrinsic introduced under a fresh spelling is code review's job.
#
# The width rule cannot tell a lane width from any other 32 or 64, and it does
# not try: every one of them must be named, whatever it counts, and the cost of
# a wrong guess is one `constexpr` line rather than a wrong answer on hardware
# nobody here owns. What it does NOT catch is a width written any other way -
# `31 + 1`, `0x20`, `1 << 5`, a `sizeof` that happens to be 32, or a name that
# lies about what it holds. It is a drift detector under the spelling somebody
# will actually type, like every rule above it.
#
# Backslash continuations split the rules in two, and the split is where each
# rule's reason points (issue #251). The collective, legacy-intrinsic and
# floating-point rules run on the SOURCE line, because a construct broken
# across lines hides the rest of itself from a line-scoped rule whether a
# macro's backslashes join it back together or not - so a collective wrapped
# inside a macro is reported exactly like one wrapped outside it, on the
# source line it wrapped at. The loop rule instead judges a continued
# construct whole, joined the way the preprocessor joins it, because that is
# what its idiom forgiveness rests on. Both report the source line, which the
# reader can open; the join's own separators are subtracted back out of the
# semicolon count so that a macro-hidden `for` header is not read as a
# counted one. A continuation left open at end of file is rejected rather
# than dropped.
#
# The `do { ... } while (0)` idiom is forgiven only where it is joined into one
# element, which is the macro spelling and the only one the tree uses. Two
# things follow, and both are the price of a line-scoped scan. A written-out
# `do { ... } while (0);` is reported, because its tail is a separate element
# that clears nothing - wrap it in a macro or name a `fuel` cap. And a macro
# carrying a real loop AND the idiom in the same element is forgiven whole,
# since the element is judged by what it contains rather than by structure the
# scan does not parse.
# Does this `for` header carry a RANGE-FOR's colon, or merely a colon?
#
# The opaque-`for` rule exempts a range-for because its bound is the
# container's size: loop-invariant and visible on the spot, exactly like the
# counted form's. A colon that is something else carries none of that, so
# testing for "a colon" granted the exemption on a coincidence (issue #250).
# A range-for's colon sits at the header's TOP parenthesis level, which is a
# property of a range-for rather than of the character, so that is what is
# tested. A ternary inside a macro argument - `for (EACH(x, w ? a : b))` -
# puts its colon a level down and is caught. `::` goes first: a qualified
# name is not a range-for either (issue #123).
#
# The residual, and it is inherited rather than introduced: this scan strips
# no string literals anywhere, so a colon inside one at the top level -
# `for (EACH(x) "a:b")` - still exempts. Stripping literals is a pass over
# every rule in this function, not a change to this one, and the function's
# "Limits" paragraph already names unstripped literals as a known blind spot.
function(cudec_for_colon_at_top_level text out_result)
  set(${out_result} FALSE PARENT_SCOPE)
  string(REPLACE "::" "" _text "${text}")
  # From the header's own '(' onward; what precedes it cannot hold the colon.
  string(REGEX MATCH "for[ \t]*\\(.*" _text "${_text}")
  string(LENGTH "${_text}" _length)
  if(_length EQUAL 0)
    return()
  endif()
  set(_depth 0)
  math(EXPR _last "${_length} - 1")
  foreach(_index RANGE ${_last})
    string(SUBSTRING "${_text}" ${_index} 1 _char)
    if(_char STREQUAL "(")
      math(EXPR _depth "${_depth} + 1")
    elseif(_char STREQUAL ")")
      math(EXPR _depth "${_depth} - 1")
      if(_depth LESS_EQUAL 0)
        # The header's parenthesis closed; the rest of the line is the body.
        return()
      endif()
    elseif(_char STREQUAL ":" AND _depth EQUAL 1)
      set(${out_result} TRUE PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

function(cudec_scan_source path check_loops out_violations)
  set(_violations "")
  set(_in_comment FALSE)
  set(_pending_reason "")
  set(_pending_line 0)
  set(_pending_left 0)
  set(_previous "")
  set(_for_open 0)
  set(_for_semis 0)
  set(_for_start 0)
  set(_for_lines 0)
  set(_for_text "")
  set(_lineno 0)
  set(_element "")
  set(_element_start 0)
  set(_element_joins 0)
  set(_element_open FALSE)
  # Read the bytes and do the splitting here rather than with
  # file(STRINGS): a trailing backslash escapes the separator file(STRINGS)
  # puts between elements, so a continued construct arrives as ONE element
  # and every line below it is numbered short by the number of continuation
  # lines above it (issue #251). Splitting here keeps the source line number
  # exact, and the join the loop rule needs is rebuilt below where the
  # element it belongs to is known.
  #
  # The FOUR characters CMake's list syntax eats are protected across the
  # split: a ';' separates elements, a '\' escapes that separator, and a '['
  # opens a bracket that swallows every separator until its ']'. One
  # unbalanced bracket - `src[from` wrapped across a line, or a lone one in a
  # comment - therefore joins the whole rest of the file into a single
  # element, and every line below it stops reaching a rule. Measured on
  # src/snappy_block.h, where 290 of 440 lines went unread (issue #294); the
  # reader in cudec_code_lines carries the same protection for the same
  # reason. The placeholders are control bytes, which no source in this tree
  # carries and which therefore cannot be produced by the source being
  # scanned.
  string(ASCII 1 _tok_backslash)
  string(ASCII 2 _tok_semicolon)
  string(ASCII 3 _tok_open)
  string(ASCII 4 _tok_close)
  file(READ "${path}" _raw)
  # "No source carries one" is the assumption this reader rests on, so it is
  # stated to the file rather than trusted: a source that does carry one would
  # have it restored below as a backslash or as a separator, which is a way to
  # smuggle either past the split. Refuse the file instead of scanning it.
  if(_raw MATCHES
     "${_tok_backslash}|${_tok_semicolon}|${_tok_open}|${_tok_close}")
    string(APPEND _violations
           "  ${path}:1: a control byte (0x01 to 0x04) in a source read as "
           "text - this scan cannot tell one from its own line markers, so "
           "the file is refused rather than scanned\n")
  endif()
  string(REPLACE "\\" "${_tok_backslash}" _raw "${_raw}")
  string(REPLACE ";" "${_tok_semicolon}" _raw "${_raw}")
  string(REPLACE "[" "${_tok_open}" _raw "${_raw}")
  string(REPLACE "]" "${_tok_close}" _raw "${_raw}")
  string(REPLACE "\r\n" "\n" _raw "${_raw}")
  string(REPLACE "\r" "\n" _raw "${_raw}")
  # A file's final newline terminates its last line rather than starting an
  # empty one; left in, it would add a line the file does not have.
  string(REGEX REPLACE "\n$" "" _raw "${_raw}")
  # What a whole read must produce, counted from the bytes before the split
  # rather than inferred from what the split returned: one line per remaining
  # newline, plus the last line, which the strip above left without one. The
  # comparison is at the bottom of the loop.
  string(REGEX MATCHALL "\n" _newlines "${_raw}")
  list(LENGTH _newlines _expected_lines)
  math(EXPR _expected_lines "${_expected_lines} + 1")
  string(REPLACE "\n" ";" _lines "${_raw}")
  foreach(_src_line IN LISTS _lines)
    math(EXPR _lineno "${_lineno} + 1")
    string(REPLACE "${_tok_semicolon}" ";" _src_line "${_src_line}")
    string(REPLACE "${_tok_backslash}" "\\" _src_line "${_src_line}")
    string(REPLACE "${_tok_open}" "[" _src_line "${_src_line}")
    string(REPLACE "${_tok_close}" "]" _src_line "${_src_line}")
    set(_line "${_src_line}")
    if(_in_comment)
      if(_line MATCHES "\\*/")
        string(REGEX REPLACE "^.*\\*/" " " _line "${_line}")
        set(_in_comment FALSE)
      else()
        set(_line "")
      endif()
    endif()
    string(REGEX REPLACE "//.*$" " " _line "${_line}")
    if(_line MATCHES "/\\*")
      string(REGEX REPLACE "/\\*.*$" " " _head "${_line}")
      if(_line MATCHES "/\\*.*\\*/")
        string(REGEX REPLACE "^.*\\*/" " " _tail "${_line}")
        set(_line "${_head} ${_tail}")
      else()
        set(_line "${_head}")
        set(_in_comment TRUE)
      endif()
    endif()
    set(_code " ${_line} ")

    # The loop rule judges a backslash-continued construct WHOLE, which is
    # what its do-idiom forgiveness rests on, so the join file(STRINGS) used
    # to perform is rebuilt here: the trailing backslash goes and a ';' takes
    # the newline's place, byte for byte what the old reader produced. What is
    # kept, and what the old reader had no way to keep, is the source line the
    # element started on and how many separators the join inserted.
    if(NOT _element_open)
      set(_element "")
      set(_element_start ${_lineno})
      set(_element_joins 0)
    endif()
    if(_src_line MATCHES "\\\\$")
      string(REGEX REPLACE "\\\\$" "" _joined "${_line}")
      string(APPEND _element "${_joined};")
      math(EXPR _element_joins "${_element_joins} + 1")
      set(_element_open TRUE)
      set(_element_code "")
    else()
      set(_element_open FALSE)
      set(_element_code " ${_element}${_line} ")
    endif()

    if(check_loops AND NOT _element_open)
      # Parenthesis balance and semicolon count of this element, used to
      # follow a loop header that wrapped at the 80-column limit instead of
      # misreading its first line as a header without a condition. The join's
      # own separators come back out of the semicolon count: left in, two of
      # them make a macro-hidden `for` header look like a counted one and the
      # rule reads a fail-open as a pass.
      string(LENGTH "${_element_code}" _len_all)
      string(REPLACE "(" "" _stripped "${_element_code}")
      string(LENGTH "${_stripped}" _len_no_open)
      string(REPLACE ")" "" _stripped "${_element_code}")
      string(LENGTH "${_stripped}" _len_no_close)
      string(REPLACE ";" "" _stripped "${_element_code}")
      string(LENGTH "${_stripped}" _len_no_semi)
      math(EXPR _opens "${_len_all} - ${_len_no_open}")
      math(EXPR _closes "${_len_all} - ${_len_no_close}")
      math(EXPR _balance "${_opens} - ${_closes}")
      math(EXPR _line_semis
           "${_len_all} - ${_len_no_semi} - ${_element_joins}")
      if(_pending_left GREATER 0)
        if(_element_code MATCHES "fuel")
          set(_pending_left 0)
        else()
          math(EXPR _pending_left "${_pending_left} - 1")
          if(_pending_left EQUAL 0)
            string(APPEND _violations
                   "  ${path}:${_pending_line}: ${_pending_reason} with no "
                   "explicit decrementing 'fuel' cap\n")
          endif()
        endif()
      endif()
      set(_form "")
      set(_form_line ${_element_start})
      if(_for_open GREATER 0)
        # Continuation of a wrapped `for` header: keep counting until the
        # parenthesis closes, then judge the whole header at once.
        math(EXPR _for_semis "${_for_semis} + ${_line_semis}")
        math(EXPR _for_open "${_for_open} + ${_balance}")
        math(EXPR _for_lines "${_for_lines} + 1")
        string(APPEND _for_text " ${_element_code}")
        set(_form_line ${_for_start})
        if(_for_open LESS_EQUAL 0)
          set(_for_open 0)
          cudec_for_colon_at_top_level("${_for_text}" _is_range_for)
          if(_for_semis LESS 2 AND NOT _is_range_for)
            set(_form "a 'for' header without a visible condition and increment")
          endif()
        elseif(_for_lines GREATER 8)
          # Bail out rather than count forever. The scan does not strip string
          # literals, so one unbalanced '(' inside a literal would otherwise
          # leave this tracker open and silently disable the whole loop rule
          # for the rest of the file - a fail-open in the one rule that exists
          # to be fail-closed.
          set(_for_open 0)
          string(APPEND _violations
                 "  ${path}:${_for_start}: a 'for' header whose closing "
                 "parenthesis was not found within 8 lines - the scan cannot "
                 "judge it, so it is rejected rather than skipped\n")
        endif()
      elseif(_element_code MATCHES "[^A-Za-z_0-9]do[ \t]*{")
        # Ahead of the `} while` tail branch on purpose. A backslash-continued
        # macro reaches this rule as ONE element, joined above the way the
        # preprocessor joins it, so a macro's
        # `do {` and its own tail arrive together. Judged after the tail branch
        # they matched it first, which forgave every `do` inside every macro
        # whatever its condition, the file exemption this replaced being what
        # hid that. Judged here the element is read whole: only the
        # `do { ... } while (0)` idiom, which makes a macro a single statement,
        # is forgiven, and a macro-hidden loop with any other condition sets
        # the form at the macro's own line.
        if(NOT _element_code MATCHES "while[ \t]*\\([ \t]*0[ \t]*\\)")
          set(_form "a 'do' loop")
        endif()
      elseif(_element_code MATCHES "\\}[ \t]*while")
        # The tail of a do-while written out over several lines; its `do {`
        # carries the requirement.
      elseif(_element_code MATCHES "[^A-Za-z_0-9]while[ \t]*\\(")
        set(_form "a 'while' loop")
      elseif(_element_code MATCHES "[^A-Za-z_0-9]for[ \t]*\\([^)]*;[ \t]*;")
        # Covers both `for (;;)` and `for (init;; incr)`.
        set(_form "a 'for' loop with an empty condition")
      elseif(_element_code MATCHES "[^A-Za-z_0-9]for[ \t]*\\(")
        cudec_for_colon_at_top_level("${_element_code}" _is_range_for)
        if(_balance GREATER 0)
          set(_for_open ${_balance})
          set(_for_semis ${_line_semis})
          set(_for_start ${_element_start})
          set(_for_lines 1)
          set(_for_text "${_element_code}")
        elseif(_line_semis LESS 2 AND NOT _is_range_for)
          # A balanced header with fewer than two semicolons has no visible
          # induction variable - a macro-hidden loop, say. A range-for
          # (`for (T& x : container)`) is exempt: its bound is the container's
          # size, loop-invariant and visible on the spot, exactly like the
          # counted form. What counts as one is decided above, by
          # cudec_for_colon_at_top_level.
          set(_form "a 'for' header without a visible condition and increment")
        endif()
      endif()
      if(_form
         AND NOT _element_code MATCHES "fuel"
         AND NOT _previous MATCHES "fuel")
        set(_pending_reason "${_form}")
        set(_pending_line ${_form_line})
        set(_pending_left 3)
      endif()
      set(_previous "${_element_code}")
    endif()

    # The legacy intrinsics, matched on the identifier rather than on a
    # following "(": a call wrapped as `__shfl\n    (v, 0)` is the same
    # violation, and the _sync forms end in "_sync" so they never match here.
    foreach(_legacy __shfl __shfl_up __shfl_down __shfl_xor __ballot __any
                    __all __match_any)
      if(_code MATCHES "[^A-Za-z_0-9]${_legacy}[^A-Za-z_0-9]")
        string(APPEND _violations
               "  ${path}:${_lineno}: legacy warp intrinsic '${_legacy}' - "
               "only the _sync form can state which lanes participate\n")
      endif()
    endforeach()

    set(_tail_balance 0)
    if(_code MATCHES "(__[A-Za-z_0-9]*_sync|__syncwarp)[ \t]*\\(")
      # Everything from the last collective's '(' to the end of the line. If
      # its parentheses do not close there, the call is wrapped - and a
      # wrapped call hides whatever follows the break from EVERY rule below,
      # which are all line-scoped. Checking the balance rather than only
      # '(' at end-of-line catches the wrap AFTER the mask, where the source
      # lane and the predicate live:
      #     __shfl_sync(0xFFFFFFFFu, value,
      #                 seq.match_len % kWarpSize);
      string(REGEX MATCH "(__[A-Za-z_0-9]*_sync|__syncwarp)[ \t]*\\(([^,)]*)"
                   _call "${_code}")
      set(_collective "${CMAKE_MATCH_1}")
      string(STRIP "${CMAKE_MATCH_2}" _mask)
      string(REGEX REPLACE "^.*(__[A-Za-z_0-9]*_sync|__syncwarp)[ \t]*\\(" "("
                   _call_tail "${_code}")
      string(LENGTH "${_call_tail}" _tail_all)
      string(REPLACE "(" "" _stripped "${_call_tail}")
      string(LENGTH "${_stripped}" _tail_no_open)
      string(REPLACE ")" "" _stripped "${_call_tail}")
      string(LENGTH "${_stripped}" _tail_no_close)
      math(EXPR _tail_opens "${_tail_all} - ${_tail_no_open}")
      math(EXPR _tail_closes "${_tail_all} - ${_tail_no_close}")
      math(EXPR _tail_balance "${_tail_opens} - ${_tail_closes}")
    endif()
    if(_code MATCHES "(__[A-Za-z_0-9]*_sync|__syncwarp)[ \t]*\\("
       AND _tail_balance GREATER 0)
      string(APPEND _violations
             "  ${path}:${_lineno}: '${_collective}' is wrapped across "
             "lines - this scan is line-scoped, so a wrapped call hides its "
             "mask, source lane and predicate from every rule here; keep a "
             "collective on one line\n")
    elseif(_code MATCHES "(__[A-Za-z_0-9]*_sync|__syncwarp)[ \t]*\\(")
      # An absent mask is the full warp, which only __syncwarp() may imply;
      # every other collective must name its participants.
      if(_mask STREQUAL "" AND NOT _collective STREQUAL "__syncwarp")
        string(APPEND _violations
               "  ${path}:${_lineno}: '${_collective}' has no mask argument "
               "on this line - only __syncwarp may imply the full warp\n")
      elseif(NOT _mask STREQUAL "" AND NOT _mask MATCHES
             "^(0[xX][fF][fF][fF][fF][fF][fF][fF][fF][uU]?|kFullWarpMask|__activemask\\(?)$")
        string(APPEND _violations
               "  ${path}:${_lineno}: '${_collective}' mask '${_mask}' is "
               "neither the full-warp constant nor __activemask()\n")
      endif()
      foreach(_field token offset literals_len match_len src_pos dst_pos)
        if(_code MATCHES "[^A-Za-z_0-9]${_field}[^A-Za-z_0-9]")
          string(APPEND _violations
                 "  ${path}:${_lineno}: '${_collective}' names the parsed "
                 "bitstream field '${_field}' - participation metadata must "
                 "never come from unvalidated input\n")
        endif()
      endforeach()
      if(_code MATCHES "[^A-Za-z_0-9](seq|sequence|parser)\\.")
        string(APPEND _violations
               "  ${path}:${_lineno}: '${_collective}' reads a parsed "
               "sequence field - participation metadata must never come from "
               "unvalidated input\n")
      endif()
    endif()

    # CUDA spells its half type four ways and C23 adds a fifth; all of them
    # reintroduce the reassociation the gpu_to_gpu level rules out.
    foreach(_fp float double half half2 __half __half2 __nv_bfloat16
                __nv_bfloat162 _Float16 __nv_fp8_e4m3 __nv_fp8_e5m2)
      if(_code MATCHES "[^A-Za-z_0-9]${_fp}[^A-Za-z_0-9]")
        string(APPEND _violations
               "  ${path}:${_lineno}: floating-point type '${_fp}' - the "
               "decode path is integer-only, which is what makes the "
               "gpu_to_gpu determinism level hold (docs/DETERMINISM.md)\n")
      endif()
    endforeach()
    if(_code MATCHES "atomic[A-Za-z]*[ \t]*\\("
       AND _code MATCHES "[^A-Za-z_0-9](float|double|half|__half|__nv_bfloat16|_Float16)[^A-Za-z_0-9]")
      string(APPEND _violations
             "  ${path}:${_lineno}: floating-point atomic - its result "
             "depends on the order the device happens to schedule\n")
    endif()

    # The wave width. The digits may appear on exactly one kind of line: the
    # one that binds them to a name, as the whole initializer, where a reader
    # looking for what the width is finds it. The dot in the exclusions keeps
    # 1.32 and .64 out, and every identifier character keeps uint32_t,
    # INT32_MAX and 0x64 out.
    set(_width_bound FALSE)
    if(_code MATCHES "constexpr[^=]*=[ \t]*(32|64)[uUlL]*[ \t]*;")
      set(_width_bound TRUE)
    elseif(_code MATCHES
           "#[ \t]*define[ \t]+[A-Za-z_][A-Za-z_0-9]*[ \t]+(32|64)[uUlL]*[ \t]*$")
      set(_width_bound TRUE)
    endif()
    if(NOT _width_bound AND _code MATCHES
       "[^A-Za-z_0-9.](32|64)[uUlL]*[^A-Za-z_0-9.]")
      string(REGEX MATCH "[^A-Za-z_0-9.](32|64)[uUlL]*[^A-Za-z_0-9.]" _unused
                   "${_code}")
      string(APPEND _violations
             "  ${path}:${_lineno}: bare wave-width literal "
             "'${CMAKE_MATCH_1}' - the width comes from the WaveSize "
             "parameter or from a constant named after it, never from digits "
             "in an expression; a hardcoded 32 is invisible on every device "
             "this project can test on\n")
    endif()
  endforeach()
  # A partial read must never look like a clean scan: a source that reached no
  # rule and one that reached every rule and satisfied them all leave the same
  # trace, which is nothing. The protections above are what make the two counts
  # agree, so this is the check that says they still do - and the one that
  # reds if a fifth character with list meaning turns up (issue #294). A hard
  # error rather than a violation: a reader that cannot read is not a finding
  # about the source it was pointed at.
  if(NOT _lineno EQUAL _expected_lines)
    message(
      FATAL_ERROR
        "${path} was read as ${_lineno} lines and carries ${_expected_lines}: "
        "this scan cannot read it whole, and a partial read reports the same "
        "silence as a clean one (issue #294)")
  endif()
  # A file whose last line ends in a backslash leaves an element the loop rule
  # never got to judge, so the construct inside it went unread. Reject rather
  # than return silently, for the same reason the unbalanced `for` header
  # below is rejected.
  if(_element_open)
    string(APPEND _violations
           "  ${path}:${_element_start}: a backslash continuation left open at "
           "end of file - the construct it carries was not judged\n")
  endif()
  if(check_loops AND _pending_left GREATER 0)
    string(APPEND _violations
           "  ${path}:${_pending_line}: ${_pending_reason} with no explicit "
           "decrementing 'fuel' cap\n")
  endif()
  # A `for` header still open at end of file means the scan lost track of the
  # parentheses and stopped enforcing the loop rule from that point on. Say so
  # rather than return silently - a scanner that quietly stops checking is the
  # failure mode these probes and this check exist to prevent.
  if(check_loops AND _for_open GREATER 0)
    string(APPEND _violations
           "  ${path}:${_for_start}: a 'for' header left unbalanced at end of "
           "file - the loop rule was not enforced below this line\n")
  endif()
  set(${out_violations} "${_violations}" PARENT_SCOPE)
endfunction()

# Liveness probes for the scanner itself, in the same spirit as the
# warnings-gate control TU above. The clean probe must come back silent -
# including its comment full of the very keywords the rules match, and
# including a legitimately wrapped `for` header, which an earlier revision of
# this scanner reported as a header without a condition. Each violating probe
# must come back with the TEXT of the rule it violates: asserting only
# non-emptiness let the floating-point-atomic rule ride on the
# floating-point-TYPE rule firing on the same probe's signature, which is
# exactly the kind of silently-dead sub-rule these probes exist to catch.
# These files are scanned as text and never compiled.
set(_probe_dir ${CMAKE_CURRENT_BINARY_DIR}/probes)
file(
  WRITE ${_probe_dir}/scan_clean.cu
  "/* Prose naming while, do, for, float, double, __shfl( and a mask from a\n"
  " * token - the comment stripper must remove all of it. */\n"
  "unsigned probe_counted(unsigned n) {\n"
  "    unsigned total = 0;\n"
  "    for (unsigned i = 0; i < n; i++) {\n"
  "        total += i;\n"
  "    }\n"
  "    for (unsigned outer = 0;\n"
  "         outer < n;\n"
  "         outer++) {\n"
  "        total += outer;\n"
  "    }\n"
  "    for (const Block& block : blocks) {\n"
  "        total += block.size;\n"
  "    }\n"
  "    for (const Block& wrapped :\n"
  "         blocks) {\n"
  "        total += wrapped.size;\n"
  "    }\n"
  "    for (const cudec_detail::Block& scoped : blocks) {\n"
  "        total += scoped.size;\n"
  "    }\n"
  "    for (const cudec_detail::Block& scoped_wrapped :\n"
  "         blocks) {\n"
  "        total += scoped_wrapped.size;\n"
  "    }\n"
  "    unsigned fuel = n + 1;\n"
  "    while (fuel-- != 0) {\n"
  "        total += 1;\n"
  "    }\n"
  "    unsigned left = n;\n"
  "    unsigned fuel_drain = n + 1;\n"
  "    do {\n"
  "        if (fuel_drain-- == 0) {\n"
  "            break;\n"
  "        }\n"
  "        left = left - 1;\n"
  "    } while (left != 0);\n"
  "    return total;\n"
  "}\n"
  "__device__ int probe_collective(int v) {\n"
  "    __syncwarp();\n"
  "    return __shfl_sync(0xFFFFFFFFu, v, 0);\n"
  "}\n"
  "#define PROBE_LANE(v) \\\n"
  "    __shfl_sync(0xFFFFFFFFu, v, 0)\n"
  "__device__ int probe_macro_collective(int v) {\n"
  "    return PROBE_LANE(v);\n"
  "}\n")

# One probe per sub-rule, each paired with the text it must produce.
file(
  WRITE ${_probe_dir}/scan_loop_while.cu
  "unsigned probe(const unsigned char* src, unsigned n) {\n"
  "    unsigned count = src[0];\n"
  "    while (count != 0) {\n"
  "        count += src[count % n];\n"
  "    }\n"
  "    return count;\n"
  "}\n")
file(
  WRITE ${_probe_dir}/scan_loop_do.cu
  "unsigned probe(const unsigned char* src, unsigned n) {\n"
  "    unsigned count = src[0];\n"
  "    do {\n"
  "        count += src[count % n];\n"
  "    } while (count != 0);\n"
  "    return count;\n"
  "}\n")
file(WRITE ${_probe_dir}/scan_loop_forever.cu
     "unsigned probe(const unsigned char* src) {\n"
     "    unsigned count = 0;\n" "    for (;;) {\n"
     "        count += src[count];\n" "    }\n" "    return count;\n" "}\n")
file(WRITE ${_probe_dir}/scan_loop_empty_condition.cu
     "unsigned probe(const unsigned char* src) {\n"
     "    unsigned count = 0;\n" "    for (unsigned i = 0;; i++) {\n"
     "        count += src[i];\n" "    }\n" "    return count;\n" "}\n")
file(WRITE ${_probe_dir}/scan_legacy_intrinsic.cu
     "__device__ int probe(int v) {\n" "    return __shfl(v, 0);\n" "}\n")
file(WRITE ${_probe_dir}/scan_legacy_intrinsic_wrapped.cu
     "__device__ int probe(int v) {\n" "    return __ballot\n"
     "        (v > 0);\n" "}\n")
file(WRITE ${_probe_dir}/scan_derived_mask.cu
     "__device__ int probe(int v, unsigned mask_bits) {\n"
     "    return __shfl_sync(mask_bits, v, 0);\n" "}\n")
file(WRITE ${_probe_dir}/scan_bitstream_field.cu
     "__device__ int probe(int v, unsigned offset) {\n"
     "    return __shfl_sync(0xFFFFFFFFu, v, offset);\n" "}\n")
file(WRITE ${_probe_dir}/scan_sequence_field.cu
     "__device__ int probe(int v, Lz4Sequence seq) {\n"
     "    return __shfl_sync(0xFFFFFFFFu, v, seq.lane);\n" "}\n")
file(WRITE ${_probe_dir}/scan_wrapped_collective.cu
     "__device__ int probe(int v, unsigned mask_from_token) {\n"
     "    return __shfl_sync(\n" "        mask_from_token, v, 0);\n" "}\n")
file(WRITE ${_probe_dir}/scan_missing_mask.cu
     "__device__ int probe(int v) {\n" "    return __shfl_sync();\n" "}\n")
file(WRITE ${_probe_dir}/scan_fp_type.cu
     "__device__ float probe(float a) {\n" "    return a + a;\n" "}\n")
file(WRITE ${_probe_dir}/scan_fp_half.cu
     "__device__ void probe(half* acc) {\n" "    *acc = *acc;\n" "}\n")
file(WRITE ${_probe_dir}/scan_fp_atomic.cu
     "__device__ void probe(void* p) {\n"
     "    atomicAdd(static_cast<float*>(p), 1);\n" "}\n")
# The remaining `for` branch: a balanced header with no visible induction
# variable, single-line and wrapped. Both paths report the same rule but reach
# it through different code, and the single-line one silently reported the
# wrong line number until it got a probe of its own.
file(WRITE ${_probe_dir}/scan_for_opaque.cu
     "unsigned probe(const unsigned char* src) {\n"
     "    unsigned count = 0;\n" "    for (EACH_CHUNK(chunk, src)) {\n"
     "        count += chunk;\n" "    }\n" "    return count;\n" "}\n")
file(WRITE ${_probe_dir}/scan_for_opaque_wrapped.cu
     "unsigned probe(const unsigned char* src) {\n"
     "    unsigned count = 0;\n" "    for (EACH_CHUNK(chunk,\n"
     "                    src)) {\n" "        count += chunk;\n" "    }\n"
     "    return count;\n" "}\n")
# The same opaque header carrying a qualified name. Its '::' is not a
# range-for's ':' and must not exempt it - single-line, and wrapped with the
# qualified name on the continuation line, where only the accumulated header
# text can see it.
file(WRITE ${_probe_dir}/scan_for_opaque_qualified.cu
     "unsigned probe(const unsigned char* src) {\n"
     "    unsigned count = 0;\n"
     "    for (EACH_CHUNK(chunk, cudec_detail::chunks_of(src))) {\n"
     "        count += chunk;\n" "    }\n" "    return count;\n" "}\n")
file(WRITE ${_probe_dir}/scan_for_opaque_qualified_wrapped.cu
     "unsigned probe(const unsigned char* src) {\n"
     "    unsigned count = 0;\n" "    for (EACH_CHUNK(chunk,\n"
     "                    cudec_detail::chunks_of(src))) {\n"
     "        count += chunk;\n" "    }\n" "    return count;\n" "}\n")
# The other two colons a macro-hidden header can plausibly carry, neither of
# them a range-for's: one inside a nested call's string literal, one a
# ternary. Both must still be reported (issue #250). The wrapped ternary
# covers the multi-line branch, which decides the exemption separately.
file(WRITE ${_probe_dir}/scan_for_colon_literal.cu
     "unsigned probe(const unsigned char* src) {\n"
     "    unsigned count = 0;\n"
     "    for (EACH_FIELD(field, src, \"key:value\")) {\n"
     "        count += field;\n" "    }\n" "    return count;\n" "}\n")
file(WRITE ${_probe_dir}/scan_for_colon_ternary.cu
     "unsigned probe(const unsigned char* src, int wide) {\n"
     "    unsigned count = 0;\n"
     "    for (EACH_CHUNK(chunk, wide ? src[0] : src[1])) {\n"
     "        count += chunk;\n" "    }\n" "    return count;\n" "}\n")
file(WRITE ${_probe_dir}/scan_for_colon_ternary_wrapped.cu
     "unsigned probe(const unsigned char* src, int wide) {\n"
     "    unsigned count = 0;\n" "    for (EACH_CHUNK(chunk,\n"
     "                    wide ? src[0] : src[1])) {\n"
     "        count += chunk;\n" "    }\n" "    return count;\n" "}\n")
# A `for` header the scan cannot balance must be rejected, not skipped: while
# the tracker is open the whole loop rule is off, so a silent skip would
# disable the termination check for the rest of the file. String literals are
# deliberately not stripped, so an unbalanced '(' inside one is exactly how
# this happens. Both exits are probed - the line cap and end of file.
file(WRITE ${_probe_dir}/scan_for_unbalanced.cu
     "void probe() {\n"
     "    for (int i = 0; i < f(\"an unbalanced ( in a literal\"); i++) {\n"
     "        g(1);\n" "        g(2);\n" "        g(3);\n" "        g(4);\n"
     "        g(5);\n" "        g(6);\n" "        g(7);\n" "        g(8);\n"
     "        g(9);\n" "    }\n" "}\n")
file(WRITE ${_probe_dir}/scan_for_unbalanced_eof.cu
     "void probe() {\n"
     "    for (int i = 0; i < f(\"an unbalanced ( in a literal\"); i++) {\n"
     "        g(i);\n" "    }\n" "}\n")
# The wrap AFTER the mask: the mask itself is valid, so only a balance-based
# wrapped-collective rule catches this one.
file(WRITE ${_probe_dir}/scan_wrapped_after_mask.cu
     "__device__ int probe(int v, Lz4Sequence seq) {\n"
     "    return __shfl_sync(0xFFFFFFFFu, v,\n"
     "                       seq.match_len % 32);\n" "}\n")
# The two halves of the macro case, and they are one pair rather than two
# probes: a backslash-continued macro arrives as a single element, so the same
# element carries the loop and its condition. The idiom must come back silent
# and a real loop wearing the same wrapper must not, or the clause that
# forgives the wrapper has turned the rule off for every macro in the tree -
# which is what the file exemption on src/cuda_raii.h was doing.
#
# Both put the #define on line 1, and the violating one asserts that line
# number: the report lands on the macro's own line only while the join still
# happens, so a scan that stopped joining would move it to line 2 and stop
# covering the case these exist for.
file(WRITE ${_probe_dir}/scan_loop_do_macro_idiom.cu
     "#define PROBE_CHECK(call, on_failure) \\\n"
     "    do {                              \\\n"
     "        if ((call) != 0) {            \\\n"
     "            on_failure;               \\\n"
     "        }                             \\\n"
     "    } while (0)\n"
     "unsigned probe(unsigned n) {\n" "    PROBE_CHECK(n, n = 0);\n"
     "    return n;\n" "}\n")
file(WRITE ${_probe_dir}/scan_loop_do_macro.cu
     "#define PROBE_RETRY(call, n) \\\n" "    do {                 \\\n"
     "        n = n + (call);  \\\n" "    } while (n != 0)\n"
     "unsigned probe(const unsigned char* src, unsigned n) {\n"
     "    PROBE_RETRY(src[n], n);\n" "    return n;\n" "}\n")
# The four continuation probes (issue #251). A collective wrapped inside a
# macro is the same violation as one wrapped outside it, and it is asserted
# with its line number: the report has to land on the line the call wrapped
# at, which is the second, and a scan that joined the macro back together
# before this rule saw it reported nothing at all.
file(WRITE ${_probe_dir}/scan_macro_wrapped_collective.cu
     "#define PROBE_LANE(v, s) \\\n"
     "    __shfl_sync(0xFFFFFFFFu, v,\\\n" "                s.lane % 32)\n")
# `s.lane` is neither in the parsed-field list nor under the sequence prefix,
# so no other rule can cover for the wrapped-collective rule here.
#
# A violation BELOW a continuation carries the line a reader can open. Every
# line under the macro was short by the number of continuation lines above it,
# so this one was reported at line 4 - inside the macro, which does not
# contain it.
file(WRITE ${_probe_dir}/scan_line_after_continuation.cu
     "#define PROBE_WRAP(a, b) \\\n" "    do {                 \\\n"
     "        (a) = (b);       \\\n" "    } while (0)\n"
     "unsigned probe(const unsigned char* src) {\n"
     "    unsigned count = src[0];\n" "    while (count != 0) {\n"
     "        count += src[count];\n" "    }\n" "    return count;\n" "}\n")
# A macro-hidden `for` header, which the loop rule must judge on the joined
# element and therefore on that element's semicolon count. The join inserts
# one separator per continuation line, and two of them are all it takes for
# this header to read as a counted `for` with a visible induction variable -
# a fail-open in the termination rule reported as a pass.
file(WRITE ${_probe_dir}/scan_macro_for_opaque.cu
     "#define PROBE_EACH(x) \\\n" "    for (EACH_CHUNK(chunk, x)) \\\n"
     "        use(chunk);\n" "unsigned probe(const unsigned char* src) {\n"
     "    PROBE_EACH(src);\n" "    return 0;\n" "}\n")
# A continuation the file never closes. The construct it carries reaches no
# rule at all, so it is rejected rather than dropped - the same direction as
# the unbalanced `for` header above.
file(WRITE ${_probe_dir}/scan_continuation_at_eof.cu
     "void probe() {\n" "    g(1);\n" "}\n" "#define PROBE_TAIL(x) \\\n")
# The wave-width rule (issue #211). One violating spelling per literal, so the
# report's own text is what the probe asserts and the alternation cannot go
# half-dead, plus the two silent shapes the rule exists to permit: the width
# arriving as the template parameter, and the one line that binds the digits to
# a name - which is the shape src/lz4_decode.cuh has carried since M1.
file(WRITE ${_probe_dir}/scan_width_literal_32.cu
     "template <int WaveSize>
"
     "__device__ unsigned probe(unsigned tid) {
"
     "    return tid % 32;
" "}
")
file(WRITE ${_probe_dir}/scan_width_literal_64.cu
     "template <int WaveSize>
"
     "__device__ unsigned probe(unsigned tid) {
"
     "    return tid / 64;
" "}
")
file(WRITE ${_probe_dir}/scan_width_from_parameter.cu
     "template <int WaveSize>
"
     "__device__ unsigned probe(unsigned tid) {
"
     "    constexpr unsigned kLanes = WaveSize;
"
     "    return tid % kLanes;
" "}
")
file(WRITE ${_probe_dir}/scan_width_named_constant.cu
     "constexpr unsigned kWaveLanes = 32;
"
     "__device__ unsigned probe(unsigned tid) {
"
     "    return tid % kWaveLanes;
" "}
")
# The bytes the reader borrows for its own bookkeeping, in the file it is
# reading. They have to come back refused, or the borrowing is a way to write a
# separator or a backslash the split will not see as one. One probe per end of
# the borrowed range: 0x01 is the backslash marker and 0x04 the bracket marker
# the bracket protection added (issue #294).
string(ASCII 1 _probe_ctrl)
file(WRITE ${_probe_dir}/scan_control_byte.cu
     "void probe() {\n" "    g(1);${_probe_ctrl}\n" "}\n")
string(ASCII 4 _probe_ctrl_close)
file(WRITE ${_probe_dir}/scan_control_byte_close.cu
     "void probe() {\n" "    g(1);${_probe_ctrl_close}\n" "}\n")

# An unbalanced bracket above a violation. Before the bracket protection, the
# '[' on the first line swallowed every line separator after it, the whole file
# arrived as one element numbered line 1, and the loop rule below it reached
# nothing: a source that violates a fail-closed rule left exactly the trace of
# one that satisfies it. The shape is copied from src/snappy_block.h, where a
# bracket in a comment ended the scan 290 lines early (issue #294). The
# expectation carries the LINE NUMBER, because a reader that swallows a
# separator and still reports the rule would report it against the wrong line.
file(
  WRITE ${_probe_dir}/scan_bracket_partial.cu
  "/* A LITERAL copies length bytes from src[from ..\n"
  " * and the bracket this comment opened never closes.\n"
  " */\n"
  "unsigned probe(unsigned n) {\n"
  "    while (n != 0) {\n"
  "        n = n - 1;\n"
  "    }\n"
  "    return n;\n"
  "}\n")

cudec_scan_source(${_probe_dir}/scan_clean.cu TRUE _probe_clean)
if(_probe_clean)
  message(
    FATAL_ERROR
      "the structural scanner rejects a compliant probe - fix the scanner "
      "before trusting it:\n${_probe_clean}")
endif()
# "probe|the text its rule must emit" - '|' rather than ';' because CMake
# would split a list element on the latter. An entry whose text is empty is a
# probe that must come back SILENT, and the loop below spells that out as its
# own branch.
set(_scanner_probes
    "scan_loop_while|a 'while' loop with no explicit decrementing 'fuel' cap"
    "scan_loop_do|a 'do' loop with no explicit decrementing 'fuel' cap"
    "scan_loop_forever|a 'for' loop with an empty condition with no explicit"
    "scan_loop_empty_condition|a 'for' loop with an empty condition with no explicit"
    "scan_legacy_intrinsic|legacy warp intrinsic '__shfl'"
    "scan_legacy_intrinsic_wrapped|legacy warp intrinsic '__ballot'"
    "scan_derived_mask|mask 'mask_bits' is neither the full-warp constant"
    "scan_bitstream_field|names the parsed bitstream field 'offset'"
    "scan_sequence_field|reads a parsed sequence field"
    "scan_wrapped_collective|is wrapped across lines"
    "scan_missing_mask|has no mask argument on this line"
    "scan_fp_type|floating-point type 'float'"
    "scan_fp_half|floating-point type 'half'"
    "scan_fp_atomic|floating-point atomic"
    "scan_for_opaque|scan_for_opaque.cu:3: a 'for' header without a visible"
    "scan_for_opaque_wrapped|scan_for_opaque_wrapped.cu:3: a 'for' header without a visible"
    "scan_for_opaque_qualified|scan_for_opaque_qualified.cu:3: a 'for' header without a visible"
    "scan_for_opaque_qualified_wrapped|scan_for_opaque_qualified_wrapped.cu:3: a 'for' header without a visible"
    "scan_for_colon_literal|scan_for_colon_literal.cu:3: a 'for' header without a visible"
    "scan_for_colon_ternary|scan_for_colon_ternary.cu:3: a 'for' header without a visible"
    "scan_for_colon_ternary_wrapped|scan_for_colon_ternary_wrapped.cu:3: a 'for' header without a visible"
    "scan_for_unbalanced|closing parenthesis was not found within 8 lines"
    "scan_for_unbalanced_eof|left unbalanced at end of file"
    "scan_wrapped_after_mask|is wrapped across lines"
    "scan_loop_do_macro_idiom|"
    "scan_loop_do_macro|scan_loop_do_macro.cu:1: a 'do' loop with no explicit decrementing 'fuel' cap"
    "scan_macro_wrapped_collective|scan_macro_wrapped_collective.cu:2: '__shfl_sync' is wrapped across"
    "scan_line_after_continuation|scan_line_after_continuation.cu:7: a 'while' loop with no explicit"
    "scan_macro_for_opaque|scan_macro_for_opaque.cu:1: a 'for' header without a visible"
    "scan_continuation_at_eof|scan_continuation_at_eof.cu:4: a backslash continuation left open at end of file"
    "scan_control_byte|a control byte (0x01 to 0x04) in a source read as text"
    "scan_control_byte_close|a control byte (0x01 to 0x04) in a source read"
    "scan_bracket_partial|scan_bracket_partial.cu:5: a 'while' loop with no explicit"
    "scan_width_literal_32|scan_width_literal_32.cu:3: bare wave-width literal '32'"
    "scan_width_literal_64|scan_width_literal_64.cu:3: bare wave-width literal '64'"
    "scan_width_from_parameter|"
    "scan_width_named_constant|"
)
foreach(_entry IN LISTS _scanner_probes)
  string(REGEX REPLACE "^([^|]+)\\|(.*)$" "\\1" _probe "${_entry}")
  string(REGEX REPLACE "^([^|]+)\\|(.*)$" "\\2" _expected "${_entry}")
  cudec_scan_source(${_probe_dir}/${_probe}.cu TRUE _probe_result)
  if(_expected STREQUAL "")
    # A silent probe is checked here rather than through string(FIND): an
    # empty needle is found at position 0 in every string, so routing it
    # through the branch below would assert nothing at all.
    if(_probe_result)
      message(
        FATAL_ERROR
          "the structural scanner rejects a compliant probe: '${_probe}' must "
          "come back silent (issue #72). It reported:\n${_probe_result}")
    endif()
  else()
    string(FIND "${_probe_result}" "${_expected}" _found_at)
    if(_found_at EQUAL -1)
      message(
        FATAL_ERROR
          "the structural scanner is dead for one rule: probe '${_probe}' did "
          "not report \"${_expected}\" (issue #72). It reported:\n"
          "${_probe_result}")
    endif()
  endif()
endforeach()


# Every source under src/ is scanned, and the termination rule applies to all
# of them unless a file is EXPLICITLY exempted below. Opt-out rather than
# opt-in on purpose: an opt-in list makes a new file invisible by default,
# which is the wrong direction for a fail-closed rule and would have hidden
# the first file to land in src/kernels/ (device code moves there from M1) or
# any future decode header that is a plain .h, like src/lz4_block.h is today.
# Each exemption carries its reason and is reviewed when it is added.
#
#  - xxhash32.h: its loops walk a caller-supplied buffer at a fixed stride
#    with no stream-derived exit condition, and it is a checksum helper rather
#    than a decoder.
#
# It is still held to the warp-collective and floating-point rules.
set(_loop_rule_exempt xxhash32.h)
file(GLOB_RECURSE _all_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../src
     CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.c
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cc
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cuh
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.hpp
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.inl)
if(NOT _all_sources)
  message(FATAL_ERROR "the structural scan found no sources under src/ - the "
                      "glob has stopped matching (issue #72)")
endif()
# An exemption for a file that no longer exists is stale and hides nothing;
# say so rather than carry it silently.
foreach(_exempt IN LISTS _loop_rule_exempt)
  if(NOT _exempt IN_LIST _all_sources)
    message(FATAL_ERROR "src/${_exempt} is exempt from the termination rule "
                        "but no longer exists - drop the exemption "
                        "(issue #72)")
  endif()
endforeach()

set(_structural_violations "")
foreach(_source IN LISTS _all_sources)
  set(_check_loops TRUE)
  if(_source IN_LIST _loop_rule_exempt)
    set(_check_loops FALSE)
  endif()
  cudec_scan_source(${CMAKE_CURRENT_SOURCE_DIR}/../src/${_source}
                    ${_check_loops} _found)
  string(APPEND _structural_violations "${_found}")
endforeach()
if(_structural_violations)
  message(FATAL_ERROR "structural rules violated (issue #72):\n"
                      "${_structural_violations}")
endif()

# Records this directory's ctest entries as timeout-checked; the sweep at the
# end of the top-level CMakeLists.txt is what makes the property unforgettable.
cudec_assert_test_timeouts()
