# facebook/zstd and its two oracle targets, zstd_oracle and zstd_full, the M5
# oracle (issue #181). In its own module for the reason the lz4, snappy and
# GDeflate modules give: a second FetchContent_Declare under the same name is
# not an error, the first declaration silently wins, and two copies of a pin
# can drift with nothing to report which one the build used. fuzz/ is that
# second consumer (issue #180), and this file is the move that keeps the pin
# in one place rather than the copy that would have made it two.
include_guard(GLOBAL)

# Supply chain: the hash-verified archive is the only acceptable oracle
# source. Stated here as well as in tests/CMakeLists.txt because whichever
# directory reaches this file first is the one that fetches.
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
include(FetchContent)

# facebook/zstd, the M5 oracle, pinned by the SHA-256 of the
# UPSTREAM-UPLOADED release asset - the shape the pinning policy prefers and
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
