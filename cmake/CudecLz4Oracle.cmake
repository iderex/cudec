# The liblz4 oracle, in one place because two directories now need it: the
# test net under tests/ and the fuzz targets under fuzz/ (issue #140). A
# second FetchContent_Declare for the same name would not be an error - the
# first declaration silently wins - so two copies of the pin could drift and
# the losing copy would never say so. One file, one pin, one pair of targets.
include_guard(GLOBAL)

# Supply chain: the hash-verified tarball is the only acceptable oracle
# source - a system copy must never silently substitute for it. Stated here
# as well as in tests/CMakeLists.txt because whichever directory reaches this
# file first is the one that fetches, and the other may never process the
# line at all.
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)

# liblz4, pinned by the SHA-256 of the upstream-uploaded release asset
# (self-computed at pin time and cross-checked against conan-center; the
# auto-generated /archive/ tarballs are avoided - GitHub regenerated them
# in 2023 and their hashes moved). Test-only dependency: the link
# allowlist in tests/CMakeLists.txt keeps it out of the library.
include(FetchContent)
FetchContent_Declare(
  lz4
  URL https://github.com/lz4/lz4/releases/download/v1.10.0/lz4-1.10.0.tar.gz
  URL_HASH
    SHA256=537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(lz4)

# The uploaded tarball has no top-level CMakeLists (lz4's lives in
# build/cmake), so MakeAvailable only populates the source tree; we compile
# the one translation unit the block-format oracle needs and depend on
# nothing of lz4's build system. SYSTEM include + none of our strict flags:
# third-party code is audited by hash, not reformatted.
add_library(lz4_oracle STATIC ${lz4_SOURCE_DIR}/lib/lz4.c)
target_include_directories(lz4_oracle SYSTEM PUBLIC ${lz4_SOURCE_DIR}/lib)

# The frame oracle adds liblz4's frame API and xxHash (the checksum the
# frame format uses) - the reference the cudec frame decoder is held to.
add_library(
  lz4_frame_oracle STATIC
  ${lz4_SOURCE_DIR}/lib/lz4.c ${lz4_SOURCE_DIR}/lib/lz4hc.c
  ${lz4_SOURCE_DIR}/lib/lz4frame.c ${lz4_SOURCE_DIR}/lib/xxhash.c)
target_include_directories(lz4_frame_oracle SYSTEM PUBLIC
                           ${lz4_SOURCE_DIR}/lib)
# -O2 unconditionally: the oracle is also the bench harness's timed
# decoder, and an empty CMAKE_BUILD_TYPE (the documented container
# command) would otherwise time -O0 reference code. Correctness is
# optimization-independent; honest numbers are not.
target_compile_options(lz4_oracle PRIVATE -O2)

# Never instrumented, in either the sanitizer gate or the fuzz gate: the
# reference is third-party code audited by hash, and a sanitizer finding
# inside it would red a gate that exists to judge this project's parser.
