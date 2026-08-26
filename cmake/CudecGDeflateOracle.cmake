# The NVIDIA/libdeflate GDeflate oracle, in one file for the reason the lz4
# and snappy modules give: a second FetchContent_Declare under the same name
# is not an error, the first declaration silently wins, and two copies of a
# pin can drift with nothing to report which one the build used. Only tests/
# reaches this today; fuzz/ is the second consumer the M4 fuzz rungs add.
include_guard(GLOBAL)

# Supply chain: the hash-verified archive is the only acceptable oracle
# source. Stated here as well as in tests/CMakeLists.txt because whichever
# directory reaches this file first is the one that fetches.
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
include(FetchContent)

# The M4 oracle. The pin is the commit and the SHA-256 of the archive at that
# commit, both decided in issue #147 and recorded in docs/MASTERPLAN.md
# section 11.7 - this executes that decision rather than taking it again.
#
# Three things about this pin differ from the other three oracles, and the
# policy bullet in section 5 names them so a later reader does not read them
# as oversights. In short: the fork publishes no release asset and no tag, so
# a commit archive is the only thing it offers; no second packaging ecosystem
# carries it, so the commit sha and the archive hash together are the
# cross-check; and the fork sits far behind upstream libdeflate, which is
# acceptable in a thing whose job is to be where real data came from and is
# the reason it is never a source to derive from.
#
# Re-derived at pin time (2026-08-08) rather than carried over:
#
#   gh api repos/NVIDIA/libdeflate/commits/gdeflate \
#     --jq '{sha: .sha, date: .commit.committer.date}'
#   {"date":"2026-07-29T18:49:44Z",
#    "sha":"8ba9502fb30d2bf728592d121f0d402e40c8cb05"}
#
#   gh api repos/ebiggers/libdeflate/compare/master...NVIDIA:libdeflate:gdeflate \
#     --jq '{status, ahead_by, behind_by}'
#   {"ahead_by":4,"behind_by":425,"status":"diverged"}
#
#   curl -sSL -o gdeflate.tar.gz https://github.com/NVIDIA/libdeflate/archive/\
#     8ba9502fb30d2bf728592d121f0d402e40c8cb05.tar.gz
#   sha256sum gdeflate.tar.gz
#   d1b4c38dce43e68a5f4c28d0fbb3f81a01953039a3dea63f4bd1a84d7ff80592
#   156610 bytes
#
# The pinned commit is the branch head as of that read, so the pin does not
# lag the fork; the fork lags upstream.
# CACHE INTERNAL rather than a plain set: this file is included from tests/,
# and the benchmark harness in bench/ prints the pin in its own methodology
# block. A directory-scoped variable would reach that harness as the empty
# string and it would attest an unnamed commit, which is the one failure a
# methodology block may not have. bench/CMakeLists.txt refuses an empty value.
set(CUDEC_GDEFLATE_COMMIT
    8ba9502fb30d2bf728592d121f0d402e40c8cb05
    CACHE INTERNAL "the pinned NVIDIA/libdeflate gdeflate commit")
FetchContent_Declare(
  gdeflate
  URL https://github.com/NVIDIA/libdeflate/archive/${CUDEC_GDEFLATE_COMMIT}.tar.gz
  URL_HASH
    SHA256=d1b4c38dce43e68a5f4c28d0fbb3f81a01953039a3dea63f4bd1a84d7ff80592
  # The fork ships Makefiles and no CMakeLists.txt, so MakeAvailable only
  # populates the source tree today. The subdirectory named here does not
  # exist on purpose: if a later pin gains a top-level CMakeLists, this keeps
  # its build system out instead of silently adopting it.
  SOURCE_SUBDIR cudec-uses-no-libdeflate-cmake
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(gdeflate)
# And the same statement as a check rather than as a comment: upstream's build
# system defines this target, so its presence means the line above stopped
# working.
if(TARGET libdeflate_static)
  message(
    FATAL_ERROR
      "libdeflate's own build system was added to this tree - the oracle rule "
      "is one target over the translation units the oracle's path needs "
      "(docs/MASTERPLAN.md section 5)")
endif()

# Six translation units, which is the whole GDeflate path in both directions.
# Derived by linking and reading what the linker asked for, not by copying
# upstream's Makefile:
#
# - gdeflate_compress.c / gdeflate_decompress.c are the two entry points.
# - deflate_compress.c holds the match finder and the block emitter that
#   gdeflate_compress.c drives; the GDeflate compressor is a thin layer on it.
# - utils.c is the allocator the alloc_/free_ calls go through.
# - x86/cpu_features.c and arm/cpu_features.c define the dispatch symbols the
#   decompressor references. Both compile everywhere: each file's body sits
#   behind its own architecture guard, so the one that is not this machine's
#   contributes nothing. Compiling only the matching one would make the
#   translation-unit list depend on the host, which is a build that differs
#   between the container and a contributor's machine for no gain.
#
# adler32.c, crc32.c and the zlib/gzip wrappers are deliberately absent:
# GDeflate carries no checksum of any kind (docs/MASTERPLAN.md section 11.4),
# so nothing on this path reaches them.
#
# SYSTEM include and none of the project's strict flags, the rule the other
# three oracles follow: third-party code is audited by hash, not reformatted.
add_library(
  gdeflate_oracle STATIC
  ${gdeflate_SOURCE_DIR}/lib/gdeflate_compress.c
  ${gdeflate_SOURCE_DIR}/lib/gdeflate_decompress.c
  ${gdeflate_SOURCE_DIR}/lib/deflate_compress.c
  ${gdeflate_SOURCE_DIR}/lib/utils.c
  ${gdeflate_SOURCE_DIR}/lib/x86/cpu_features.c
  ${gdeflate_SOURCE_DIR}/lib/arm/cpu_features.c)
target_include_directories(
  gdeflate_oracle SYSTEM PUBLIC ${gdeflate_SOURCE_DIR}
                                ${gdeflate_SOURCE_DIR}/common)
# -O2 for the reason lz4_oracle and snappy_oracle carry it: the documented
# container command sets no CMAKE_BUILD_TYPE, and the reference codec runs
# over whole corpora during fixture generation.
target_compile_options(gdeflate_oracle PRIVATE -O2)

# Never instrumented, in either the sanitizer gate or the fuzz gate, for the
# reason the other oracles state: a sanitizer finding inside third-party code
# audited by hash would red a gate that exists to judge this project's parser.
