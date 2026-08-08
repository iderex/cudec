# The google/snappy oracle, in one place because two directories need it: the
# test net under tests/ and the fuzz targets under fuzz/ (issue #90). The same
# reasoning as cmake/CudecLz4Oracle.cmake, and for the same hazard: a second
# FetchContent_Declare under one name is not an error, the first declaration
# wins in silence, so two copies of the pin could drift with nothing to report
# which one the build used.
include_guard(GLOBAL)

# Supply chain: the hash-verified archive is the only acceptable oracle
# source. Stated here as well as in tests/CMakeLists.txt because whichever
# directory reaches this file first is the one that fetches.
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
include(FetchContent)

# The M3 oracle, pinned by the SHA-256 of the tag archive. snappy publishes no
# maintainer-uploaded release asset, so this is the fallback the pinning
# policy names (docs/MASTERPLAN.md section 5): the auto-generated archive,
# pinned, with a future hash mismatch read as the invariant working rather
# than as noise. Cross-checked at pin time against a second packaging
# ecosystem each - the SHA-256 against the Homebrew formula, the SHA-512
# 0c1e1019e1bec9281f9877996d896e59e1533456130143224acb9cbfc35c1b0dd9de0a76e4a36494844d9ec58c295eed8c50bdf6dbabe47cf679652eb24b1281
# against vcpkg (conan-center only reaches 1.2.1). 1108618 bytes.
set(CUDEC_SNAPPY_VERSION 1.2.2)
FetchContent_Declare(
  snappy
  URL https://github.com/google/snappy/archive/refs/tags/${CUDEC_SNAPPY_VERSION}.tar.gz
  URL_HASH
    SHA256=90f74bc1fbf78a6c56b3c4a082a05103b3a56bb17bca1a27e052ea11723292dc
  # snappy, unlike lz4, DOES ship a top-level CMakeLists.txt, and pointing
  # MakeAvailable at a directory that holds none is what keeps that build out
  # of this tree: one oracle target over the sources the decode path needs,
  # never the project's own build system, which would bring its tests, its
  # install rules and its generated config.h with it.
  SOURCE_SUBDIR cudec-uses-no-snappy-cmake
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(snappy)
# And the same statement as a check rather than as a comment: snappy's build
# system defines this target, so its presence means the line above stopped
# working - a newer snappy layout, or a MakeAvailable that stopped honouring
# SOURCE_SUBDIR.
if(TARGET snappy)
  message(
    FATAL_ERROR
      "snappy's own build system was added to this tree - the oracle rule is "
      "one target over the translation units the decode path needs "
      "(docs/MASTERPLAN.md section 5)")
endif()

# The source directory as a cache entry, because the guard above means only
# the first directory to include this file evaluates it: a later includer sees
# the targets, which are global, but not FetchContent's directory-scoped
# variables. tests/ reads this for snappy's own testdata corpus.
set(CUDEC_SNAPPY_SOURCE_DIR "${snappy_SOURCE_DIR}" CACHE INTERNAL
                                                          "snappy source tree")

# snappy.h includes snappy-stubs-public.h, which snappy generates from a .in
# at configure time; generated here instead, with the four substitutions it
# has and nothing else. config.h stays undefined: every reference to it in
# these translation units sits behind `#if HAVE_CONFIG_H`, so the platform
# detection falls back to the compiler's own macros rather than to a header
# this project would have to keep true.
#
# A function, not inline code: the .in names ${PROJECT_VERSION_MAJOR} and its
# siblings, which are CMake's own variables for the enclosing project, and
# setting them for the rest of this directory to satisfy a third-party
# template is a trap for whoever reads one later.
function(cudec_generate_snappy_stubs version out_dir)
  string(REPLACE "." ";" _parts "${version}")
  list(LENGTH _parts _count)
  if(NOT _count EQUAL 3)
    message(FATAL_ERROR "the snappy pin '${version}' is not major.minor.patch "
                        "- the version macros cannot be derived from it")
  endif()
  list(GET _parts 0 PROJECT_VERSION_MAJOR)
  list(GET _parts 1 PROJECT_VERSION_MINOR)
  list(GET _parts 2 PROJECT_VERSION_PATCH)
  # 1 for the Linux container this builds in, which is where iovec lives. On a
  # host without sys/uio.h the header defines its own, so the 0 case is real
  # code and not a fallback that has to be added later.
  set(HAVE_SYS_UIO_H_01 1)
  # No @ONLY: the template's placeholders are ${...}, which is exactly what
  # the default substitution reads.
  configure_file(${snappy_SOURCE_DIR}/snappy-stubs-public.h.in
                 ${out_dir}/snappy-stubs-public.h)
endfunction()
set(_snappy_generated_dir ${CMAKE_CURRENT_BINARY_DIR}/snappy-generated)
cudec_generate_snappy_stubs(${CUDEC_SNAPPY_VERSION} ${_snappy_generated_dir})

# Two translation units, which is the whole decode path: snappy.cc, and
# snappy-sinksource.cc for the ByteArraySource the Source-taking calls read
# through. snappy-stubs-internal.cc carries none of it. SYSTEM include and
# none of the project's strict flags, the rule lz4_oracle follows for the
# same reason - third-party code is audited by hash, not reformatted.
add_library(snappy_oracle STATIC ${snappy_SOURCE_DIR}/snappy.cc
                                 ${snappy_SOURCE_DIR}/snappy-sinksource.cc)
target_include_directories(snappy_oracle SYSTEM PUBLIC ${snappy_SOURCE_DIR}
                                                       ${_snappy_generated_dir})
set_target_properties(snappy_oracle PROPERTIES CXX_STANDARD 11
                                               CXX_STANDARD_REQUIRED ON)
# -O2 for the same reason lz4_oracle carries it: the documented container
# command sets no CMAKE_BUILD_TYPE, and the reference decoder runs over whole
# corpora during fixture generation.
target_compile_options(snappy_oracle PRIVATE -O2)
