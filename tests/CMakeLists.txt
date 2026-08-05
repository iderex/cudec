# The test net: ctest is the runner, one executable per test group, the
# assertion header tests/require.h as the only "framework". Decision record
# and reassessment trigger: docs/MASTERPLAN.md section 5.

# Supply chain: the hash-verified tarball is the only acceptable oracle
# source - a system copy must never silently substitute for it.
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)

# liblz4, pinned by the SHA-256 of the maintainer-uploaded release asset
# (self-computed at pin time and cross-checked against conan-center; the
# auto-generated /archive/ tarballs are avoided - GitHub regenerated them
# in 2023 and their hashes moved). Test-only dependency: the link
# allowlist below keeps it out of the library.
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

# google/snappy, the M3 oracle, pinned by the SHA-256 of the tag archive.
# snappy publishes no maintainer-uploaded release asset, so this is the
# fallback the pinning policy names (docs/MASTERPLAN.md section 5): the
# auto-generated archive, pinned, with a future hash mismatch read as the
# invariant working rather than as noise. Cross-checked at pin time against a
# second packaging ecosystem each - the SHA-256 against the Homebrew formula,
# the SHA-512
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

# PRIVATE on purpose: the archive still links through, but the lz4 and snappy
# include paths stay out of every consumer's command line - so an accidental
# #include <lz4.h> or <snappy.h> in a .cu test is a compile error, and the
# "nvcc never sees the oracle headers" rule is enforced, not just observed.
add_library(cudec_test_fixtures STATIC fixtures.cpp)
target_link_libraries(cudec_test_fixtures PRIVATE lz4_oracle snappy_oracle)
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
file(GLOB_RECURSE _batch_limit_sources RELATIVE
     ${CMAKE_CURRENT_SOURCE_DIR}/../src CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.c
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cc
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cuh
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.hpp
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.inl)
if(NOT _batch_limit_sources)
  message(FATAL_ERROR "the batch-limit sweep found no sources under src/ - the "
                      "glob has stopped matching (issue #39)")
endif()
foreach(_source IN LISTS _batch_limit_sources)
  if(NOT _source STREQUAL "batch_limits.h")
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/../src/${_source} _source_src)
    if(_source_src MATCHES "INT32_MAX")
      message(
        FATAL_ERROR
          "src/${_source} builds its own bound from INT32_MAX - the batch "
          "launch limit is cudec_detail::kMaxBatchChunks in "
          "src/batch_limits.h (issue #39)")
    endif()
  endif()
endforeach()

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
  # The two characters CMake's list syntax eats are protected across the
  # split: a ';' separates elements and a '\' escapes that separator. The
  # placeholders are control bytes, which no source in this tree carries and
  # which therefore cannot be produced by the source being scanned.
  string(ASCII 1 _tok_backslash)
  string(ASCII 2 _tok_semicolon)
  file(READ "${path}" _raw)
  # "No source carries one" is the assumption this reader rests on, so it is
  # stated to the file rather than trusted: a source that does carry one would
  # have it restored below as a backslash or as a separator, which is a way to
  # smuggle either past the split. Refuse the file instead of scanning it.
  if(_raw MATCHES "${_tok_backslash}|${_tok_semicolon}")
    string(APPEND _violations
           "  ${path}:1: a control byte (0x01 or 0x02) in a source read as "
           "text - this scan cannot tell one from its own line markers, so "
           "the file is refused rather than scanned\n")
  endif()
  string(REPLACE "\\" "${_tok_backslash}" _raw "${_raw}")
  string(REPLACE ";" "${_tok_semicolon}" _raw "${_raw}")
  string(REPLACE "\r\n" "\n" _raw "${_raw}")
  string(REPLACE "\r" "\n" _raw "${_raw}")
  # A file's final newline terminates its last line rather than starting an
  # empty one; left in, it would add a line the file does not have.
  string(REGEX REPLACE "\n$" "" _raw "${_raw}")
  string(REPLACE "\n" ";" _lines "${_raw}")
  foreach(_src_line IN LISTS _lines)
    math(EXPR _lineno "${_lineno} + 1")
    string(REPLACE "${_tok_semicolon}" ";" _src_line "${_src_line}")
    string(REPLACE "${_tok_backslash}" "\\" _src_line "${_src_line}")
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
# The byte the reader borrows for its own bookkeeping, in the file it is
# reading. It has to come back refused, or the borrowing is a way to write a
# separator or a backslash the split will not see as one.
string(ASCII 1 _probe_ctrl)
file(WRITE ${_probe_dir}/scan_control_byte.cu
     "void probe() {\n" "    g(1);${_probe_ctrl}\n" "}\n")

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
    "scan_control_byte|a control byte (0x01 or 0x02) in a source read as text"
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
