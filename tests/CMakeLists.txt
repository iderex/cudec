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

# PRIVATE on purpose: the archive still links through, but the lz4 include
# path stays out of every consumer's command line - so an accidental
# #include <lz4.h> in a .cu test is a compile error, and the "nvcc never
# sees lz4.h" rule is enforced, not just observed.
add_library(cudec_test_fixtures STATIC fixtures.cpp)
target_link_libraries(cudec_test_fixtures PRIVATE lz4_oracle)
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
  if(T_GPU)
    set_tests_properties(${name} PROPERTIES LABELS gpu RUN_SERIAL TRUE)
  endif()
endfunction()

cudec_add_test(conformance_c SOURCES conformance_c.c)
set_target_properties(conformance_c PROPERTIES C_STANDARD 99
                                               C_STANDARD_REQUIRED ON)
cudec_add_test(oracle_lz4 SOURCES oracle_lz4.cpp LIBS cudec_test_fixtures)
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
cudec_add_test(frame_host_negative SOURCES frame_host_negative.cpp)
target_include_directories(frame_host_negative
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
